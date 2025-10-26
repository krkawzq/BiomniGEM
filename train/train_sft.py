#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BiomniGEM SFT 训练（重构并行稳健版）
------------------------------------------------------------
改动要点（接口与业务逻辑保持不变）：
- 使用 HuggingFace Trainer（Accelerate 后端）统一分布式（DDP/FSDP/DeepSpeed），移除外层 Accelerator.prepare 与 TRL SFTTrainer 的叠加冲突
- 训练数据纯 text 预分词 + DataCollatorForLanguageModeling(mlm=False) 动态填充，避免 messages/pad 报错
- 验证时临时左填充、批量 generate，维持原有 <answer> 解析、coverage/accuracy 统计、best checkpoint 选择与清理
- LoRA 注入策略与原脚本保持一致（自动探测 q/k/v/o 与 up/down/gate）
"""

import os
import re
import json
import csv
import sys
import time
import shutil
import logging
import random
import signal
import argparse
from typing import Dict, Any, List, Optional, Tuple

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import is_main_process
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# =========================
# 常量 / 工具
# =========================

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.S)  # 仅解析，不作停止符
DEFAULT_QWEN_PATHS = [
    "/root/autodl-fs/wzq/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-8B",
]


def _is_local(path: str) -> bool:
    return os.path.exists(path) or path.startswith("/")


def build_logger(out_dir: str, local_rank: int = -1) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if is_main_process(local_rank):
        fh = logging.FileHandler(os.path.join(out_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def check_disk_space(path: str, min_gb: float = 10.0) -> bool:
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        return free_gb >= min_gb
    except Exception:
        return True  # 检查失败则假设充足


def get_gpu_memory_info() -> str:
    if not torch.cuda.is_available():
        return "无 GPU"
    try:
        info_lines = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            info_lines.append(f"GPU{i}: {props.name} ({total_gb:.1f}GB)")
        return ", ".join(info_lines)
    except Exception as e:
        return f"无法获取 GPU 信息: {e}"


def persist_run_config(args_namespace, out_dir: str):
    run_cfg = vars(args_namespace).copy()
    try:
        import transformers as _t; run_cfg["transformers"] = _t.__version__
    except Exception: pass
    try:
        import peft as _peft; run_cfg["peft"] = _peft.__version__
    except Exception: pass
    run_cfg["torch"] = torch.__version__
    run_cfg["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)


# ---------- 文本与校验 ----------

def get_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else ""


def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def validate_equal(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def apply_stop(decoded: str, stops: List[str]) -> str:
    earliest = len(decoded)
    for s in stops or []:
        pos = decoded.find(s)
        if pos != -1:
            earliest = min(earliest, pos)
    return decoded[:earliest] if earliest != len(decoded) else decoded


def _require_columns(ds: Dataset, cols: List[str], name: str):
    miss = [c for c in cols if c not in ds.column_names]
    if miss:
        raise ValueError(f"{name} 缺少字段: {miss}，现有列: {ds.column_names}")


# =========================
# 参数校验
# =========================

def validate_args(args, logger: logging.Logger) -> bool:
    issues = []
    warnings = []
    if args.lora_r <= 0:
        issues.append(f"lora_r 必须 > 0，当前: {args.lora_r}")
    if args.lora_alpha <= 0:
        issues.append(f"lora_alpha 必须 > 0，当前: {args.lora_alpha}")
    if args.lr <= 0:
        issues.append(f"lr 必须 > 0，当前: {args.lr}")
    if args.epochs <= 0:
        issues.append(f"epochs 必须 > 0，当前: {args.epochs}")
    if args.per_device_train_batch_size <= 0:
        issues.append(f"per_device_train_batch_size 必须 > 0，当前: {args.per_device_train_batch_size}")
    if args.grad_accum <= 0:
        issues.append(f"grad_accum 必须 > 0，当前: {args.grad_accum}")

    if args.lora_r > 256:
        warnings.append(f"lora_r 过大 ({args.lora_r})，可能导致过拟合或显存不足")
    if args.per_device_train_batch_size > 8:
        warnings.append(f"batch_size 较大 ({args.per_device_train_batch_size})，注意显存占用")
    if args.cutoff_len > 8192:
        warnings.append(f"cutoff_len 很大 ({args.cutoff_len})，可能导致显存不足")
    if args.lr > 1e-3:
        warnings.append(f"学习率较高 ({args.lr})，可能不稳定")
    if args.warmup_ratio > 0.5:
        warnings.append(f"warmup_ratio 过大 ({args.warmup_ratio})，可能影响训练")
    if args.load_in_4bit and args.load_in_8bit:
        issues.append("不能同时启用 4bit 和 8bit 量化")

    if issues:
        for issue in issues:
            logger.error(f"参数错误: {issue}")
        raise ValueError(f"发现 {len(issues)} 个参数错误")

    if warnings:
        logger.warning("参数警告:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    return True


# =========================
# 数据格式化函数
# =========================

def make_messages(ex: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex.get("assistant", "")},
        ]
    }


def keep_eval_fields(ex: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": ex.get("id", None),
        "system": ex["system"],
        "user": ex["user"],
        "assistant": ex.get("assistant", ""),
        "answer": ex.get("answer", None),
    }


def create_formatting_func(tokenizer, enable_thinking_default: bool = True):
    """messages -> text（尽量启用思考模板；不返回 None）"""
    def formatting_func(example):
        messages = example["messages"]
        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking_default
            )
        except TypeError:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            # 回退到简易模板
            result = ""
            for msg in messages:
                if msg["role"] == "system":
                    result += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "user":
                    result += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    result += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        if not isinstance(result, str):
            result = str(result)
        return result
    return formatting_func


# =========================
# 训练过程回调
# =========================

class TrainingMetricsCallback(TrainerCallback):
    def __init__(self, out_dir: str, logger: Optional[logging.Logger], local_rank: int):
        self.out_dir = out_dir
        self.logger = logger or logging.getLogger(__name__)
        self.local_rank = local_rank
        os.makedirs(self.out_dir, exist_ok=True)
        self.detail_path = os.path.join(self.out_dir, "training_metrics.jsonl")
        self.summary_path = os.path.join(self.out_dir, "training_summary.csv")

    def on_train_begin(self, args, state, control, **kwargs):
        if is_main_process(self.local_rank) and not os.path.exists(self.summary_path):
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["global_step","epoch","loss","learning_rate","grad_norm","timestamp"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not is_main_process(self.local_rank):
            return
        entry = {"global_step": state.global_step, "epoch": state.epoch, **logs, "timestamp": time.time()}
        with open(self.detail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if "loss" in logs:
            with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    state.global_step,
                    f"{state.epoch:.2f}" if state.epoch is not None else "",
                    f"{logs.get('loss',''):.6f}" if 'loss' in logs else "",
                    f"{logs.get('learning_rate',''):.2e}" if 'learning_rate' in logs else "",
                    f"{logs.get('grad_norm',''):.4f}" if 'grad_norm' in logs else "",
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class BestCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, save_total_limit: int, metric_name: str, logger: logging.Logger, local_rank: int):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.metric_name = metric_name
        self.logger = logger
        self.local_rank = local_rank
        self.checkpoint_info_path = os.path.join(output_dir, "checkpoint_info.json")
        self.checkpoints: List[Tuple[int, float, str]] = []
        self.best_checkpoint: Optional[Tuple[int, float, str]] = None

    def _save_state(self):
        if not is_main_process(self.local_rank): return
        data = {"checkpoints": self.checkpoints, "best_checkpoint": self.best_checkpoint}
        with open(self.checkpoint_info_path, "w") as f:
            json.dump(data, f, indent=2)

    def _cleanup(self):
        if len(self.checkpoints) <= self.save_total_limit:
            return
        self.checkpoints.sort(key=lambda x: x[0])
        keep_paths = set()
        if self.best_checkpoint:
            keep_paths.add(self.best_checkpoint[2])
        recent_limit = max(0, self.save_total_limit - 1)
        for _, _, p in self.checkpoints[-recent_limit:]:
            keep_paths.add(p)
        new_ckpts = []
        for step, acc, path in self.checkpoints:
            if path in keep_paths:
                new_ckpts.append((step, acc, path))
            else:
                try:
                    if os.path.exists(path):
                        shutil.rmtree(path)
                        self.logger.info(f"删除旧 checkpoint: {path}")
                except Exception as e:
                    self.logger.warning(f"删除 checkpoint 失败 {path}: {e}")
                    new_ckpts.append((step, acc, path))
        self.checkpoints = new_ckpts

    def update(self, step: int, metric_value: float):
        if not is_main_process(self.local_rank): return
        path = os.path.join(self.output_dir, f"checkpoint-{step}")
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint 不存在: {path}")
            return
        self.checkpoints.append((step, metric_value, path))
        if self.best_checkpoint is None or metric_value > self.best_checkpoint[1]:
            self.best_checkpoint = (step, metric_value, path)
            self.logger.info(f"🏆 新最佳 checkpoint: step={step}, {self.metric_name}={metric_value:.4f}")
        self._cleanup()
        self._save_state()


# =========================
# 评估（作为回调触发）
# =========================

class EvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        val_examples: List[Dict[str, Any]],
        out_dir: str,
        logger: logging.Logger,
        checkpoint_cb: Optional[BestCheckpointCallback],
        local_rank: int,
        max_eval_samples: int = 256,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        stop_tokens: Optional[List[str]] = None,
        shuffle_eval_each_time: bool = True,
        eval_batch_size: int = 1,
    ):
        self.tok = tokenizer
        self.val_examples = val_examples
        self.out_dir = out_dir
        self.logger = logger
        self.checkpoint_cb = checkpoint_cb
        self.local_rank = local_rank
        self.max_eval_samples = max_eval_samples
        self.eval_batch_size = eval_batch_size
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 768,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        self.stop_tokens = stop_tokens or ["<|im_end|>"]  # 不含 </answer>
        os.makedirs(self.out_dir, exist_ok=True)
        self.detail_path = os.path.join(self.out_dir, "eval_detail.jsonl")
        self.summary_path = os.path.join(self.out_dir, "eval_summary.csv")
        self.shuffle_eval_each_time = shuffle_eval_each_time

    def _build_prompt(self, ex: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
        ]
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _gold(self, ex: Dict[str, Any]) -> str:
        if "answer" in ex and ex["answer"]:
            return str(ex["answer"]).strip()
        return get_answer(ex.get("assistant", ""))

    def on_step_end(self, args, state, control, **kwargs):
        # 兼容 Trainer 的 save_steps/eval_steps：此处与 save_steps 对齐触发
        interval = getattr(args, "eval_steps", None) or args.save_steps
        if state.global_step > 0 and interval and state.global_step % interval == 0:
            model = kwargs["model"]
            self.run_eval(model, state.global_step)

    def run_eval(self, model, global_step: int):
        # 仅主进程评估，避免多进程重复写文件
        if not is_main_process(self.local_rank):
            return
        self.logger.info(f"开始评估 (step {global_step}, batch_size={self.eval_batch_size})...")
        model.eval()

        # unwrap（FSDP/DeepSpeed 环境）
        try:
            from transformers.trainer_utils import unwrap_model
            core_model = unwrap_model(model)
        except Exception:
            core_model = getattr(model, "module", model)

        device = next(core_model.parameters()).device

        # decoder-only 批量生成时临时左填充
        original_padding_side = self.tok.padding_side
        self.tok.padding_side = 'left'

        try:
            n_total = len(self.val_examples)
            n = min(self.max_eval_samples, n_total)
            indices = list(range(n_total))
            if self.shuffle_eval_each_time:
                random.shuffle(indices)
            indices = indices[:n]

            stats = {"n": 0, "parsed": 0, "correct": 0}
            t0 = time.time()

            # 首次创建 summary 表头
            if not os.path.exists(self.summary_path):
                with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["global_step","n","parsed","correct","accuracy","coverage","time_sec"])

            with open(self.detail_path, "a", encoding="utf-8") as fout, torch.inference_mode():
                pbar = tqdm(total=len(indices), desc=f"Eval@{global_step}", unit="sample",
                            ncols=100, disable=not is_main_process(self.local_rank))
                for batch_start in range(0, len(indices), self.eval_batch_size):
                    batch_end = min(batch_start + self.eval_batch_size, len(indices))
                    batch_indices = indices[batch_start:batch_end]
                    batch_examples = [self.val_examples[idx] for idx in batch_indices]
                    batch_prompts = [self._build_prompt(ex) for ex in batch_examples]
                    batch_golds = [self._gold(ex) for ex in batch_examples]

                    try:
                        eval_max_len = min(getattr(self.tok, "model_max_length", 4096), 4096)
                        inputs = self.tok(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=eval_max_len,
                            return_attention_mask=True
                        ).to(device)

                        outputs = core_model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            pad_token_id=self.tok.pad_token_id,
                            use_cache=True,
                            **self.gen_kwargs
                        )

                        for j, (idx, ex, gold) in enumerate(zip(batch_indices, batch_examples, batch_golds)):
                            input_len = inputs["input_ids"].shape[1]
                            gen_ids = outputs[j][input_len:]
                            decoded = self.tok.decode(gen_ids, skip_special_tokens=True)
                            decoded = apply_stop(decoded, self.stop_tokens)

                            pred = get_answer(decoded)
                            parsed = pred != ""
                            stats["n"] += 1
                            if parsed:
                                stats["parsed"] += 1
                                if validate_equal(pred, gold):
                                    stats["correct"] += 1

                            fout.write(json.dumps({
                                "id": ex.get("id", idx),
                                "prompt_preview": (batch_prompts[j][:200] + "...") if len(batch_prompts[j]) > 200 else batch_prompts[j],
                                "raw_output": decoded,
                                "pred_answer": pred,
                                "gold_answer": gold,
                                "parsed": parsed,
                                "correct": parsed and validate_equal(pred, gold),
                                "global_step": global_step,
                            }, ensure_ascii=False) + "\n")

                        pbar.update(len(batch_indices))
                        pbar.set_postfix({"acc": f"{stats['correct']/max(1,stats['n']):.3f}",
                                          "cov": f"{stats['parsed']/max(1,stats['n']):.3f}"})
                    except Exception as e:
                        err_msg = str(e)
                        self.logger.warning(f"评估批次 {batch_start}-{batch_end} 失败: {err_msg}")
                        if "out of memory" in err_msg.lower():
                            self.logger.warning("建议: 减小 --eval_batch_size 或 --cutoff_len")
                        # 逐个回退
                        for idx in batch_indices:
                            try:
                                ex = self.val_examples[idx]
                                prompt = self._build_prompt(ex)
                                gold = self._gold(ex)
                                eval_max_len = min(getattr(self.tok, "model_max_length", 4096), 4096)
                                single = self.tok(prompt, return_tensors="pt", truncation=True,
                                                  max_length=eval_max_len,
                                                  return_attention_mask=True).to(device)
                                outputs = core_model.generate(
                                    **single,
                                    pad_token_id=self.tok.pad_token_id,
                                    use_cache=True,
                                    **self.gen_kwargs
                                )
                                input_len = single["input_ids"].shape[1]
                                gen_ids = outputs[0][input_len:]
                                decoded = self.tok.decode(gen_ids, skip_special_tokens=True)
                                decoded = apply_stop(decoded, self.stop_tokens)

                                pred = get_answer(decoded)
                                parsed = pred != ""
                                stats["n"] += 1
                                if parsed:
                                    stats["parsed"] += 1
                                    if validate_equal(pred, gold):
                                        stats["correct"] += 1

                                fout.write(json.dumps({
                                    "id": ex.get("id", idx),
                                    "prompt_preview": (prompt[:200] + "...") if len(prompt) > 200 else prompt,
                                    "raw_output": decoded,
                                    "pred_answer": pred,
                                    "gold_answer": gold,
                                    "parsed": parsed,
                                    "correct": parsed and validate_equal(pred, gold),
                                    "global_step": global_step,
                                }, ensure_ascii=False) + "\n")
                            except Exception as e2:
                                self.logger.warning(f"  单样本 {idx} 失败: {e2}")
                                continue
                        pbar.update(len(batch_indices))

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                pbar.close()

            acc = stats["correct"] / max(1, stats["n"])
            cov = stats["parsed"] / max(1, stats["n"])
            elapsed = time.time() - t0
            with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([global_step, stats["n"], stats["parsed"], stats["correct"], f"{acc:.4f}", f"{cov:.4f}", f"{elapsed:.1f}"])
            self.logger.info(f"[Eval step {global_step}] acc={acc:.4f}, coverage={cov:.4f}, n={stats['n']}, time={elapsed:.1f}s")

            if self.checkpoint_cb and global_step > 0:
                self.checkpoint_cb.update(global_step, acc)

        finally:
            self.tok.padding_side = original_padding_side
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model.train()
        return acc


# =========================
# 模型 / Tokenizer / LoRA
# =========================

def try_enable_sdp_and_compile(model, logger):
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        logger.info("✓ 已启用 SDPA（Flash/MemEfficient）")
    except Exception as e:
        logger.warning(f"SDPA 启用失败：{e}")
    try:
        model = torch.compile(model, mode="max-autotune")
        logger.info("✓ 已启用 torch.compile(max-autotune)")
    except Exception as e:
        logger.warning(f"torch.compile 启用失败：{e}")
    return model


def detect_lora_target_modules(model) -> List[str]:
    names = set()
    for name, module in model.named_modules():
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            leaf = name.split(".")[-1]
            if any(k in name for k in ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj")):
                names.add(leaf)
    fixed = {"q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"}
    return sorted(set(names) | fixed)


def load_tokenizer(args, logger) -> Any:
    logger.info("加载 Tokenizer...")
    is_local_model = _is_local(args.model_path)
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=is_local_model
    )
    tok.truncation_side = "right"
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        logger.info(f"设置 pad_token = eos_token ({tok.eos_token})")

    if not getattr(tok, "chat_template", None):
        logger.warning(f"模型 {args.model_path} 缺少 chat_template，尝试从 Qwen3-8B 复制...")
        copied = False
        for cand in DEFAULT_QWEN_PATHS:
            try:
                qtok = AutoTokenizer.from_pretrained(
                    cand, use_fast=True, trust_remote_code=True,
                    local_files_only=_is_local(cand)
                )
                if getattr(qtok, "chat_template", None):
                    tok.chat_template = qtok.chat_template
                    logger.info(f"✓ 已从 {cand} 复制 chat_template")
                    copied = True
                    break
            except Exception as e:
                logger.warning(f"从 {cand} 复制失败: {e}")
        if not copied:
            raise ValueError("无法获取 chat_template；请确保本地或网络可用的 Qwen3-8B 存在。")

    supports_thinking = False
    try:
        _ = tok.apply_chat_template([{"role": "user", "content": "test"}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
        supports_thinking = True
        logger.info("✓ Chat template 支持 enable_thinking")
    except TypeError:
        logger.info("⚠️ Chat template 不支持 enable_thinking")

    return tok, supports_thinking


def load_model_with_lora(args, tok, logger):
    logger.info("加载模型...")
    is_local_model = _is_local(args.model_path)
    quant_config = None
    if args.load_in_4bit or args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            logger.info(f"✓ 启用量化: {'4bit' if args.load_in_4bit else '8bit'}")
        except ImportError:
            logger.error("量化需要安装 bitsandbytes 库: pip install bitsandbytes")
            raise

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else "auto",
            trust_remote_code=True,
            quantization_config=quant_config,
            local_files_only=is_local_model
        )
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            logger.error("建议: 1) 启用量化 --load_in_4bit, 2) 使用 DeepSpeed/FSDP, 3) 减少 batch size")
        raise

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    if model.config.vocab_size != len(tok):
        logger.warning(f"vocab 不匹配：model={model.config.vocab_size}, tok={len(tok)}，调整 embedding")
        model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    logger.info("注入 LoRA...")
    target_modules = detect_lora_target_modules(model)
    logger.info(f"  检测到的 LoRA 目标层: {target_modules if target_modules else 'None'}")
    if not target_modules:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        logger.info(f"  使用默认目标层: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        logger.error(f"LoRA 注入失败: {e}")
        logger.error(f"尝试的目标模块: {target_modules}")
        raise

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ LoRA 注入完成 | 可训练参数: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    model.enable_input_require_grads()
    logger.info("✓ 已启用输入梯度（gradient checkpointing 支持）")

    if args.enable_compile:
        model = try_enable_sdp_and_compile(model, logger)

    return model


# =========================
# 数据准备（训练集 text → tokenized）
# =========================

def prepare_datasets(args, tok, supports_thinking: bool, logger):
    logger.info("加载数据集...")
    train_raw = load_from_disk(args.train_dir)
    val_raw = load_from_disk(args.val_dir)
    logger.info(f"原始数据: 训练={len(train_raw)}, 验证={len(val_raw)}")

    _require_columns(train_raw, ["system", "user", "assistant"], "train_raw")
    _require_columns(val_raw, ["system", "user"], "val_raw")

    def _ok_train_row(ex: Dict[str, Any]) -> bool:
        return isinstance(ex.get("system"), str) and isinstance(ex.get("user"), str) and isinstance(ex.get("assistant", ""), str)

    def _ok_eval_row(ex: Dict[str, Any]) -> bool:
        return isinstance(ex.get("system"), str) and isinstance(ex.get("user"), str)

    train_raw = train_raw.filter(_ok_train_row, num_proc=args.num_proc)
    val_raw = val_raw.filter(_ok_eval_row, num_proc=args.num_proc)
    logger.info(f"清洗后: 训练={len(train_raw)}, 验证={len(val_raw)}")

    if args.filter_quality and args.filter_quality.strip():
        if "quality" not in train_raw.column_names:
            logger.warning(f"⚠️ 数据集缺少 'quality' 字段，跳过过滤")
        else:
            logger.info(f"过滤训练集 (quality == '{args.filter_quality}')...")
            train_raw = train_raw.filter(lambda ex: ex.get("quality") == args.filter_quality, num_proc=args.num_proc)
            if len(train_raw) == 0:
                raise ValueError(f"过滤后训练集为空！quality='{args.filter_quality}' 不存在。")
    else:
        logger.info("跳过 quality 过滤（使用全部训练数据）")

    # 训练：messages -> text（仅保留 text 字段）
    train_ds = train_raw.map(make_messages, remove_columns=train_raw.column_names, num_proc=args.num_proc)
    enable_thinking_flag = args.enable_thinking and supports_thinking
    formatting = create_formatting_func(tok, enable_thinking_default=enable_thinking_flag)

    def _to_text(example: Dict[str, Any]) -> Dict[str, str]:
        return {"text": formatting(example)}

    train_text = train_ds.map(_to_text, remove_columns=train_ds.column_names, num_proc=args.num_proc)
    if args.max_train_samples and args.max_train_samples < len(train_text):
        train_text = train_text.select(range(args.max_train_samples))

    # 训练：预分词（动态 padding 更高效，此处不做 max_length 填充）
    max_len = min(args.cutoff_len, getattr(tok, "model_max_length", args.cutoff_len))

    def tokenize_function(ex):
        out = tok(
            ex["text"],
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        # labels = input_ids（标准 Causal LM）
        out["labels"] = out["input_ids"].copy()
        return out

    train_tokenized = train_text.map(
        tokenize_function,
        batched=False,
        remove_columns=train_text.column_names,
        num_proc=args.num_proc
    )

    # 验证：保留结构化字段供评估构造 prompt
    val_ds = val_raw.map(
        keep_eval_fields,
        remove_columns=[c for c in val_raw.column_names if c not in ("id", "system", "user", "assistant", "answer")],
        num_proc=args.num_proc
    )
    if len(val_ds) == 0:
        raise ValueError("验证集为空或缺少必要字段！")

    logger.info(f"✓ 数据准备完成 | 训练(tokenized)={len(train_tokenized)}, 验证={len(val_ds)}")
    return train_tokenized, val_ds, enable_thinking_flag


# =========================
# 主流程
# =========================

def main():
    parser = argparse.ArgumentParser(description="BiomniGEM SFT 训练（并行稳健版）")
    # 基础
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=96)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # 优化
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    # 训练控制
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # 日志/保存/评估
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--initial_eval", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples_arg", type=int, default=256)
    # 数据
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packing", action="store_true")  # 接口保留（本实现不启用 packing，避免跨样本拼接带来的边界效应）
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--filter_quality", type=str, default="gold", help="过滤训练数据的 quality 值，设为空字符串则不过滤")
    # 生成/停止符（不含 </answer>）
    parser.add_argument("--extra_stop_tokens", type=str, nargs="*", default=None)
    # 思考模板
    parser.add_argument("--enable_thinking", action="store_true")
    # 量化加载（可选）
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    # 编译/SDPA
    parser.add_argument("--enable_compile", action="store_true")

    args = parser.parse_args()

    # 进程信息
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 初始化
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    logger = build_logger(args.output_dir, local_rank)

    # 参数校验
    validate_args(args, logger)

    # bf16 可用性
    if args.bf16 and not torch.cuda.is_available():
        logger.warning("bf16 指定但未检测到 CUDA，自动关闭 bf16")
        args.bf16 = False

    # 优雅退出
    def _graceful_exit(signum, frame):
        if is_main_process(local_rank):
            logger.warning(f"收到信号 {signum}，保存状态后退出...")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    logger.info("=" * 60); logger.info("BiomniGEM SFT 训练（并行稳健版）启动"); logger.info("=" * 60)
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU 信息: {get_gpu_memory_info()}")

    # 路径与磁盘
    is_local_model = _is_local(args.model_path)
    if is_local_model and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"本地模型路径不存在: {args.model_path}")
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"训练数据路径不存在: {args.train_dir}")
    if not os.path.exists(args.val_dir):
        raise FileNotFoundError(f"验证数据路径不存在: {args.val_dir}")

    if not check_disk_space(args.output_dir, min_gb=10.0):
        logger.warning("⚠️ 输出目录磁盘空间不足 10GB，训练可能失败")

    # Tokenizer
    tok, supports_thinking = load_tokenizer(args, logger)
    logger.info(f"✓ Tokenizer 就绪 | 词表: {len(tok)} | EOS: {tok.eos_token} | padding_side(train)={tok.padding_side}")

    # 模型+LoRA
    model = load_model_with_lora(args, tok, logger)
    logger.info(f"✓ 模型加载完成 | dtype: {model.dtype}")

    # 数据
    train_tokenized, val_ds, enable_thinking_flag = prepare_datasets(args, tok, supports_thinking, logger)

    # 配置摘要
    logger.info("=" * 60)
    logger.info("训练配置摘要:")
    logger.info(f"  模型: {args.model_path}")
    logger.info(f"  输出: {args.output_dir}")
    logger.info(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"  优化: lr={args.lr}, weight_decay={args.weight_decay}, warmup={args.warmup_ratio}")
    logger.info(f"  训练: epochs={args.epochs}, batch={args.per_device_train_batch_size}, grad_accum={args.grad_accum}")
    logger.info(f"  数据: 训练样本={len(train_tokenized)}, 验证样本={len(val_ds)}, cutoff={args.cutoff_len}")
    logger.info(f"  精度: bf16={args.bf16}, 4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}")
    logger.info(f"  其他: packing={args.packing}, thinking={args.enable_thinking}, compile={args.enable_compile}")
    logger.info(f"  并行: world_size={world_size}, local_rank={local_rank}")
    logger.info("=" * 60)

    # DataCollator（动态 padding，更省显存）
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Trainer/Accelerate 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="no",     # 我们用自定义 EvalCallback 定时评估
        save_total_limit=args.save_total_limit,
        logging_first_step=True,
        report_to=[],                 # 关闭 wandb 等外部日志
        dataloader_pin_memory=True,
        deepspeed=args.deepspeed,     # 可选：传入 deepspeed 配置 json 路径
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        tokenizer=tok,
        data_collator=data_collator,
    )

    # 回调：训练指标 / 最优 ckpt
    metrics_cb = TrainingMetricsCallback(
        out_dir=os.path.join(args.output_dir, "metrics"),
        logger=logger,
        local_rank=local_rank
    )
    trainer.add_callback(metrics_cb)

    ckpt_cb = BestCheckpointCallback(
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        metric_name="eval_accuracy",
        logger=logger,
        local_rank=local_rank
    )
    trainer.add_callback(ckpt_cb)

    # 评估准备
    val_examples = [val_ds[i] for i in range(len(val_ds))]
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.per_device_train_batch_size
    logger.info(f"评估批次大小: {eval_batch_size}")

    # 生成配置
    eos_ids: List[int] = []
    for tok_text in ("<|im_end|>", tok.eos_token):
        try:
            _id = tok.convert_tokens_to_ids(tok_text)
            if isinstance(_id, int) and _id != tok.unk_token_id:
                eos_ids.append(_id)
        except Exception:
            pass
    gen_kwargs = {
        "max_new_tokens": 768,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    }
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

    stop_tokens = args.extra_stop_tokens if args.extra_stop_tokens is not None else ["<|im_end|>"]

    eval_cb = EvalCallback(
        tokenizer=tok,
        val_examples=val_examples,
        out_dir=os.path.join(args.output_dir, "eval"),
        logger=logger,
        checkpoint_cb=ckpt_cb,
        local_rank=local_rank,
        max_eval_samples=args.max_eval_samples_arg,
        gen_kwargs=gen_kwargs,
        stop_tokens=stop_tokens,
        shuffle_eval_each_time=True,
        eval_batch_size=eval_batch_size,
    )
    trainer.add_callback(eval_cb)

    # 初始评估（可选，仅主进程）
    if args.initial_eval and is_main_process(local_rank):
        logger.info("=" * 60); logger.info("初始评估（训练前）"); logger.info("=" * 60)
        eval_cb.run_eval(trainer.model, global_step=0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ 已清理评估后的显存缓存")

    # 落盘运行配置与 tokenizer（仅主进程）
    if is_main_process(local_rank):
        persist_run_config(args, args.output_dir)
        tok.save_pretrained(args.output_dir)

    # 打印训练规模估计（信息性，不影响并行）
    total_train_samples = len(train_tokenized)
    per_device_batch = args.per_device_train_batch_size
    grad_accum = args.grad_accum
    effective_batch = per_device_batch * grad_accum * max(1, world_size)
    steps_per_epoch = total_train_samples // max(1, effective_batch)
    total_steps = steps_per_epoch * args.epochs
    if is_main_process(local_rank):
        logger.info(f"训练进度估计: {steps_per_epoch} steps/epoch × {args.epochs} epochs = {total_steps} total steps")
        logger.info(f"有效 batch size: {per_device_batch} × {grad_accum} × {max(1,world_size)} GPU = {effective_batch}")

    # 训练
    logger.info("=" * 60); logger.info("开始训练..."); logger.info("=" * 60)
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except RuntimeError as e:
        err_msg = str(e)
        if "CUDA out of memory" in err_msg or "out of memory" in err_msg.lower():
            if is_main_process(local_rank):
                logger.error("=" * 60)
                logger.error("检测到显存不足 (OOM)")
                logger.error("=" * 60)
                logger.error("建议措施（按优先级）：")
                logger.error("  1. 减小 batch size: --per_device_train_batch_size 1")
                logger.error("  2. 增加梯度累积: --grad_accum 64")
                logger.error("  3. 启用 4bit 量化: --load_in_4bit")
                logger.error("  4. 减小序列长度: --cutoff_len 2048")
                logger.error("  5. 减小 LoRA rank: --lora_r 64")
                logger.error(f"  当前配置: batch={per_device_batch}, grad_accum={grad_accum}, cutoff={args.cutoff_len}, lora_r={args.lora_r}")
                logger.error("=" * 60)
        raise
    except KeyboardInterrupt:
        if is_main_process(local_rank):
            logger.warning("训练被用户中断，尝试保存状态...")
            try:
                trainer.save_state()
                logger.info("✓ 状态已保存")
            except Exception as e:
                logger.warning(f"状态保存失败: {e}")
        raise
    finally:
        if is_main_process(local_rank):
            try:
                trainer.save_state()
            except Exception:
                pass

    # 最终评估与保存（仅主进程）
    if is_main_process(local_rank):
        logger.info("=" * 60); logger.info("最终评估..."); logger.info("=" * 60)
        eval_cb.run_eval(trainer.model, global_step=-1)
        logger.info("=" * 60); logger.info("保存模型与 tokenizer..."); logger.info("=" * 60)
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        try:
            if hasattr(trainer.model, "save_pretrained"):
                trainer.model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        except Exception:
            pass
        logger.info("训练完成 ✅")


if __name__ == "__main__":
    main()
