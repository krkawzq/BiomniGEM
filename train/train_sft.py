#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiomniGEM SFT 训练脚本（优化版）
------------------------------------------------------------
核心改进：
- ✅ 评估批量支持 + 进度条（可设置 eval_batch_size）
- ✅ 推理时自动切换到左填充（decoder-only 模型要求）
- ✅ 正确的序列切分：使用固定输入长度而非 attention_mask.sum()
- ✅ 截断策略：truncation_side='right'（保留开头 system prompt）
- ✅ 本地/远程模型自动判定（local_files_only）
- ✅ generation_config 对齐 pad/eos，减少警告
- ✅ 训练前数据清洗（过滤无效样本）
- ✅ 评估批末显存清理（torch.cuda.empty_cache）
- ✅ 最优 checkpoint 管理（保留最佳 + 最近 k-1）
- ✅ 去掉 </answer> 作为停止符，仅用于答案解析
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
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

# =========================
# 常量 / 工具
# =========================

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.S)  # 仅解析，不作停止符
DEFAULT_QWEN_PATHS = [
    "/root/autodl-fs/hf-models/Qwen--Qwen3-8B",
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

def get_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else ""

def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def validate_equal(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

def apply_stop(decoded: str, stops: List[str]) -> str:
    """遇到任意 stop 字符串即截断（不包含停符本身）"""
    earliest = len(decoded)
    for s in stops or []:
        pos = decoded.find(s)
        if pos != -1:
            earliest = min(earliest, pos)
    return decoded[:earliest] if earliest != len(decoded) else decoded

def _require_columns(ds, cols: List[str], name: str):
    miss = [c for c in cols if c not in ds.column_names]
    if miss:
        raise ValueError(f"{name} 缺少字段: {miss}，现有列: {ds.column_names}")

def _ok_train_row(ex: Dict[str, Any]) -> bool:
    return isinstance(ex.get("system"), str) and isinstance(ex.get("user"), str) and isinstance(ex.get("assistant", ""), str)

def _ok_eval_row(ex: Dict[str, Any]) -> bool:
    return isinstance(ex.get("system"), str) and isinstance(ex.get("user"), str)

def detect_lora_target_modules(model) -> List[str]:
    """自动探测 + 常见子串并集，容忍命名差异"""
    names = set()
    for name, module in model.named_modules():
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            leaf = name.split(".")[-1]
            if any(k in name for k in ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj")):
                names.add(leaf)
    fixed = {"q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"}
    return sorted(set(names) | fixed)

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

def persist_run_config(args_namespace, out_dir: str):
    run_cfg = vars(args_namespace).copy()
    try:
        import transformers as _t; run_cfg["transformers"] = _t.__version__
    except Exception: pass
    try:
        import trl as _trl; run_cfg["trl"] = _trl.__version__
    except Exception: pass
    try:
        import peft as _peft; run_cfg["peft"] = _peft.__version__
    except Exception: pass
    run_cfg["torch"] = torch.__version__
    run_cfg["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

# =========================
# 数据格式化
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
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking_default
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
    return formatting_func

# =========================
# 回调：指标与最优 ckpt
# =========================

class TrainingMetricsCallback(TrainerCallback):
    def __init__(self, out_dir: str, logger: Optional[logging.Logger], local_rank: int):
        self.out_dir = out_dir
        self.logger = logger or logging.getLogger(__name__)
        self.local_rank = local_rank
        os.makedirs(self.out_dir, exist_ok=True)
        self.detail_path = os.path.join(self.out_dir, "training_metrics.jsonl")
        self.summary_path = os.path.join(self.out_dir, "training_summary.csv")
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
        self._load()

    def _load(self):
        if os.path.exists(self.checkpoint_info_path):
            try:
                with open(self.checkpoint_info_path, "r") as f:
                    data = json.load(f)
                self.checkpoints = data.get("checkpoints", [])
                self.best_checkpoint = data.get("best_checkpoint")
                self.logger.info(f"加载 checkpoint 记录：{len(self.checkpoints)}")
            except Exception as e:
                self.logger.warning(f"加载 checkpoint 信息失败: {e}")

    def _save(self):
        if not is_main_process(self.local_rank): return
        data = {"checkpoints": self.checkpoints, "best_checkpoint": self.best_checkpoint}
        with open(self.checkpoint_info_path, "w") as f:
            json.dump(data, f, indent=2)

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
        self._save()

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
            if path in keep_paths: new_ckpts.append((step, acc, path))
            else:
                try:
                    shutil.rmtree(path)
                    self.logger.info(f"删除旧 checkpoint: {path}")
                except Exception as e:
                    self.logger.warning(f"删除 checkpoint 失败 {path}: {e}")
        self.checkpoints = new_ckpts

# =========================
# 评估
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
        if is_main_process(self.local_rank) and not os.path.exists(self.summary_path):
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["global_step","n","parsed","correct","accuracy","coverage","time_sec"])

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
        interval = getattr(args, "eval_steps", None) or args.save_steps
        if state.global_step > 0 and interval and state.global_step % interval == 0:
            self.run_eval(kwargs["model"], state.global_step)

    def run_eval(self, model, global_step: int):
        if not is_main_process(self.local_rank):
            return
        self.logger.info(f"开始评估 (step {global_step}, batch_size={self.eval_batch_size})...")
        model.eval()
        device = next(model.parameters()).device

        # 临时左填充（decoder-only 批量生成需要）
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
                        inputs = self.tok(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=getattr(self.tok, "model_max_length", 4096),
                            return_attention_mask=True
                        ).to(device)

                        outputs = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            pad_token_id=self.tok.pad_token_id,
                            use_cache=True,
                            **self.gen_kwargs
                        )

                        for j, (idx, ex, gold) in enumerate(zip(batch_indices, batch_examples, batch_golds)):
                            # 切分生成部分：从输入序列长度后开始（包含padding）
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
                        self.logger.warning(f"评估批次 {batch_start}-{batch_end} 失败: {e}")
                        # 回退逐个
                        for idx in batch_indices:
                            try:
                                ex = self.val_examples[idx]
                                prompt = self._build_prompt(ex)
                                gold = self._gold(ex)
                                single = self.tok(prompt, return_tensors="pt", truncation=True,
                                                  max_length=getattr(self.tok, "model_max_length", 4096),
                                                  return_attention_mask=True).to(device)
                                outputs = model.generate(
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

                    # 批末清理显存碎片
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

        model.train()
        return acc

# =========================
# 主流程
# =========================

def main():
    parser = argparse.ArgumentParser(description="BiomniGEM SFT 训练（稳健清晰版）")
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
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=None)
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

    # 初始化
    os.makedirs(args.output_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    logger = build_logger(args.output_dir, local_rank)
    set_seed(args.seed)

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
    signal.signal(signal.SIGINT,  _graceful_exit)

    logger.info("="*60); logger.info("BiomniGEM SFT 训练（稳健清晰版）启动"); logger.info("="*60)

    # 加载 tokenizer（本地/远程自动判定）
    is_local_model = _is_local(args.model_path)
    logger.info("加载 Tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=is_local_model
    )

    # 截断与填充策略
    tok.truncation_side = "right"  # 保留开头（system prompt）
    tok.padding_side    = "right"  # 训练右填充；评估会临时改 left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        logger.info(f"设置 pad_token = eos_token ({tok.eos_token})")

    # chat_template 检查/复制
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

    # enable_thinking 探测
    supports_thinking = False
    try:
        _ = tok.apply_chat_template([{"role":"user","content":"test"}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
        supports_thinking = True
        logger.info("✓ Chat template 支持 enable_thinking")
    except TypeError:
        logger.info("⚠️ Chat template 不支持 enable_thinking")
    enable_thinking_flag = args.enable_thinking and supports_thinking

    logger.info(f"✓ Tokenizer 就绪 | 词表: {len(tok)} | EOS: {tok.eos_token} | padding_side(train)={tok.padding_side}")

    # 加载模型
    logger.info("加载模型...")
    quant_config = None
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map=None if args.deepspeed else "auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        local_files_only=is_local_model
    )
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    # 词表对齐
    if model.config.vocab_size != len(tok):
        logger.warning(f"vocab 不匹配：model={model.config.vocab_size}, tok={len(tok)}，调整 embedding")
        model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    # LoRA 注入
    logger.info("注入 LoRA...")
    target_modules = detect_lora_target_modules(model)
    logger.info(f"  LoRA 目标层: {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ LoRA 注入完成 | 可训练参数: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    # 编译/SDPA（可选）
    if args.enable_compile:
        model = try_enable_sdp_and_compile(model, logger)

    # 数据加载
    logger.info("加载数据集...")
    train_raw = load_from_disk(args.train_dir)
    val_raw   = load_from_disk(args.val_dir)
    logger.info(f"原始数据: 训练={len(train_raw)}, 验证={len(val_raw)}")

    _require_columns(train_raw, ["system","user","assistant","quality"], "train_raw")
    _require_columns(val_raw,   ["system","user"], "val_raw")

    # 清洗 + 过滤
    train_raw = train_raw.filter(_ok_train_row, num_proc=args.num_proc)
    val_raw   = val_raw.filter(_ok_eval_row,   num_proc=args.num_proc)
    logger.info(f"清洗后: 训练={len(train_raw)}, 验证={len(val_raw)}")

    logger.info("过滤训练集 (quality == 'gold')...")
    train_raw = train_raw.filter(lambda ex: ex.get("quality") == "gold", num_proc=args.num_proc)
    if len(train_raw) == 0:
        raise ValueError("过滤后训练集为空！请检查数据或去掉 quality 过滤。")

    train_ds = train_raw.map(
        make_messages,
        remove_columns=[c for c in train_raw.column_names if c != "messages"],
        num_proc=args.num_proc
    )
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        train_ds = train_ds.select(range(args.max_train_samples))
    val_ds = val_raw.map(
        keep_eval_fields,
        remove_columns=[c for c in val_raw.column_names if c not in ("id","system","user","assistant","answer")],
        num_proc=args.num_proc
    )
    if len(val_ds) == 0:
        raise ValueError("验证集为空或缺少必要字段！")

    logger.info(f"✓ 数据准备完成 | 训练={len(train_ds)}, 验证={len(val_ds)}")

    # 训练配置
    max_len = min(args.cutoff_len, getattr(tok, "model_max_length", args.cutoff_len))
    formatting_func = create_formatting_func(tok, enable_thinking_default=enable_thinking_flag)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        logging_first_step=True,
        report_to=["none"],
        max_length=max_len,
        packing=args.packing,
        deepspeed=args.deepspeed,
        max_grad_norm=1.0,  # 稳定训练
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=sft_config,
        train_dataset=train_ds,
        formatting_func=formatting_func,
    )

    # 指标回调
    metrics_cb = TrainingMetricsCallback(
        out_dir=os.path.join(args.output_dir, "metrics"),
        logger=logger,
        local_rank=local_rank
    )
    trainer.add_callback(metrics_cb)

    # 最优 ckpt 管理
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

    # 构造 eos（不含 </answer>）
    eos_ids: List[int] = []
    for tok_text in ("<|im_end|>", tok.eos_token):
        try:
            _id = tok.convert_tokens_to_ids(tok_text)
            if isinstance(_id, int) and _id != tok.unk_token_id:
                eos_ids.append(_id)
        except Exception:
            pass
    # 对齐到 generation_config
    gen_cfg = model.generation_config
    gen_cfg.pad_token_id = tok.pad_token_id
    if eos_ids:
        gen_cfg.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]

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

    # 初始评估
    if args.initial_eval and is_main_process(local_rank):
        logger.info("="*60); logger.info("初始评估（训练前）"); logger.info("="*60)
        eval_cb.run_eval(model, global_step=0)

    # 落盘运行配置与 tokenizer
    if is_main_process(local_rank):
        persist_run_config(args, args.output_dir)
        tok.save_pretrained(args.output_dir)

    # 训练
    logger.info("="*60); logger.info("开始训练..."); logger.info("="*60)
    try:
        if args.resume_from_checkpoint:
            logger.info(f"从 checkpoint 恢复: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("检测到 OOM：请降低 batch 或启用 4/8bit / gradient_checkpointing / 减小 cutoff_len")
        raise
    finally:
        if is_main_process(local_rank):
            try: trainer.save_state()
            except Exception: pass

    # 最终评估与保存
    if is_main_process(local_rank):
        logger.info("="*60); logger.info("最终评估..."); logger.info("="*60)
        eval_cb.run_eval(model, global_step=-1)
        logger.info("="*60); logger.info("保存模型与 tokenizer..."); logger.info("="*60)
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        try:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        except Exception:
            pass
        logger.info("训练完成 ✅")

if __name__ == "__main__":
    main()
