#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================
# 可配置超参数（集中修改这里）
# ==============================
MODEL_PATH = "/root/autodl-fs/wzq/models/models--SciReason--SciReasoner-8B/snapshots/772c4adaf43c750db5ef04d6f567148ca3daf7b0"
TRAIN_DATASET_PATH = "/root/autodl-fs/wzq/datasets/SynBioCoT/json/train.json"
VALIDATE_DATASET_PATH = "/root/autodl-fs/wzq/datasets/SynBioCoT/json/cell_validation.json"  # 仅保留变量，不再使用
OUTPUT_DIR = "./outputs/biomnigem-sft-lora"

# LoRA
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.1
# 新版 peft 支持 "all-linear"；若是旧版 peft，请改成逗号分隔列表：
# "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# 训练
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 1  # 已不用，但保留变量
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 4096  # 与数据构建一致

# 日志/保存（评估被禁用）
LOGGING_STEPS = 10
EVAL_STEPS = 1000000  # 已无效
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 20
LR_SCHEDULER_TYPE = "cosine"

# 生成评估（已禁用，不再使用）
PREDICT_WITH_GENERATE = False
GENERATION_MAX_LENGTH = 1024
GENERATION_NUM_BEAMS = 1

# DDP / 混精度 / 其他
DDP_FIND_UNUSED_PARAMETERS = False
GRADIENT_CHECKPOINTING = False
SEED = 42
# 混精度：None=自动选择；也可显式 True/False 强制
USE_BF16 = True
USE_FP16 = None

# ==============================
# 依赖 & 工具
# ==============================
import os
from typing import List
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from tools import (
    set_seed,
    load_tokenizer,
    get_train_validate_dataset,  # 仍复用此函数，但只取训练数据
    load_model,
    get_data_collator,
    # validate_one,  # 不再需要
)

def _auto_mp():
    """根据硬件与显式配置决定 bf16/fp16（不改变你设置的语义）"""
    bf16_supported = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if USE_BF16 is None and USE_FP16 is None:
        use_bf16 = bf16_supported
        use_fp16 = (torch.cuda.is_available() and not use_bf16)
    else:
        use_bf16 = bool(USE_BF16) if USE_BF16 is not None else False
        use_fp16 = bool(USE_FP16) if USE_FP16 is not None else (torch.cuda.is_available() and not use_bf16)
    return use_bf16, use_fp16

def _parse_target_modules(s: str):
    if isinstance(s, str) and s != "all-linear":
        return [x.strip() for x in s.split(",") if x.strip()]
    return s  # "all-linear" 或已是列表

def _maybe_set_mem_env():
    # 降低显存碎片风险（不改变训练逻辑/超参）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        # 不改数值语义，放宽 TF32（A100/H800 matmul），与 bf16 共存
        torch.backends.cuda.matmul.allow_tf32 = True
        # 在较新 PyTorch 中提高 float32 matmul 精度设置（对稳定性更友好）
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def _maybe_enable_flash_attn(model):
    # 不强制；若环境具备 flash-attn2，则开启；否则维持默认实现
    try:
        # transformers>=4.40 支持该字段；若不支持会抛异常
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

def build_trainer():
    # 0) 内存/后端小优化（不改变训练语义）
    _maybe_set_mem_env()

    # 1) 随机种子
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2) Tokenizer
    tokenizer = load_tokenizer(MODEL_PATH, padding_side="right")
    # 对齐 pad/eos，避免 padding 警告与无谓的张量扩张
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) 数据集 —— 只取训练数据；验证集/答案被忽略
    train_text, _, _ = get_train_validate_dataset(
        TRAIN_DATASET_PATH, VALIDATE_DATASET_PATH, tokenizer=tokenizer, seed=SEED
    )
    # 4) Collator（仅对 assistant 计算损失）
    collator = get_data_collator(tokenizer)

    # 5) 模型 + LoRA
    model = load_model(MODEL_PATH)
    _maybe_enable_flash_attn(model)

    target_modules = _parse_target_modules(LORA_TARGET_MODULES)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # 6) 混精度
    use_bf16, use_fp16 = _auto_mp()
    try:
        if use_bf16:
            model.to(dtype=torch.bfloat16)
    except Exception:
        # 个别后端/LoRA 包装不支持整模转换则忽略
        pass

    # 便于（可能的）GC；是否开启仍由 GRADIENT_CHECKPOINTING 控制
    if hasattr(model, "config"):
        model.config.use_cache = False

    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 7) SFT 配置：完全关闭评估，只训练；按 SAVE_STEPS 保存
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        # 仅训练，不需要 eval batch
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,

        # 🚫 关闭评估
        do_eval=False,
        eval_strategy="no",   # 关键：不评估
        # eval_steps=EVAL_STEPS,    # 不需要
        # predict_with_generate=False,

        # ✅ 只按步保存 checkpoint
        save_strategy="steps",
        save_steps=SAVE_STEPS,      # 100
        save_total_limit=SAVE_TOTAL_LIMIT,

        # 🚫 无评估就不要加载最优模型
        load_best_model_at_end=False,
        # metric_for_best_model="acc",   # 不需要
        # greater_is_better=True,        # 不需要

        lr_scheduler_type=LR_SCHEDULER_TYPE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
        bf16=use_bf16,
        fp16=(use_fp16 and not use_bf16),
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
        report_to="none",
    )

    # 8) 组装 Trainer（不传 eval_dataset / compute_metrics）
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_text,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 可训练参数量（仅主进程）
    if int(os.environ.get("RANK", "0")) == 0:
        try:
            trainable, total = 0, 0
            for _, p in model.named_parameters():
                n = p.numel()
                total += n
                if p.requires_grad:
                    trainable += n
            print(f"Trainable params: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M "
                  f"({100*trainable/max(1,total):.2f}%)")
            print(f"Train samples: {len(train_text)}")
        except Exception:
            pass

    return trainer

def main():
    trainer = build_trainer()
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(OUTPUT_DIR)   # 保存 LoRA 适配器
        trainer.tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Done. LoRA adapters & tokenizer saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
