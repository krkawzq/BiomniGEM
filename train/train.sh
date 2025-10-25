#!/usr/bin/env bash
# ============================================================
# BiomniGEM LoRA-SFT 训练启动脚本
# 双卡 5090 优化配置 (32GB × 2)
# 支持批量评估加速
# ============================================================

set -e  # 出错即退出
set -o pipefail

# -------------------------------
# 基本路径配置（按需修改）
# -------------------------------
MODEL_PATH="/root/autodl-fs/hf-models/Base-SR"
TRAIN_DIR="/root/autodl-fs/hf-datasets/SynBioCoT/arrow/train"
VAL_DIR="/root/autodl-fs/hf-datasets/SynBioCoT/arrow/cell_validation"
OUTPUT_DIR="/root/autodl-fs/wzq/BiomniGEM/experiment/test-$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}"

# -------------------------------
# LoRA 参数 (优化: 增大容量)
# -------------------------------
LORA_R=128              # 从96→128 (显存充足,提升模型容量)
LORA_ALPHA=256          # 保持 alpha=2×r
LORA_DROPOUT=0.05

# -------------------------------
# 训练超参数 (优化: 双卡配置)
# -------------------------------
LR=2e-4
WEIGHT_DECAY=0.05
EPOCHS=3                # 从2→3 (16k数据需要更多训练)
PER_DEVICE_BATCH=4      # 从2→4 (32GB显存可支持)
GRAD_ACCUM=16           # 从32→16 (有效batch=4×2卡×16=128)
CUTOFF_LEN=4096
WARMUP_RATIO=0.1        # 从0.05→0.1 (更平滑的预热)
SEED=42

# -------------------------------
# 日志与保存 (优化: 每epoch保存)
# -------------------------------
SAVE_STEPS=125          # 从400→125 (16k÷128≈125 steps/epoch)
LOGGING_STEPS=20
SAVE_TOTAL_LIMIT=5      # 从3→5 (保留更多checkpoint)

# -------------------------------
# 评估配置 (优化: 更全面评估)
# -------------------------------
INITIAL_EVAL=true          # 是否训练前先评估一次
MAX_TRAIN_SAMPLES=0        # 0 表示不限制
MAX_EVAL_SAMPLES=512       # 从256→512 (更全面的评估)
EVAL_BATCH_SIZE=8          # 评估批次大小（推理无梯度，可设置比训练更大）
NUM_PROC=4

# -------------------------------
# 高级配置（可选）
# -------------------------------
BF16=true                  # bfloat16 训练
DEEPSPEED=""               # 可指定配置文件，如 "ds_config.json"
PACKING=false              # 启用样本打包（多数情况不建议）
ENABLE_THINKING=true       # 启用 Qwen3 思考模式
ENABLE_COMPILE=true        # 启用 torch.compile + SDPA
LOAD_IN_4BIT=false         # 启用 4bit 量化加载
LOAD_IN_8BIT=false         # 启用 8bit 量化加载
RESUME_FROM_CKPT=""        # 可指定 checkpoint 恢复路径
EXTRA_STOP_TOKENS="<|im_end|>"  # 附加停止符（不会当作 eos）

# ============================================================
# 构建命令
# ============================================================

CMD="python3 train_sft.py \
  --model_path ${MODEL_PATH} \
  --train_dir ${TRAIN_DIR} \
  --val_dir ${VAL_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --epochs ${EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH} \
  --grad_accum ${GRAD_ACCUM} \
  --cutoff_len ${CUTOFF_LEN} \
  --warmup_ratio ${WARMUP_RATIO} \
  --save_steps ${SAVE_STEPS} \
  --logging_steps ${LOGGING_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --num_proc ${NUM_PROC} \
  --max_eval_samples_arg ${MAX_EVAL_SAMPLES} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --seed ${SEED} \
  --packing ${PACKING}"

# 可选项拼接
if [ "${BF16}" = true ]; then CMD="${CMD} --bf16"; fi
if [ "${INITIAL_EVAL}" = true ]; then CMD="${CMD} --initial_eval"; fi
if [ -n "${DEEPSPEED}" ]; then CMD="${CMD} --deepspeed ${DEEPSPEED}"; fi
if [ "${ENABLE_THINKING}" = true ]; then CMD="${CMD} --enable_thinking"; fi
if [ "${ENABLE_COMPILE}" = true ]; then CMD="${CMD} --enable_compile"; fi
if [ "${LOAD_IN_4BIT}" = true ]; then CMD="${CMD} --load_in_4bit"; fi
if [ "${LOAD_IN_8BIT}" = true ]; then CMD="${CMD} --load_in_8bit"; fi
if [ -n "${RESUME_FROM_CKPT}" ]; then CMD="${CMD} --resume_from_checkpoint ${RESUME_FROM_CKPT}"; fi
if [ -n "${EXTRA_STOP_TOKENS}" ]; then CMD="${CMD} --extra_stop_tokens '${EXTRA_STOP_TOKENS}'"; fi
if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then CMD="${CMD} --max_train_samples ${MAX_TRAIN_SAMPLES}"; fi

# ============================================================
# 启动训练
# ============================================================

echo "============================================================"
echo "🚀 启动 BiomniGEM SFT LoRA 训练 (双卡5090优化配置)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
echo "训练配置:"
echo "  - 训练 Batch Size: ${PER_DEVICE_BATCH} × 2卡 × ${GRAD_ACCUM} = $((PER_DEVICE_BATCH * 2 * GRAD_ACCUM))"
echo "  - 评估 Batch Size: ${EVAL_BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "  - 每 ${SAVE_STEPS} steps 保存/评估"
echo "  - 预计训练时间: 2-3小时"
echo "============================================================"
echo "执行命令："
echo "${CMD}"
echo "============================================================"

# 运行命令
eval ${CMD}