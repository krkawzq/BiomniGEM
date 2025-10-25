#!/usr/bin/env bash
# ============================================================
# BiomniGEM LoRA-SFT 训练测试脚本
# 用于快速验证训练流程（小数据量、快速迭代）
# ============================================================

set -e  # 出错即退出
set -o pipefail

# -------------------------------
# 基本路径配置
# -------------------------------
MODEL_PATH="/root/autodl-fs/hf-models/Base-SR"
TRAIN_DIR="/root/autodl-fs/hf-datasets/SynBioCoT/arrow/train"
VAL_DIR="/root/autodl-fs/hf-datasets/SynBioCoT/arrow/cell_validation"
OUTPUT_DIR="/root/autodl-fs/wzq/BiomniGEM/experiment/test-$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}"

# -------------------------------
# LoRA 参数 (测试配置：小容量)
# -------------------------------
LORA_R=32              # 小 rank 快速测试
LORA_ALPHA=64          # alpha=2×r
LORA_DROPOUT=0.05

# -------------------------------
# 训练超参数 (测试配置：快速迭代)
# -------------------------------
LR=2e-4
WEIGHT_DECAY=0.05
EPOCHS=1                # 只训练1个epoch
PER_DEVICE_BATCH=32      # 小批次
GRAD_ACCUM=4            # 小累积（有效batch=2×2×4=16）
CUTOFF_LEN=2048         # 短序列长度
WARMUP_RATIO=0.05
SEED=42

# -------------------------------
# 日志与保存 (测试配置：频繁保存)
# -------------------------------
SAVE_STEPS=50           # 频繁保存用于测试
LOGGING_STEPS=5         # 频繁日志
SAVE_TOTAL_LIMIT=2      # 只保留2个checkpoint

# -------------------------------
# 评估配置 (测试配置：小样本快速评估)
# -------------------------------
INITIAL_EVAL=true          # 测试初始评估
MAX_TRAIN_SAMPLES=1024      # 只用500个样本训练
MAX_EVAL_SAMPLES=50        # 只用50个样本评估
EVAL_BATCH_SIZE=48          # 小批次评估
NUM_PROC=4

# -------------------------------
# 高级配置
# -------------------------------
BF16=true                  # bfloat16 训练
DEEPSPEED=""               # 测试不用 DeepSpeed
PACKING=false
ENABLE_THINKING=true
ENABLE_COMPILE=false       # 测试时关闭 compile（避免编译开销）
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false
RESUME_FROM_CKPT=""
EXTRA_STOP_TOKENS="<|im_end|>"

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
  --max_train_samples ${MAX_TRAIN_SAMPLES} \
  --seed ${SEED}"

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

# ============================================================
# 启动测试训练
# ============================================================

echo "============================================================"
echo "🧪 BiomniGEM SFT 快速测试模式"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
echo "测试配置 (快速验证流程):"
echo "  - 训练样本: ${MAX_TRAIN_SAMPLES}"
echo "  - 评估样本: ${MAX_EVAL_SAMPLES}"
echo "  - 训练 Batch: ${PER_DEVICE_BATCH} × 2卡 × ${GRAD_ACCUM} = $((PER_DEVICE_BATCH * 2 * GRAD_ACCUM))"
echo "  - 评估 Batch: ${EVAL_BATCH_SIZE}"
echo "  - Epoch: ${EPOCHS}"
echo "  - LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "  - 序列长度: ${CUTOFF_LEN}"
echo "  - 每 ${SAVE_STEPS} steps 保存/评估"
echo "  - 预计时间: 5-10分钟"
echo "============================================================"
echo "⚠️  这是测试配置，不适用于正式训练"
echo "============================================================"
echo "执行命令："
echo "${CMD}"
echo "============================================================"

# 运行命令
eval ${CMD}

echo ""
echo "============================================================"
echo "✅ 测试训练完成！"
echo "查看结果: ${OUTPUT_DIR}"
echo "============================================================"
