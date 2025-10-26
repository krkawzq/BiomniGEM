#!/usr/bin/env bash
# ============================================================
# BiomniGEM LoRA-SFT 训练脚本（多GPU优化版，适配重构训练器）
# 使用 Accelerate/HF Trainer 进行高效多GPU数据并行训练
# 对应 Python 脚本：train_sft_refactored.py
# ============================================================

set -euo pipefail

# -------------------------------
# GPU 检测和配置
# -------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误: 未找到 nvidia-smi，请在含 GPU 的环境中运行。"
  exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l | awk '{print $1}')
if [[ ${NUM_GPUS} -le 0 ]]; then
  echo "错误: 未检测到GPU。"
  exit 1
fi

echo "检测到 ${NUM_GPUS} 个 GPU"
echo "GPU 信息:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits || true

# -------------------------------
# 基本路径配置（按需修改）
# -------------------------------
MODEL_PATH="/root/autodl-fs/wzq/models/models--SciReason--SciReasoner-8B/snapshots/772c4adaf43c750db5ef04d6f567148ca3daf7b0"
TRAIN_DIR="/root/autodl-fs/wzq/datasets/SynBioCoT/arrow/train"
VAL_DIR="/root/autodl-fs/wzq/datasets/SynBioCoT/arrow/cell_validation"
OUTPUT_DIR="/root/autodl-fs/wzq/BiomniGEM/experiment/sr-sft-$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}"

# 路径存在性检查（早失败更友好）
[[ -d "${MODEL_PATH}" || "${MODEL_PATH}" =~ ^[A-Za-z0-9._/-]+/[A-Za-z0-9._-]+$ ]] || { echo "错误: MODEL_PATH 不存在且不是远程标识：${MODEL_PATH}"; exit 1; }
[[ -d "${TRAIN_DIR}" ]] || { echo "错误: 训练数据目录不存在：${TRAIN_DIR}"; exit 1; }
[[ -d "${VAL_DIR}" ]]   || { echo "错误: 验证数据目录不存在：${VAL_DIR}"; exit 1; }

# -------------------------------
# LoRA 参数
# -------------------------------
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05

# -------------------------------
# 训练超参数（随 GPU 数量自适应）
# -------------------------------
LR=1.5e-4
WEIGHT_DECAY=0.05
EPOCHS=3

if   [[ ${NUM_GPUS} -ge 4 ]]; then
  PER_DEVICE_BATCH=8
  GRAD_ACCUM=2
elif [[ ${NUM_GPUS} -ge 2 ]]; then
  PER_DEVICE_BATCH=12
  GRAD_ACCUM=2
else
  PER_DEVICE_BATCH=16
  GRAD_ACCUM=4
fi

CUTOFF_LEN=4096
WARMUP_RATIO=0.05
SEED=42

# -------------------------------
# 日志与保存
# -------------------------------
SAVE_STEPS=100
LOGGING_STEPS=10
SAVE_TOTAL_LIMIT=5

# -------------------------------
# 评估配置
# -------------------------------
INITIAL_EVAL=false
MAX_TRAIN_SAMPLES=0   # 0 表示全部
MAX_EVAL_SAMPLES=96
EVAL_BATCH_SIZE=48
NUM_PROC=16
FILTER_QUALITY="gold"

# -------------------------------
# 高级配置
# -------------------------------
BF16=true
DEEPSPEED=""          # 设置为 ds_config.json 路径可启用 DeepSpeed
PACKING=false         # 保留开关（当前实现不启用 packing）
ENABLE_THINKING=true
ENABLE_COMPILE=false
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false
RESUME_FROM_CKPT=""
# 多个停止符可空格分隔：例如 "<|im_end|>" "</answer>"
EXTRA_STOP_TOKENS="<|im_end|>"

# -------------------------------
# 分布式端口（可用 MAIN_PORT 覆盖）
# -------------------------------
MAIN_PORT_DEFAULT=29501
MAIN_PORT="${MAIN_PORT:-${MAIN_PORT_DEFAULT}}"

# 常见 NCCL 稳健设置（按集群网络情况调整/注释）
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0

# ============================================================
# 构建命令
# ============================================================
# 计算脚本目录并锁定入口 Python 绝对路径（关键修复）
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/train_sft.py"

# 基础加速参数：按 GPU 数量决定是否加 --multi_gpu
ACCELERATE_ARGS=(launch "--num_processes=${NUM_GPUS}" "--main_process_port=${MAIN_PORT}")
if [[ ${NUM_GPUS} -ge 2 ]]; then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

CMD=(accelerate "${ACCELERATE_ARGS[@]}" "${PY_SCRIPT}"
  --model_path "${MODEL_PATH}"
  --train_dir "${TRAIN_DIR}"
  --val_dir "${VAL_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --epochs "${EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH}"
  --grad_accum "${GRAD_ACCUM}"
  --cutoff_len "${CUTOFF_LEN}"
  --warmup_ratio "${WARMUP_RATIO}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --num_proc "${NUM_PROC}"
  --max_eval_samples_arg "${MAX_EVAL_SAMPLES}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --filter_quality "${FILTER_QUALITY}"
  --seed "${SEED}"
)

# 可选项拼接（只在需要时追加）
[[ "${BF16}" == true ]] && CMD+=(--bf16)
[[ "${PACKING}" == true ]] && CMD+=(--packing)
[[ "${ENABLE_THINKING}" == true ]] && CMD+=(--enable_thinking)
[[ "${ENABLE_COMPILE}" == true ]] && CMD+=(--enable_compile)
[[ "${LOAD_IN_4BIT}" == true ]] && CMD+=(--load_in_4bit)
[[ "${LOAD_IN_8BIT}" == true ]] && CMD+=(--load_in_8bit)
[[ -n "${DEEPSPEED}" ]] && CMD+=(--deepspeed "${DEEPSPEED}")
[[ -n "${RESUME_FROM_CKPT}" ]] && CMD+=(--resume_from_checkpoint "${RESUME_FROM_CKPT}")

# MAX_TRAIN_SAMPLES=0 视为“全部”→ 不追加该参数；>0 才追加
if [[ "${MAX_TRAIN_SAMPLES}" -gt 0 ]]; then
  CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

# 停止符：允许多个，以空格切分追加多次
if [[ -n "${EXTRA_STOP_TOKENS}" ]]; then
  # shellcheck disable=SC2086
  for tok in ${EXTRA_STOP_TOKENS}; do
    CMD+=(--extra_stop_tokens "${tok}")
  done
fi

# ============================================================
# 启动信息
# ============================================================
EFFECTIVE_BATCH=$(( PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS ))

echo "============================================================"
echo "🚀 BiomniGEM SFT 并行训练（快速验证/可用于正式）"
echo "Model:   ${MODEL_PATH}"
echo "Train:   ${TRAIN_DIR}"
echo "Val:     ${VAL_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "============================================================"
echo "配置概要："
echo "  - GPUs: ${NUM_GPUS} (port=${MAIN_PORT})"
echo "  - 训练 Batch: ${PER_DEVICE_BATCH} × ${NUM_GPUS} × grad_accum(${GRAD_ACCUM}) = ${EFFECTIVE_BATCH}"
echo "  - 评估 Batch: ${EVAL_BATCH_SIZE}"
echo "  - Epochs:     ${EPOCHS}"
echo "  - LR:         ${LR}"
echo "  - LoRA:       r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "  - 序列长度:   ${CUTOFF_LEN}"
echo "  - 保存/日志:  save_steps=${SAVE_STEPS}, logging_steps=${LOGGING_STEPS}, save_total_limit=${SAVE_TOTAL_LIMIT}"
echo "  - 评估样本:   ${MAX_EVAL_SAMPLES}"
echo "  - Thinking:   ${ENABLE_THINKING}"
echo "  - 4bit/8bit:  ${LOAD_IN_4BIT}/${LOAD_IN_8BIT}"
echo "  - DeepSpeed:  ${DEEPSPEED:-disabled}"
echo "============================================================"
echo "执行命令："
printf ' %q' "${CMD[@]}"; echo
echo "============================================================"

# ============================================================
# 运行
# ============================================================
"${CMD[@]}"

echo ""
echo "============================================================"
echo "✅ 训练完成！结果目录: ${OUTPUT_DIR}"
echo "============================================================"
