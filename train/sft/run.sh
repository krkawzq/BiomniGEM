#!/usr/bin/env bash
# 用途：加速器(Accelerate) + FSDP + LoRA 的 SFT 训练一键启动脚本
# 环境依赖：accelerate、transformers、datasets、peft、torch 均需预先安装
# 配置文件建议参考 accelerate config，选择 FSDP / bf16 / 进程数=GPU 数量

set -euo pipefail

############################
# 可根据实际情况修改的默认推荐配置
############################

# ========== 必填 ==========
MODEL_PATH="/root/autodl-fs/wzq/models/models--SciReason--SciReasoner-8B/snapshots/772c4adaf43c750db5ef04d6f567148ca3daf7b0"         # 预训练模型目录
TRAIN_JSONL="/root/autodl-fs/wzq/BiomniGEM/train/llama-factory/data/train_sft.jsonl"                                              # 训练集
VAL_JSONL="/root/autodl-fs/wzq/datasets/SynBioCoT/jsonl/cell_validation.jsonl"                                                    # 验证集，如无可设为空字符串
OUT_ROOT="/root/autodl-fs/wzq/BiomniGEM/experiment"                                                                              # 输出实验根目录

# ========== 可选 ==========
# 指定GPU列表（如：CUDA_VISIBLE_DEVICES="0,1,2,3"；留空使用全部可用）
CUDA_VISIBLE_DEVICES="0,1,2,3"

# accelerate 配置文件路径（留空则用默认）
ACCEL_CFG="/root/autodl-tmp/huggingface/accelerate/default_config.yaml"

# 训练推荐参数（如需个性化修改，可提前 export 环境变量覆盖默认值）
SEED=42
PER_DEVICE_TRAIN_BATCH_SIZE=8        # 建议: 8-16，根据GPU显存适配
PER_DEVICE_EVAL_BATCH_SIZE=32
GRAD_ACCUM="${GRAD_ACCUM:-8}"        # 梯度累积，显存紧张时可适当加大
LR=1e-4
WEIGHT_DECAY=0.05
EPOCHS=3
SCHED=cosine
WARMUP_RATIO=0.03                    # 稍微减少，提升收敛速度
CUTOFF_LEN=4096                      # 最大 token 长度
MAX_NEW_TOKENS_EVAL=1024

# 生成相关参数
TEMP=0.7
TOP_P=0.9
REP_PEN=1.1                          # 防止重复度更高一点

# LoRA 相关参数
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGET="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# 日志/模型保存/评估设置
LOGGING_STEPS=10
SAVE_STEPS=100
EVAL_STEPS=100
KEEP_BEST_K=2
KEEP_RECENT_N=5

############################
# 路径与时间戳处理
############################
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_ROOT}/exp-${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

############################
# 基本检查
############################
if [[ ! -d "${MODEL_PATH}" && ! "${MODEL_PATH}" =~ "/" ]]; then
  echo "[WARN] MODEL_PATH='${MODEL_PATH}' 不是本地目录，后续将尝试从 Huggingface Hub 拉取"
fi

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[ERR ] 训练集不存在: ${TRAIN_JSONL}"
  exit 1
fi

if [[ -z "${VAL_JSONL}" ]]; then
  echo "[INFO] 未指定验证集，将跳过评估"
elif [[ ! -f "${VAL_JSONL}" ]]; then
  echo "[ERR ] 验证集不存在: ${VAL_JSONL}"
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERR ] accelerate 未安装，请先执行 'pip install accelerate'"
  exit 1
fi

############################
# 拼接命令
############################
LAUNCH_CMD=(accelerate launch)
if [[ -n "${ACCEL_CFG}" ]]; then
  LAUNCH_CMD+=(--config_file "${ACCEL_CFG}")
fi

PY_SCRIPT="/root/autodl-fs/wzq/BiomniGEM/train/sft/train.py"

RUN_ARGS=(
  --model_path "${MODEL_PATH}"
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUT_DIR}"
  --seed "${SEED}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${EPOCHS}"
  --lr_scheduler_type "${SCHED}"
  --warmup_ratio "${WARMUP_RATIO}"
  --cutoff_len "${CUTOFF_LEN}"
  --max_new_tokens_eval "${MAX_NEW_TOKENS_EVAL}"
  --temperature "${TEMP}"
  --top_p "${TOP_P}"
  --repetition_penalty "${REP_PEN}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target "${LORA_TARGET}"
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --keep_best_k "${KEEP_BEST_K}"
  --keep_recent_n "${KEEP_RECENT_N}"
)

if [[ -n "${VAL_JSONL}" ]]; then
  RUN_ARGS+=(--val_jsonl "${VAL_JSONL}")
fi

############################
# 打印任务信息并执行
############################
echo "================= LLM SFT 训练任务启动 ================="
echo "MODEL_PATH          : ${MODEL_PATH}"
echo "TRAIN_JSONL         : ${TRAIN_JSONL}"
echo "VAL_JSONL           : ${VAL_JSONL:-<none>}"
echo "OUT_DIR             : ${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<inherit>}"
echo "--------------------------------------------------------"
echo "${LAUNCH_CMD[@]} ${PY_SCRIPT} \\"
for a in "${RUN_ARGS[@]}"; do printf "  %q " "$a"; done; echo
echo "========================================================"

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${LAUNCH_CMD[@]}" "${PY_SCRIPT}" "${RUN_ARGS[@]}"
else
  "${LAUNCH_CMD[@]}" "${PY_SCRIPT}" "${RUN_ARGS[@]}"
fi
