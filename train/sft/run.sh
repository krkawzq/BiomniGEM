#!/usr/bin/env bash
# 用途：Accelerate + (DDP/FSDP) + LoRA 的 SFT 训练一键启动脚本（稳健版）
# 依赖：accelerate、transformers、datasets、peft、torch、jinja2
# 建议先 `accelerate config` 选择 FSDP/bf16；多卡务必用 accelerate/torchrun 启动

set -euo pipefail

########################################
# 必填/常改配置
########################################
MODEL_PATH="/root/autodl-fs/wzq/models/models--SciReason--SciReasoner-8B/snapshots/772c4adaf43c750db5ef04d6f567148ca3daf7b0"
TRAIN_JSONL="/root/autodl-fs/wzq/BiomniGEM/train/llama-factory/data/train_sft.jsonl"
VAL_JSONL="/root/autodl-fs/wzq/datasets/SynBioCoT/jsonl/cell_validation.jsonl"   # 留空则跳过评估
OUT_ROOT="/root/autodl-fs/wzq/BiomniGEM/experiment"
PY_SCRIPT="/root/autodl-fs/wzq/BiomniGEM/train/sft/train.py"

# 指定 GPU（逗号分隔），留空=使用全部
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# accelerate 配置文件（可留空）
ACCEL_CFG="${ACCEL_CFG:-/root/autodl-tmp/huggingface/accelerate/default_config.yaml}"

########################################
# 训练超参（可通过外部 export 覆盖）
########################################
SEED="${SEED:-42}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
EPOCHS="${EPOCHS:-3}"
SCHED="${SCHED:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
CUTOFF_LEN="${CUTOFF_LEN:-4096}"
MAX_NEW_TOKENS_EVAL="${MAX_NEW_TOKENS_EVAL:-1024}"

# 生成相关
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
REP_PEN="${REP_PEN:-1.1}"

# LoRA
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET="${LORA_TARGET:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# 日志/评估/保存
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
KEEP_BEST_K="${KEEP_BEST_K:-2}"
KEEP_RECENT_N="${KEEP_RECENT_N:-5}"

# —— 分布式/数据管线稳健参数（与训练脚本保持一致）——
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-10}"
MAP_NUM_PROC="${MAP_NUM_PROC:-32}"       # datasets.map 多进程
DL_WORKERS="${DL_WORKERS:-32}"           # DataLoader num_workers
PIN_MEMORY="${PIN_MEMORY:-false}"       # true/false
DISABLE_TOKENIZERS_PARALLEL="${DISABLE_TOKENIZERS_PARALLEL:-true}"
DEBUG_MARKS="${DEBUG_MARKS:-true}"

# —— 新增：评估控制参数 —— 
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-96}"        # 0=用全量验证集
EVAL_SHUFFLE="${EVAL_SHUFFLE:-true}"            # true/false：每次评估前是否打乱
EVAL_SHUFFLE_SEED="${EVAL_SHUFFLE_SEED:-42}"  # 打乱基准种子（脚本内会用 seed+global_step）

########################################
# NCCL/系统环境（可按需覆盖）
########################################
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=$([[ "${DISABLE_TOKENIZERS_PARALLEL}" == "true" ]] && echo "false" || echo "true")

# 在多数 PCIe/容器环境下，以下设置更稳（可按需注释掉逐步开放）
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

########################################
# 输出与 GPU 数计算
########################################
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_ROOT}/exp-${TIMESTAMP}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run.log"

# 计算进程数（与 GPU 数一致）
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_gpus[@]}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC=1
  fi
fi

########################################
# 基本检查
########################################
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[ERR ] 训练集不存在: ${TRAIN_JSONL}" | tee -a "${LOG_FILE}"
  exit 1
fi
if [[ -n "${VAL_JSONL}" && ! -f "${VAL_JSONL}" ]]; then
  echo "[ERR ] 验证集不存在: ${VAL_JSONL}" | tee -a "${LOG_FILE}"
  exit 1
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERR ] accelerate 未安装，请先 'pip install -U accelerate'" | tee -a "${LOG_FILE}"
  exit 1
fi

########################################
# 组装启动命令
########################################
LAUNCH_CMD=(accelerate launch --num_processes "${NPROC}" --num_machines 1)
[[ -n "${ACCEL_CFG}" && -f "${ACCEL_CFG}" ]] && LAUNCH_CMD+=(--config_file "${ACCEL_CFG}")

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
  # 与训练脚本对齐的稳健参数
  --ddp_timeout_minutes "${DDP_TIMEOUT_MIN}"
  --map_num_proc "${MAP_NUM_PROC}"
  --dl_workers "${DL_WORKERS}"
  --pin_memory "${PIN_MEMORY}"
  --disable_tokenizers_parallel "${DISABLE_TOKENIZERS_PARALLEL}"
  # —— 新增：评估控制参数 —— 
  --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --eval_shuffle "${EVAL_SHUFFLE}"
  --eval_shuffle_seed "${EVAL_SHUFFLE_SEED}"
)
[[ -n "${VAL_JSONL}" ]] && RUN_ARGS+=(--val_jsonl "${VAL_JSONL}")
[[ "${DEBUG_MARKS}" == "true" ]] && RUN_ARGS+=(--debug_marks)

########################################
# 打印信息并执行
########################################
echo "================= LLM SFT 训练任务启动 =================" | tee -a "${LOG_FILE}"
echo "MODEL_PATH          : ${MODEL_PATH}"                      | tee -a "${LOG_FILE}"
echo "TRAIN_JSONL         : ${TRAIN_JSONL}"                     | tee -a "${LOG_FILE}"
echo "VAL_JSONL           : ${VAL_JSONL:-<none>}"               | tee -a "${LOG_FILE}"
echo "OUT_DIR             : ${OUT_DIR}"                         | tee -a "${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<inherit>}" | tee -a "${LOG_FILE}"
echo "NPROC               : ${NPROC}"                           | tee -a "${LOG_FILE}"
echo "ACCEL_CFG           : ${ACCEL_CFG:-<default>}"            | tee -a "${LOG_FILE}"
echo "EVAL_MAX_SAMPLES    : ${EVAL_MAX_SAMPLES}"                | tee -a "${LOG_FILE}"
echo "EVAL_SHUFFLE        : ${EVAL_SHUFFLE}"                    | tee -a "${LOG_FILE}"
echo "EVAL_SHUFFLE_SEED   : ${EVAL_SHUFFLE_SEED}"               | tee -a "${LOG_FILE}"
echo "--------------------------------------------------------" | tee -a "${LOG_FILE}"
echo "${LAUNCH_CMD[@]} ${PY_SCRIPT} \\"                         | tee -a "${LOG_FILE}"
for a in "${RUN_ARGS[@]}"; do printf "  %q " "$a"; done | tee -a "${LOG_FILE}"; echo | tee -a "${LOG_FILE}"
echo "========================================================" | tee -a "${LOG_FILE}"

# 将本次会话stdout/err同时写入日志
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${LAUNCH_CMD[@]}" "${PY_SCRIPT}" "${RUN_ARGS[@]}"
else
  "${LAUNCH_CMD[@]}" "${PY_SCRIPT}" "${RUN_ARGS[@]}"
fi
