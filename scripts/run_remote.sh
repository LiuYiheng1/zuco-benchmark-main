#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/yiheng/projects/zuco-benchmark-main"
CONDA_SH="/opt/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-zuco_benchmark_gpu}"
SKIP_GIT_PULL="${SKIP_GIT_PULL:-0}"

cd "$PROJECT_DIR"
if [[ "$SKIP_GIT_PULL" == "1" ]]; then
    echo "SKIP_GIT_PULL=1, skipping git pull"
else
    git pull
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8

mkdir -p logs checkpoints outputs

TRAIN_CMD=""

# Current project DDP support has not been confirmed.
# Do not use torchrun --nproc_per_node=9 until the training entry,
# DistributedSampler, rank/local_rank handling, rank-0 checkpoint/logging,
# and validation metric aggregation are implemented and verified.
if [ -z "$TRAIN_CMD" ]; then
  echo "请先确认训练入口命令"
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/run_${timestamp}.log"

echo "Running: $TRAIN_CMD"
bash -lc "$TRAIN_CMD" 2>&1 | tee "$log_file"
