#!/usr/bin/env bash
set -euo pipefail

# Multi-process multi-GPU parallel launcher for AdaGTCN experiments.
# This is script-level parallelism: one single-GPU Python process per GPU.
# It is not DDP and intentionally does not use torchrun.

PROJECT_DIR="/home/yiheng/projects/zuco-benchmark-main"
CONDA_SH="/opt/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-zuco_benchmark_gpu}"

RUN_MODE="${RUN_MODE:-pilot_parallel_smoke}"
DRY_RUN="${DRY_RUN:-1}"
RUN_TAG="${RUN_TAG:-valfix_v1}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7,8}"

SPLIT_JSON="${SPLIT_JSON:-reports/adagtcn_aligned/subject_splits.json}"
PILOT_SEQUENCE_JSONL="${PILOT_SEQUENCE_JSONL:-data/adagtcn_aligned/pilot_y16_60sent_band_vectors_sequences.jsonl}"
FULL_SEQUENCE_JSONL="${FULL_SEQUENCE_JSONL:-data/adagtcn_aligned/paper_y16_full_band_vectors_sequences.jsonl}"

BASE_PROTOCOL="${BASE_PROTOCOL:-Y16_12_2_2_seed0}"
MODEL="${MODEL:-gaze_only_ssm}"
EPOCHS="${EPOCHS:-1}"
PATIENCE="${PATIENCE:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
MAX_LEN="${MAX_LEN:-80}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_DIR"
git -c http.version=HTTP/1.1 pull --ff-only

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda activation script not found: $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

case "$RUN_MODE" in
    pilot_parallel_smoke)
        SEQUENCE_JSONL="$PILOT_SEQUENCE_JSONL"
        ;;
    full_parallel_short)
        SEQUENCE_JSONL="$FULL_SEQUENCE_JSONL"
        MODEL="${FULL_MODEL:-full_cnogsm}"
        EPOCHS="${FULL_EPOCHS:-3}"
        PATIENCE="${FULL_PATIENCE:-2}"
        BATCH_SIZE="${FULL_BATCH_SIZE:-16}"
        HIDDEN_DIM="${FULL_HIDDEN_DIM:-64}"
        MAX_LEN="${FULL_MAX_LEN:-80}"
        ;;
    full_loso)
        SEQUENCE_JSONL="$FULL_SEQUENCE_JSONL"
        MODEL="${FULL_MODEL:-full_cnogsm}"
        EPOCHS="${FULL_EPOCHS:-30}"
        PATIENCE="${FULL_PATIENCE:-8}"
        BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
        HIDDEN_DIM="${FULL_HIDDEN_DIM:-64}"
        MAX_LEN="${FULL_MAX_LEN:-80}"
        if [[ ! -f "$SEQUENCE_JSONL" ]]; then
            echo "Full-mode JSONL not found: $SEQUENCE_JSONL" >&2
            echo "Set FULL_SEQUENCE_JSONL to the confirmed full training JSONL before running a full mode." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported RUN_MODE: $RUN_MODE" >&2
        echo "Supported values: pilot_parallel_smoke, full_parallel_short, full_loso" >&2
        exit 1
        ;;
esac

if [[ ! -f "$SEQUENCE_JSONL" ]]; then
    echo "Sequence JSONL not found: $SEQUENCE_JSONL" >&2
    exit 1
fi

if [[ ! -f "$SPLIT_JSON" ]]; then
    echo "Split JSON not found: $SPLIT_JSON" >&2
    exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -gt 9 ]]; then
    echo "This launcher caps concurrent tasks at 9 GPUs. Got: ${#GPUS[@]}" >&2
    exit 1
fi

mapfile -t LOSO_PROTOCOLS < <(python - "$SPLIT_JSON" <<'PY'
import json
import sys

split_json = sys.argv[1]
obj = json.load(open(split_json, "r", encoding="utf-8"))
protocols = []
for item in obj.get("splits", []):
    protocol = item.get("protocol")
    if isinstance(protocol, str) and protocol.startswith("Y16_LOSO_"):
        protocols.append(protocol)
for protocol in protocols:
    print(protocol)
PY
)

PROTOCOLS=()
SEEDS=()
if [[ "${#LOSO_PROTOCOLS[@]}" -gt 0 ]]; then
    protocol_limit="${#LOSO_PROTOCOLS[@]}"
    if [[ "$RUN_MODE" != "full_loso" ]] && [[ "$protocol_limit" -gt "${#GPUS[@]}" ]]; then
        protocol_limit="${#GPUS[@]}"
    fi

    for ((i = 0; i < protocol_limit; i++)); do
        PROTOCOLS+=("${LOSO_PROTOCOLS[$i]}")
        SEEDS+=("0")
    done
else
    for ((i = 0; i < ${#GPUS[@]}; i++)); do
        PROTOCOLS+=("$BASE_PROTOCOL")
        SEEDS+=("$i")
    done
fi

LOG_DIR="logs/adagtcn_parallel/$RUN_MODE/$RUN_TAG"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "RUN_MODE=$RUN_MODE"
echo "DRY_RUN=$DRY_RUN"
echo "RUN_TAG=$RUN_TAG"
echo "SEQUENCE_JSONL=$SEQUENCE_JSONL"
echo "SPLIT_JSON=$SPLIT_JSON"
echo "MODEL=$MODEL"
echo "GPUS=${GPUS[*]}"
echo "TASKS=${#PROTOCOLS[@]}"

print_command() {
    local gpu="$1"
    local protocol="$2"
    local seed="$3"
    local output_dir="$4"
    local log_file="$5"
    shift 5
    local cmd=("$@")

    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '> %q 2>&1\n' "$log_file"
    printf '# output_dir=%s protocol=%s seed=%s\n' "$output_dir" "$protocol" "$seed"
}

FAILED=0
PIDS=()
PID_LABELS=()
for ((i = 0; i < ${#PROTOCOLS[@]}; i++)); do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    protocol="${PROTOCOLS[$i]}"
    seed="${SEEDS[$i]}"
    output_dir="outputs/adagtcn_parallel/$RUN_MODE/$RUN_TAG/$protocol/${MODEL}_seed${seed}"
    log_file="$LOG_DIR/${RUN_MODE}_${protocol}_gpu${gpu}_${TIMESTAMP}.log"

    cmd=(
        python -m src.adagtcn_aligned.train_cnogsm
        --sequence-jsonl "$SEQUENCE_JSONL"
        --split-json "$SPLIT_JSON"
        --protocol "$protocol"
        --model "$MODEL"
        --epochs "$EPOCHS"
        --patience "$PATIENCE"
        --batch-size "$BATCH_SIZE"
        --hidden-dim "$HIDDEN_DIM"
        --max-len "$MAX_LEN"
        --device "$DEVICE"
        --output-dir "$output_dir"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        print_command "$gpu" "$protocol" "$seed" "$output_dir" "$log_file" "${cmd[@]}"
        continue
    fi

    if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "Skip existing non-empty output dir: $output_dir" >&2
        continue
    fi

    mkdir -p "$output_dir" "$LOG_DIR"
    echo "Starting gpu=$gpu protocol=$protocol seed=$seed log=$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1 &
    PIDS+=("$!")
    PID_LABELS+=("gpu=$gpu protocol=$protocol seed=$seed")

    if [[ "${#PIDS[@]}" -ge "${#GPUS[@]}" ]] || [[ "$i" -eq "$((${#PROTOCOLS[@]} - 1))" ]]; then
        NEXT_PIDS=()
        NEXT_LABELS=()
        for ((j = 0; j < ${#PIDS[@]}; j++)); do
            if ! wait "${PIDS[$j]}"; then
                echo "Task failed: ${PID_LABELS[$j]}" >&2
                FAILED=1
            fi
        done
        PIDS=("${NEXT_PIDS[@]}")
        PID_LABELS=("${NEXT_LABELS[@]}")
    fi
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run only. Set DRY_RUN=0 to launch these single-GPU processes."
    exit 0
fi

if [[ "$FAILED" -ne 0 ]]; then
    echo "One or more parallel tasks failed. Check logs in $LOG_DIR." >&2
    exit 1
fi

echo "All parallel tasks completed. Logs: $LOG_DIR"
