#!/bin/bash
# Run benchmark CNN experiment: multi-network training + calibration + OOD detection
#
# Usage:
#   ./cli/run_benchmark_experiment.sh
#   ./cli/run_benchmark_experiment.sh --num_net 1            # Quick test with 1 net
#   ./cli/run_benchmark_experiment.sh --num_net 30 --calib-args --methods baseline --ood-args --methods baseline

set -euo pipefail

# Change to project root directory
cd "$(dirname "$0")/.."

NUM_NET=30
SEED=42
TRAIN_ARGS=()
CALIB_ARGS=()
OOD_ARGS=()
TARGET="train"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_net)
            NUM_NET="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --calib-args)
            TARGET="calibration"
            shift
            ;;
        --ood-args)
            TARGET="ood"
            shift
            ;;
        *)
            case "$TARGET" in
                train)
                    TRAIN_ARGS+=("$1")
                    ;;
                calibration)
                    CALIB_ARGS+=("$1")
                    ;;
                ood)
                    OOD_ARGS+=("$1")
                    ;;
            esac
            shift
            ;;
    esac
done

DEFAULT_ARGS=(
    --batch_size 256
    --lr 0.1
    --momentum 0.9
    --weight_decay 0
    --epochs 50
    --epochs_noise 5
    --data_path ./data
)

EXP_ID="$(date +%y%m%d_%H%M%S)_benchmark_nn${NUM_NET}"
EXP_DIR="./exp/$EXP_ID"
mkdir -p "$EXP_DIR"

echo "==================================="
echo " Benchmark CNN Experiment"
echo " num_net: $NUM_NET"
echo " exp_id:  $EXP_ID"
echo "==================================="
echo ""

for (( i=0; i<NUM_NET; i++ )); do
    NET_SEED=$((SEED + i))
    NET_DIR="$EXP_DIR/net_${i}"

    echo "--- Network $((i+1))/$NUM_NET (seed=$NET_SEED) ---"

    python -u scripts/benchmark_training.py \
        "${DEFAULT_ARGS[@]}" \
        "${TRAIN_ARGS[@]}" \
        --seed "$NET_SEED" \
        --output_dir "$NET_DIR"

    echo ""
done

EVAL_CMD=(./cli/run_benchmark_evaluation.sh "$EXP_ID")
if (( ${#CALIB_ARGS[@]} > 0 )); then
    EVAL_CMD+=(--calib-args "${CALIB_ARGS[@]}")
fi
if (( ${#OOD_ARGS[@]} > 0 )); then
    EVAL_CMD+=(--ood-args "${OOD_ARGS[@]}")
fi

"${EVAL_CMD[@]}"

echo ""
echo "==================================="
echo " Benchmark Experiment Complete"
echo "==================================="
echo "Experiment ID: $EXP_ID"
echo "Results: $EXP_DIR/"
echo "Summary: $EXP_DIR/summary.json"
