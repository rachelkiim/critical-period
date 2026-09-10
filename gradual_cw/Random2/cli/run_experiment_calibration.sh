#!/bin/bash
# Run calibration experiment pipeline (training + calibration)
#
# Usage:
#   ./cli/run_experiment_calibration.sh
#   ./cli/run_experiment_calibration.sh --num_net 1              # Quick test with 1 net
#
# Notes:
#   - Arguments before `--calib-args` are passed to training.
#   - Arguments after `--calib-args` are passed to calibration.

set -euo pipefail

# Change to project root directory
cd "$(dirname "$0")/.."

# Split args into training/calibration groups.
TRAIN_ARGS=()
CAL_ARGS=()
TARGET="train"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --calib-args)
            TARGET="calibration"
            shift
            ;;
        *)
            if [[ "$TARGET" == "train" ]]; then
                TRAIN_ARGS+=("$1")
            else
                CAL_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

echo "==================================="
echo " Experiment Runner"
echo "==================================="
echo ""

# Run training stage first
TRAIN_LOG=$(mktemp)
trap 'rm -f "$TRAIN_LOG"' EXIT
python -u scripts/experiment_training.py \
    --batch_size 128 \
    --depth 6 \
    --width 256 \
    --lr_noise 1e-4 \
    --weight_decay_noise 0 \
    --num_noise 32000 \
    --epochs_noise 15 \
    --lr 2e-5 \
    --weight_decay 1e-3 \
    --num_image 4000 \
    --epochs 30 \
    --num_nets 30 \
    --seed 42 \
    --data_path ./data \
    --no_wandb \
    "${TRAIN_ARGS[@]}" | tee "$TRAIN_LOG"

EXP_ID=$(grep '^EXP_ID=' "$TRAIN_LOG" | tail -1 | cut -d= -f2-)
EXP_PATH=$(grep '^EXP_PATH=' "$TRAIN_LOG" | tail -1 | cut -d= -f2-)

if [ -z "${EXP_PATH:-}" ]; then
    echo "Error: Could not determine EXP_PATH from training output."
    exit 1
fi

python -u scripts/experiment_calibration.py --exp_path "$EXP_PATH" "${CAL_ARGS[@]}"

echo ""
echo "==================================="
echo "Experiment Complete"
echo "==================================="
echo "Experiment ID: ${EXP_ID:-$(basename "$EXP_PATH")}"
echo "Results: $EXP_PATH"
