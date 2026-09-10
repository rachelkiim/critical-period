#!/bin/bash
# Parameter sweep: depth × num_image with multiple independent networks
#
# Usage:
#   ./cli/run_sweep.sh
#   ./cli/run_sweep.sh --num_net 1              # Quick test with 1 net

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# ── Parse arguments ──
NUM_NET=10
SEED=42
DRY_RUN=false
EXTRA_ARGS=()

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
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Sweep parameters ──
DEPTHS=(2 3 4 5 6)
NUM_IMAGES=(500 1000 2000 4000 8000 16000 32000)

# Fixed parameters
BATCH_SIZE=128
WIDTH=256
LR_NOISE=1e-4
WEIGHT_DECAY_NOISE=0
NUM_NOISE=32000
EPOCHS_NOISE=15
LR=2e-5
WEIGHT_DECAY=1e-3
EPOCHS=30

# ── Setup ──
TOTAL=$(( ${#DEPTHS[@]} * ${#NUM_IMAGES[@]} ))
EXP_ID="$(date +%y%m%d_%H%M%S)_sweep"
EXP_DIR="./exp/$EXP_ID"

echo "==================================="
echo " Parameter Sweep"
echo "==================================="
echo " Depths:     ${DEPTHS[*]}"
echo " Num images: ${NUM_IMAGES[*]}"
echo " Num nets:   $NUM_NET"
echo " Total runs: $TOTAL"
echo " Exp ID:     $EXP_ID"
echo "==================================="
echo ""

if [ "$DRY_RUN" = false ]; then
    mkdir -p "$EXP_DIR"
fi

# ── Sweep loop ──
COUNT=0

for depth in "${DEPTHS[@]}"; do
for num_image in "${NUM_IMAGES[@]}"; do
    COUNT=$((COUNT + 1))
    COND="d${depth}_nd${num_image}"
    COND_DIR="$EXP_DIR/${COND}"

    echo "[$COUNT/$TOTAL] ${COND} (num_nets=$NUM_NET, seed=$SEED)"

    TRAIN_CMD="python -u scripts/experiment_training.py \
        --output_dir $COND_DIR \
        --batch_size $BATCH_SIZE \
        --depth $depth \
        --width $WIDTH \
        --lr_noise $LR_NOISE \
        --weight_decay_noise $WEIGHT_DECAY_NOISE \
        --num_noise $NUM_NOISE \
        --epochs_noise $EPOCHS_NOISE \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --num_image $num_image \
        --epochs $EPOCHS \
        --num_nets $NUM_NET \
        --seed $SEED \
        --data_path ./data \
        --no_wandb \
        ${EXTRA_ARGS[*]}"
    CAL_CMD="python -u scripts/experiment_calibration.py --exp_path $COND_DIR"

    if [ "$DRY_RUN" = true ]; then
        echo "  $TRAIN_CMD"
        echo "  $CAL_CMD"
        echo ""
    else
        eval $TRAIN_CMD
        eval $CAL_CMD
        echo ""
    fi

done
done

if [ "$DRY_RUN" = true ]; then
    echo "==================================="
    echo "Dry run complete: $COUNT commands"
    echo "==================================="
    exit 0
fi

# ── Generate summary.json ──
python -u -c "
import json, os, numpy as np

exp_dir = '$EXP_DIR'
depths = [2, 3, 4, 5, 6]
num_images = [500, 1000, 2000, 4000, 8000, 16000, 32000]

summary = {}

for depth in depths:
    for num_image in num_images:
        cond = f'd{depth}_nd{num_image}'
        cal_path = os.path.join(exp_dir, cond, 'calibration.json')
        if not os.path.exists(cal_path):
            continue
        with open(cal_path) as f:
            cal = json.load(f)

        ece_wo_list = cal['without_warmup'].get('ece_list', [cal['without_warmup']['ece_final']])
        ece_w_list = cal['with_warmup'].get('ece_list', [cal['with_warmup']['ece_final']])

        summary[cond] = {
            'depth': depth,
            'num_image': num_image,
            'ece_wo': ece_wo_list,
            'ece_w': ece_w_list,
            'ece_wo_mean': float(np.mean(ece_wo_list)),
            'ece_wo_std':  float(np.std(ece_wo_list)),
            'ece_w_mean':  float(np.mean(ece_w_list)),
            'ece_w_std':   float(np.std(ece_w_list)),
        }

with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('=== ECE Summary ===')
print(f'{\"Condition\":>15s}  {\"ECE (w/o)\":>20s}  {\"ECE (w/)\":>20s}')
for cond, v in summary.items():
    wo = f\"{v['ece_wo_mean']:.4f} +/- {v['ece_wo_std']:.4f}\"
    w  = f\"{v['ece_w_mean']:.4f} +/- {v['ece_w_std']:.4f}\"
    print(f'{cond:>15s}  {wo:>20s}  {w:>20s}')
"

echo ""
echo "==================================="
echo " Sweep Complete: $COUNT runs"
echo "==================================="
echo "Experiment ID: $EXP_ID"
echo "Results: $EXP_DIR/"
echo "Summary: $EXP_DIR/summary.json"
