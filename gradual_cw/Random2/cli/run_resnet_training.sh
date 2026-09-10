#!/bin/bash
# Run ResNet18 experiment (pretrained/untrained): training + ECE + AUROC
# Supports multiple independent runs via --num_net
#
# Usage:
#   ./cli/run_resnet_training.sh
#   ./cli/run_resnet_training.sh --num_net 1              # Quick test with 1 net

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# ── Parse --num_net from args (extract before passing rest to Python) ──
NUM_NET=1
SEED=42
INIT_MODE=untrained
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_net)
            NUM_NET="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            PASS_ARGS+=("$1" "$2")
            shift 2
            ;;
        --init_mode)
            INIT_MODE="$2"
            PASS_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Default Python args (can be overridden via PASS_ARGS) ──
DEFAULT_ARGS=(
    --batch_size 256
    --lr 0.01
    --momentum 0.9
    --weight_decay 1e-4
    --epochs 80
    --epochs_noise 5
    --data_path ./data
)

# ── Generate experiment ID and base directory ──
EXP_ID="$(date +%y%m%d_%H%M%S)_resnet_${INIT_MODE}_nn${NUM_NET}"
EXP_DIR="./exp/$EXP_ID"
mkdir -p "$EXP_DIR"

echo "==================================="
echo " ResNet18 Experiment (init_mode: $INIT_MODE)"
echo " num_net: $NUM_NET"
echo " exp_id:  $EXP_ID"
echo "==================================="
echo ""

# ── Loop over networks ──
for (( i=0; i<NUM_NET; i++ )); do
    NET_SEED=$((SEED + i))
    NET_DIR="$EXP_DIR/net_${i}"

    echo "--- Network $((i+1))/$NUM_NET (seed=$NET_SEED) ---"

    python -u scripts/experiment_resnet.py \
        "${DEFAULT_ARGS[@]}" \
        --seed "$NET_SEED" \
        --output_dir "$NET_DIR" \
        "${PASS_ARGS[@]}"

    echo ""
done

# ── Generate summary.json ──
python -u -c "
import json, os, numpy as np

exp_dir = '$EXP_DIR'
num_net = $NUM_NET
networks = []

for i in range(num_net):
    path = os.path.join(exp_dir, f'net_{i}', 'training_results.json')
    with open(path) as f:
        data = json.load(f)
    networks.append({
        'net_id': i,
        'test_acc_wo': data['without_warmup']['test_acc'][-1],
        'test_acc_w':  data['with_warmup']['test_acc'][-1],
        'ece_wo':      data['without_warmup']['ece'],
        'ece_w':       data['with_warmup']['ece'],
        'auroc_wo':    data['without_warmup']['auroc'],
        'auroc_w':     data['with_warmup']['auroc'],
    })

keys = ['test_acc_wo', 'test_acc_w', 'ece_wo', 'ece_w', 'auroc_wo', 'auroc_w']
mean = {k: float(np.mean([n[k] for n in networks])) for k in keys}
std  = {k: float(np.std([n[k] for n in networks]))  for k in keys}

summary = {'num_net': num_net, 'networks': networks, 'mean': mean, 'std': std}

with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('=== Summary ===')
for k in keys:
    print(f'  {k:15s}: {mean[k]:.4f} +/- {std[k]:.4f}')
"

echo ""
echo "==================================="
echo " ResNet18 Experiment Complete"
echo "==================================="
echo "Experiment ID: $EXP_ID"
echo "Results: $EXP_DIR/"
echo "Summary: $EXP_DIR/summary.json"
