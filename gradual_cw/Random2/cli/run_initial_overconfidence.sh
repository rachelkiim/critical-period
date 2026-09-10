#!/bin/bash
# Run initial overconfidence experiment
#
# Usage:
#   ./cli/run_initial_overconfidence.sh                          # Default parameters
#   ./cli/run_initial_overconfidence.sh --experiment 1           # Exp1 only (vary outputs)
#   ./cli/run_initial_overconfidence.sh --experiment 2           # Exp2 only (vary depth)
#   ./cli/run_initial_overconfidence.sh --num_net 1              # Quick test with 1 net

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "==================================="
echo " Initial Overconfidence Experiment"
echo "==================================="
echo ""

# Run the experiment
python -u scripts/initial_overconfidence.py \
    --experiment both \
    --num_net 30 \
    --num_hidden 100 \
    --depth 2 \
    --num_outputs 2 \
    --num_output_list 2,3,4,5,6,7,8,9,10 \
    --depth_list 2,3,4,5,6,7,8,9,10 \
    --epochs 30 \
    --batch_size 128 \
    --num_noise 10000 \
    --lr 0.01 \
    --momentum 0.9 \
    --data_path ./data \
    --output_path ./exp \
    "$@"

# Get the experiment ID from the last run
EXP_ID=$(ls -t exp/ 2>/dev/null | grep overconf | head -1)

echo ""
echo "==================================="
echo "Experiment Complete"
echo "==================================="
echo "Experiment ID: $EXP_ID"
echo "Results: ./exp/$EXP_ID/"
