#!/bin/bash
# Run input space (2D) uncertainty experiment
#
# Usage:
#   ./cli/run_input_space.sh                          # Default parameters
#   ./cli/run_input_space.sh --num_net 1              # Quick test with 1 net

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "==================================="
echo " Input Space Uncertainty Experiment"
echo "==================================="
echo ""

# Run the experiment
python -u scripts/input_space.py \
    --model_structure 2,10,10,2 \
    --num_net 30 \
    --epochs 100 \
    --batch_size 100 \
    --num_noise 1000 \
    --num_test 1000 \
    --lr 1e-3 \
    --input_range 10 \
    --output_path ./exp \
    "$@"

# Get the experiment ID from the last run
EXP_ID=$(ls -t exp/ 2>/dev/null | grep input_space | head -1)

echo ""
echo "==================================="
echo "Experiment Complete"
echo "==================================="
echo "Experiment ID: $EXP_ID"
echo "Results: ./exp/$EXP_ID/"
