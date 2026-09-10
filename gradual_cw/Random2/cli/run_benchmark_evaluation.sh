#!/bin/bash
# Run benchmark evaluation (calibration + OOD detection) on multi-network experiment
#
# Usage:
#   ./cli/run_benchmark_evaluation.sh <exp_id>
#   ./cli/run_benchmark_evaluation.sh <exp_id> --calib-args --methods baseline --ood-args --methods baseline

set -euo pipefail

# Change to project root directory
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <exp_id> [--calib-args ...] [--ood-args ...]"
    exit 1
fi

EXP_ID="$1"
shift

CALIB_ARGS=()
OOD_ARGS=()
TARGET="shared"

while [[ $# -gt 0 ]]; do
    case "$1" in
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
                shared)
                    CALIB_ARGS+=("$1")
                    OOD_ARGS+=("$1")
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

EXP_DIR="./exp/$EXP_ID"
if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: Experiment directory not found: $EXP_DIR"
    exit 1
fi

mapfile -t NET_DIRS < <(find "$EXP_DIR" -maxdepth 1 -mindepth 1 -type d -name 'net_*' | sort -V)
if (( ${#NET_DIRS[@]} == 0 )); then
    echo "Error: No net_* directories found under $EXP_DIR"
    echo "This evaluator expects multi-network benchmark layout (exp/<exp_id>/net_i)."
    exit 1
fi

echo "==================================="
echo " Benchmark Evaluation"
echo "==================================="
echo "Experiment ID: $EXP_ID"
echo "Networks: ${#NET_DIRS[@]}"
echo ""

for NET_DIR in "${NET_DIRS[@]}"; do
    NET_NAME="$(basename "$NET_DIR")"
    echo "--- Evaluating $NET_NAME ---"

    python -u scripts/benchmark_calibration.py --exp_path "$NET_DIR" "${CALIB_ARGS[@]}"
    python -u scripts/benchmark_ood_detection.py --exp_path "$NET_DIR" "${OOD_ARGS[@]}"

    echo ""
done

python -u - <<'PY' "$EXP_DIR"
import json
import os
import sys
import numpy as np

exp_dir = sys.argv[1]
net_dirs = sorted(
    [d for d in os.listdir(exp_dir) if d.startswith("net_") and os.path.isdir(os.path.join(exp_dir, d))],
    key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
)


def _net_id_from_name(name: str, fallback: int) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return fallback


def _extract_scalar(method_dict: dict, key: str):
    if key in method_dict and method_dict[key] is not None:
        return float(method_dict[key])

    list_key = f"{key}_list"
    values = method_dict.get(list_key)
    if isinstance(values, list) and values:
        return float(np.mean(values))

    return None


# summary.json from training_results.json
summary_networks = []
for idx, net_name in enumerate(net_dirs):
    net_id = _net_id_from_name(net_name, idx)
    train_path = os.path.join(exp_dir, net_name, "training_results.json")
    if not os.path.exists(train_path):
        continue

    with open(train_path, "r") as f:
        train_data = json.load(f)

    test_acc_wo = train_data["without_warmup"]["test_acc"][-1]
    test_acc_w = train_data["with_warmup"]["test_acc"][-1]

    summary_networks.append(
        {
            "net_id": net_id,
            "test_acc_wo": float(test_acc_wo),
            "test_acc_w": float(test_acc_w),
        }
    )

summary = {
    "num_net": len(summary_networks),
    "networks": summary_networks,
    "mean": {},
    "std": {},
}
if summary_networks:
    keys = ["test_acc_wo", "test_acc_w"]
    summary["mean"] = {k: float(np.mean([n[k] for n in summary_networks])) for k in keys}
    summary["std"] = {k: float(np.std([n[k] for n in summary_networks])) for k in keys}

with open(os.path.join(exp_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)


# calibration_comparison.json (root aggregate, scalar ECE per net)
calib_networks = []
calib_aggregate_buf = {}
for idx, net_name in enumerate(net_dirs):
    net_id = _net_id_from_name(net_name, idx)
    calib_path = os.path.join(exp_dir, net_name, "calibration_comparison.json")
    if not os.path.exists(calib_path):
        continue

    with open(calib_path, "r") as f:
        calib_data = json.load(f)

    methods_out = {}
    for method, method_dict in calib_data.items():
        if method in {"n_repeat", "seed"} or not isinstance(method_dict, dict):
            continue

        ece_wo = _extract_scalar(method_dict, "ece_wo")
        ece_w = _extract_scalar(method_dict, "ece_w")
        if ece_wo is None or ece_w is None:
            continue

        methods_out[method] = {
            "ece_wo": ece_wo,
            "ece_w": ece_w,
        }

        if method not in calib_aggregate_buf:
            calib_aggregate_buf[method] = {"ece_wo_list": [], "ece_w_list": []}
        calib_aggregate_buf[method]["ece_wo_list"].append(ece_wo)
        calib_aggregate_buf[method]["ece_w_list"].append(ece_w)

    calib_networks.append({"net_id": net_id, "methods": methods_out})

calib_aggregate = {}
for method, vals in calib_aggregate_buf.items():
    ece_wo_list = vals["ece_wo_list"]
    ece_w_list = vals["ece_w_list"]
    calib_aggregate[method] = {
        "ece_wo_list": ece_wo_list,
        "ece_w_list": ece_w_list,
        "ece_wo_mean": float(np.mean(ece_wo_list)) if ece_wo_list else None,
        "ece_wo_std": float(np.std(ece_wo_list)) if ece_wo_list else None,
        "ece_w_mean": float(np.mean(ece_w_list)) if ece_w_list else None,
        "ece_w_std": float(np.std(ece_w_list)) if ece_w_list else None,
    }

calib_root = {
    "num_net": len(calib_networks),
    "networks": calib_networks,
    "aggregate": calib_aggregate,
}
with open(os.path.join(exp_dir, "calibration_comparison.json"), "w") as f:
    json.dump(calib_root, f, indent=2)


# ood_comparison.json (keep per-net raw method payload)
ood_networks = []
ood_aggregate_buf = {}
for idx, net_name in enumerate(net_dirs):
    net_id = _net_id_from_name(net_name, idx)
    ood_path = os.path.join(exp_dir, net_name, "ood_comparison.json")
    if not os.path.exists(ood_path):
        continue

    with open(ood_path, "r") as f:
        ood_data = json.load(f)

    methods_out = {}
    for method, method_dict in ood_data.items():
        if not isinstance(method_dict, dict):
            continue

        methods_out[method] = method_dict

        auroc_wo = _extract_scalar(method_dict, "auroc_wo")
        auroc_w = _extract_scalar(method_dict, "auroc_w")
        if auroc_wo is None or auroc_w is None:
            continue

        if method not in ood_aggregate_buf:
            ood_aggregate_buf[method] = {"auroc_wo_list": [], "auroc_w_list": []}
        ood_aggregate_buf[method]["auroc_wo_list"].append(auroc_wo)
        ood_aggregate_buf[method]["auroc_w_list"].append(auroc_w)

    ood_networks.append({"net_id": net_id, "methods": methods_out})

ood_aggregate = {}
for method, vals in ood_aggregate_buf.items():
    auroc_wo_list = vals["auroc_wo_list"]
    auroc_w_list = vals["auroc_w_list"]
    ood_aggregate[method] = {
        "auroc_wo_list": auroc_wo_list,
        "auroc_w_list": auroc_w_list,
        "auroc_wo_mean": float(np.mean(auroc_wo_list)) if auroc_wo_list else None,
        "auroc_wo_std": float(np.std(auroc_wo_list)) if auroc_wo_list else None,
        "auroc_w_mean": float(np.mean(auroc_w_list)) if auroc_w_list else None,
        "auroc_w_std": float(np.std(auroc_w_list)) if auroc_w_list else None,
    }

ood_root = {
    "num_net": len(ood_networks),
    "networks": ood_networks,
    "aggregate": ood_aggregate,
}
with open(os.path.join(exp_dir, "ood_comparison.json"), "w") as f:
    json.dump(ood_root, f, indent=2)

print("=== Root Aggregation Complete ===")
print(f"summary.json: {os.path.join(exp_dir, 'summary.json')}")
print(f"calibration_comparison.json: {os.path.join(exp_dir, 'calibration_comparison.json')}")
print(f"ood_comparison.json: {os.path.join(exp_dir, 'ood_comparison.json')}")
PY

echo ""
echo "==================================="
echo " Evaluation Complete"
echo "==================================="
echo "Results: $EXP_DIR/"
