#!/usr/bin/env python3
"""Benchmark calibration comparison: evaluate post-hoc calibration methods."""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torchvision

from src.data.dataloader import get_cifar10_transform, split_dataset
from src.evaluation.inference import inference
from src.evaluation.calibration import reliability_diagram, ece
from src.models import get_model
from src.baseline.calibration_methods import (
    find_best_temperature,
    temperature_scaling_ece,
    train_vector_scaling,
    vector_scaling_ece,
    train_isotonic_regression,
    isotonic_regression_ece,
)
from src.logger import NumpyEncoder


ALL_METHODS = ["baseline", "temp_scaling", "vec_scaling", "isotonic"]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark calibration comparison")
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument("--exp_id", type=str,
                           help="Experiment ID (directory under exp/)")
    exp_group.add_argument("--exp_path", type=str,
                           help="Absolute or relative path to experiment directory")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--methods", type=str, default="all",
                        help=f"Comma-separated methods: {','.join(ALL_METHODS)} (default: all)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--n_repeat", type=int, default=10,
                        help="Number of repeated evaluations with different splits")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_exp_path(args):
    if args.exp_path:
        return args.exp_path
    return os.path.join("exp", args.exp_id)


def _evaluate_single_split(model_wo, model_w, val_loader, test_loader,
                           methods, num_classes, device):
    """Run all selected calibration methods on a single val/test split.

    Returns dict mapping method name -> per-run result dict.
    """
    run = {}

    # Baseline
    if "baseline" in methods:
        pred_wo, conf_wo = inference(model_wo, test_loader, device=device)
        pred_w, conf_w = inference(model_w, test_loader, device=device)

        acc_wo, conf_bin_wo, ns_wo = reliability_diagram(pred_wo, conf_wo, num_bin=10)
        acc_w, conf_bin_w, ns_w = reliability_diagram(pred_w, conf_w, num_bin=10)

        run["baseline"] = {
            "ece_wo": ece(acc_wo, conf_bin_wo, ns_wo),
            "ece_w": ece(acc_w, conf_bin_w, ns_w),
        }

    # Temperature scaling
    if "temp_scaling" in methods:
        t_candidates = [1, 1.05, 1.1, 1.2, 1.5, 2, 3, 5]
        t_wo = find_best_temperature(model_wo, val_loader, t_candidates, device)
        t_w = find_best_temperature(model_w, val_loader, t_candidates, device)

        run["temp_scaling"] = {
            "ece_wo": temperature_scaling_ece(model_wo, test_loader, t_wo, device),
            "ece_w": temperature_scaling_ece(model_w, test_loader, t_w, device),
            "T_wo": t_wo,
            "T_w": t_w,
        }

    # Vector scaling
    if "vec_scaling" in methods:
        vs_wo = train_vector_scaling(model_wo, val_loader, num_classes, device=device)
        vs_w = train_vector_scaling(model_w, val_loader, num_classes, device=device)

        run["vec_scaling"] = {
            "ece_wo": vector_scaling_ece(model_wo, vs_wo, test_loader, device),
            "ece_w": vector_scaling_ece(model_w, vs_w, test_loader, device),
        }

    # Isotonic regression
    if "isotonic" in methods:
        ir_wo = train_isotonic_regression(model_wo, val_loader, device)
        ir_w = train_isotonic_regression(model_w, val_loader, device)

        run["isotonic"] = {
            "ece_wo": isotonic_regression_ece(model_wo, ir_wo, test_loader, device=device),
            "ece_w": isotonic_regression_ece(model_w, ir_w, test_loader, device=device),
        }

    return run


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_path = resolve_exp_path(args)

    # Load config
    with open(os.path.join(exp_path, "config.json")) as f:
        config = json.load(f)
    num_classes = config.get("num_classes", 10)
    model_name = config.get("model", "resnet18")

    # Determine methods to run
    if args.methods == "all":
        methods = ALL_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Load models (once)
    model_wo = get_model(model_name, num_classes=num_classes)
    model_wo.load_state_dict(torch.load(
        os.path.join(exp_path, "checkpoints", "model_wo.pth"), map_location=device))
    model_wo.to(device).eval()

    model_w = get_model(model_name, num_classes=num_classes)
    model_w.load_state_dict(torch.load(
        os.path.join(exp_path, "checkpoints", "model_w.pth"), map_location=device))
    model_w.to(device).eval()

    # Load dataset (once)
    test_transform = get_cifar10_transform()
    cifar10_test = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=test_transform)

    # Collect per-repeat results
    all_runs = []
    for repeat_idx in range(args.n_repeat):
        np.random.seed(args.seed + repeat_idx)

        cifar10_val, cifar10_test_split = split_dataset(
            cifar10_test, validation_size=args.val_size, test_size=args.test_size)

        val_loader = torch.utils.data.DataLoader(
            cifar10_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            cifar10_test_split, batch_size=args.batch_size, shuffle=False, num_workers=0)

        run = _evaluate_single_split(
            model_wo, model_w, val_loader, test_loader,
            methods, num_classes, device)
        all_runs.append(run)

        if args.n_repeat > 1:
            print(f"Repeat {repeat_idx + 1}/{args.n_repeat} done")

    # Aggregate results
    if args.n_repeat == 1:
        results = all_runs[0]
        for method, vals in results.items():
            print(f"{method} ECE (w/o): {vals['ece_wo']:.4f}")
            print(f"{method} ECE (w/) : {vals['ece_w']:.4f}")
    else:
        results = {"n_repeat": args.n_repeat, "seed": args.seed}
        for method in methods:
            if method not in all_runs[0]:
                continue
            method_result = {}
            keys = list(all_runs[0][method].keys())
            for key in keys:
                values = [run[method][key] for run in all_runs]
                if key.startswith("ece_"):
                    method_result[key] = float(np.mean(values))
                    method_result[f"{key}_list"] = values
                else:
                    method_result[f"{key}_list"] = values
            results[method] = method_result

            ece_wo = method_result["ece_wo"]
            ece_w = method_result["ece_w"]
            std_wo = float(np.std(method_result["ece_wo_list"]))
            std_w = float(np.std(method_result["ece_w_list"]))
            print(f"{method} ECE (w/o): {ece_wo:.4f} +/- {std_wo:.4f}")
            print(f"{method} ECE (w/) : {ece_w:.4f} +/- {std_w:.4f}")

    output_path = os.path.join(exp_path, "calibration_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
