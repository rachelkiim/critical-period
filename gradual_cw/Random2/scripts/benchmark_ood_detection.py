#!/usr/bin/env python3
"""Benchmark OOD detection comparison: evaluate OOD detection methods."""

import sys
import os
import json
import argparse
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torchvision

from src.data.dataloader import (
    get_cifar10_transform, split_dataset,
    CIFAR10_MEAN, CIFAR10_STD,
)
from src.models import get_model
from src.baseline.ood_methods import (
    msp_score,
    temperature_scaled_score,
    odin_score,
    compute_scores_for_loader,
    compute_auroc,
    find_best_params,
)
from src.logger import NumpyEncoder


ALL_METHODS = ["baseline", "temp_scaling", "odin", "energy_score"]

T_CANDIDATES = [1.5, 2, 5, 10, 20]
EPSILON_CANDIDATES = [0.001, 0.01, 0.02, 0.05]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OOD detection comparison")
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
    parser.add_argument("--ood_max_samples", type=int, default=10000,
                        help="Maximum OOD samples to use")
    return parser.parse_args()


def resolve_exp_path(args):
    if args.exp_path:
        return args.exp_path
    return os.path.join("exp", args.exp_id)


def _evaluate_ood_method(model, id_test_loader, ood_test_loader, score_fn, device):
    """Evaluate a single OOD method: compute ID/OOD scores and AUROC."""
    id_scores = compute_scores_for_loader(model, id_test_loader, score_fn, device)
    ood_scores = compute_scores_for_loader(model, ood_test_loader, score_fn, device)
    auroc = compute_auroc(id_scores, ood_scores)
    return auroc


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

    if args.methods == "all":
        methods = ALL_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Load models
    model_wo = get_model(model_name, num_classes=num_classes)
    model_wo.load_state_dict(torch.load(
        os.path.join(exp_path, "checkpoints", "model_wo.pth"), map_location=device))
    model_wo.to(device).eval()

    model_w = get_model(model_name, num_classes=num_classes)
    model_w.load_state_dict(torch.load(
        os.path.join(exp_path, "checkpoints", "model_w.pth"), map_location=device))
    model_w.to(device).eval()

    # Load datasets
    test_transform = get_cifar10_transform()

    cifar10_test = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=test_transform)

    svhn_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    svhn_test = torchvision.datasets.SVHN(
        root=args.data_path, split='test', download=True, transform=svhn_transform)

    # Limit OOD samples
    svhn_test.data = svhn_test.data[:args.ood_max_samples]
    svhn_test.targets = svhn_test.labels[:args.ood_max_samples]

    # Split into val/test
    cifar10_val, cifar10_test_split = split_dataset(
        cifar10_test, args.val_size, args.test_size)
    svhn_val, svhn_test_split = split_dataset(
        svhn_test, args.val_size, args.test_size)

    id_val_loader = torch.utils.data.DataLoader(
        cifar10_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    id_test_loader = torch.utils.data.DataLoader(
        cifar10_test_split, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ood_val_loader = torch.utils.data.DataLoader(
        svhn_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ood_test_loader = torch.utils.data.DataLoader(
        svhn_test_split, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = {}

    # Baseline (MSP)
    if "baseline" in methods:
        auroc_wo = _evaluate_ood_method(
            model_wo, id_test_loader, ood_test_loader, msp_score, device)
        auroc_w = _evaluate_ood_method(
            model_w, id_test_loader, ood_test_loader, msp_score, device)

        results["baseline"] = {"auroc_wo": auroc_wo, "auroc_w": auroc_w}
        print(f"Baseline AUROC (w/o): {auroc_wo:.4f}")
        print(f"Baseline AUROC (w/) : {auroc_w:.4f}")

    # Temperature scaling
    if "temp_scaling" in methods:
        param_grid = [{"T": t} for t in T_CANDIDATES]

        def temp_factory(params):
            return partial(temperature_scaled_score, T=params["T"])

        best_wo, _ = find_best_params(
            model_wo, id_val_loader, ood_val_loader, param_grid, temp_factory, device)
        best_w, _ = find_best_params(
            model_w, id_val_loader, ood_val_loader, param_grid, temp_factory, device)

        score_fn_wo = partial(temperature_scaled_score, T=best_wo["T"])
        score_fn_w = partial(temperature_scaled_score, T=best_w["T"])

        auroc_wo = _evaluate_ood_method(
            model_wo, id_test_loader, ood_test_loader, score_fn_wo, device)
        auroc_w = _evaluate_ood_method(
            model_w, id_test_loader, ood_test_loader, score_fn_w, device)

        results["temp_scaling"] = {
            "auroc_wo": auroc_wo, "auroc_w": auroc_w,
            "T_wo": best_wo["T"], "T_w": best_w["T"],
        }
        print(f"Temp scaling AUROC (w/o): {auroc_wo:.4f} (T={best_wo['T']})")
        print(f"Temp scaling AUROC (w/) : {auroc_w:.4f} (T={best_w['T']})")

    # ODIN
    if "odin" in methods:
        param_grid = [{"T": t, "epsilon": eps}
                      for t in T_CANDIDATES for eps in EPSILON_CANDIDATES]

        def odin_factory(params):
            return partial(odin_score, T=params["T"], epsilon=params["epsilon"])

        best_wo, _ = find_best_params(
            model_wo, id_val_loader, ood_val_loader, param_grid, odin_factory, device)
        best_w, _ = find_best_params(
            model_w, id_val_loader, ood_val_loader, param_grid, odin_factory, device)

        score_fn_wo = partial(odin_score, T=best_wo["T"], epsilon=best_wo["epsilon"])
        score_fn_w = partial(odin_score, T=best_w["T"], epsilon=best_w["epsilon"])

        auroc_wo = _evaluate_ood_method(
            model_wo, id_test_loader, ood_test_loader, score_fn_wo, device)
        auroc_w = _evaluate_ood_method(
            model_w, id_test_loader, ood_test_loader, score_fn_w, device)

        results["odin"] = {
            "auroc_wo": auroc_wo, "auroc_w": auroc_w,
            "T_wo": best_wo["T"], "T_w": best_w["T"],
            "epsilon_wo": best_wo["epsilon"], "epsilon_w": best_w["epsilon"],
        }
        print(f"ODIN AUROC (w/o): {auroc_wo:.4f} (T={best_wo['T']}, eps={best_wo['epsilon']})")
        print(f"ODIN AUROC (w/) : {auroc_w:.4f} (T={best_w['T']}, eps={best_w['epsilon']})")

    # Energy score (uses ODIN with energy=True)
    if "energy_score" in methods:
        param_grid = [{"T": t, "epsilon": eps}
                      for t in T_CANDIDATES for eps in EPSILON_CANDIDATES]

        def energy_factory(params):
            return partial(odin_score, T=params["T"], epsilon=params["epsilon"],
                           use_energy=True)

        best_wo, _ = find_best_params(
            model_wo, id_val_loader, ood_val_loader, param_grid, energy_factory, device)
        best_w, _ = find_best_params(
            model_w, id_val_loader, ood_val_loader, param_grid, energy_factory, device)

        score_fn_wo = partial(odin_score, T=best_wo["T"], epsilon=best_wo["epsilon"],
                              use_energy=True)
        score_fn_w = partial(odin_score, T=best_w["T"], epsilon=best_w["epsilon"],
                             use_energy=True)

        auroc_wo = _evaluate_ood_method(
            model_wo, id_test_loader, ood_test_loader, score_fn_wo, device)
        auroc_w = _evaluate_ood_method(
            model_w, id_test_loader, ood_test_loader, score_fn_w, device)

        results["energy_score"] = {
            "auroc_wo": auroc_wo, "auroc_w": auroc_w,
            "T_wo": best_wo["T"], "T_w": best_w["T"],
            "epsilon_wo": best_wo["epsilon"], "epsilon_w": best_w["epsilon"],
        }
        print(f"Energy score AUROC (w/o): {auroc_wo:.4f} (T={best_wo['T']}, eps={best_wo['epsilon']})")
        print(f"Energy score AUROC (w/) : {auroc_w:.4f} (T={best_w['T']}, eps={best_w['epsilon']})")

    output_path = os.path.join(exp_path, "ood_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
