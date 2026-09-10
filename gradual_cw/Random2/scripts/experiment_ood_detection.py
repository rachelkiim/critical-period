#!/usr/bin/env python
"""OOD detection stage for calibration experiment."""

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.data.dataloader import load_datasets, load_ood_datasets
from src.evaluation.inference import get_confidence_only, inference
from src.evaluation.ood_detection import auc, roc_curve
from src.logger import JSONLogger
from src.models.mlp import LinearNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Run OOD detection stage on an existing experiment")
    parser.add_argument("--exp_path", type=str, default=None,
                        help="Absolute or relative path to experiment directory")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment ID under exp_base_path")
    parser.add_argument("--exp_base_path", type=str, default="./exp",
                        help="Base path used with --exp_id (default: ./exp)")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip OOD figure generation")
    return parser.parse_args()


def resolve_exp_path(args) -> Path:
    if args.exp_path:
        return Path(args.exp_path)
    if args.exp_id:
        return Path(args.exp_base_path) / args.exp_id
    raise ValueError("Provide either --exp_path or --exp_id")


def load_config(exp_path: Path) -> dict:
    config_path = exp_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def config_from_dict(config_dict: dict, exp_path: Path) -> ExperimentConfig:
    init_fields = {f.name for f in fields(ExperimentConfig) if f.init}
    kwargs = {k: v for k, v in config_dict.items() if k in init_fields}
    config = ExperimentConfig(**kwargs)
    config.exp_path = str(exp_path)
    config.exp_id = exp_path.name
    return config


def merge_figure_paths(exp_path: Path, new_paths: dict):
    """Merge stage figure paths into figures/figure_paths.json."""
    figures_dir = exp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths_file = figures_dir / "figure_paths.json"

    all_paths = {}
    if paths_file.exists():
        with open(paths_file, "r") as f:
            all_paths = json.load(f)

    all_paths.update(new_paths)

    with open(paths_file, "w") as f:
        json.dump(all_paths, f, indent=2)


def load_model(config: ExperimentConfig, exp_path: Path, net_id: int, with_warmup: bool, device: torch.device):
    num_hidden_list = [config.width] * (config.depth - 1) + [config.num_classes]
    model = LinearNetwork(
        in_features=3072,
        num_layers=config.depth,
        num_hidden_list=num_hidden_list,
        mode=config.mode,
        w_seed=net_id,
    ).to(device)

    suffix = "w" if with_warmup else "wo"
    ckpt_path = exp_path / "checkpoints" / f"final_{suffix}_{net_id}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_ood(model, test_loader, ood_loader, device: torch.device):
    """Evaluate OOD detection performance for one model."""
    _, conf_id = inference(model, test_loader, device=device)
    conf_ood = get_confidence_only(model, ood_loader, device=device)

    fp, tp = roc_curve(conf_id, conf_ood)
    auroc = auc(fp, tp)

    return {
        "conf_id": conf_id.tolist(),
        "conf_ood": conf_ood.tolist(),
        "roc": {"fp": fp.tolist(), "tp": tp.tolist()},
        "auroc": float(auroc),
    }


def generate_ood_figures(
    ood: dict,
    save_dir: Path,
    file_prefix: str = "",
    per_net_wo: list | None = None,
    per_net_w: list | None = None,
):
    """Generate OOD detection figures."""
    from src.visualization.ood_plots import (
        plot_auroc_bar,
        plot_auroc_boxplot,
        plot_cdf,
        plot_id_ood_histogram,
        plot_ood_boxplot,
        plot_ood_histogram,
        plot_roc_curve,
    )

    paths = {}

    ood_wo = ood.get("without_warmup", {})
    ood_w = ood.get("with_warmup", {})

    if not ood_wo or not ood_w:
        print("Warning: No OOD results found")
        return paths

    save_dir.mkdir(parents=True, exist_ok=True)

    conf_id_wo = np.array(ood_wo.get("conf_id", []))
    conf_id_w = np.array(ood_w.get("conf_id", []))
    conf_ood_wo = np.array(ood_wo.get("conf_ood", []))
    conf_ood_w = np.array(ood_w.get("conf_ood", []))

    if len(conf_ood_wo) > 0 and len(conf_ood_w) > 0:
        path = save_dir / f"{file_prefix}ood_histogram.svg"
        plot_ood_histogram(conf_ood_wo, conf_ood_w, str(path))
        paths["ood_histogram"] = str(path)

        path = save_dir / f"{file_prefix}ood_boxplot.svg"
        plot_ood_boxplot(conf_ood_wo, conf_ood_w, str(path))
        paths["ood_boxplot"] = str(path)

    # Per-network OOD histogram/boxplot (w/o vs w/)
    if per_net_wo and per_net_w:
        num_pairs = min(len(per_net_wo), len(per_net_w))
        for net_id in range(num_pairs):
            conf_ood_wo_net = np.array(per_net_wo[net_id].get("conf_ood", []))
            conf_ood_w_net = np.array(per_net_w[net_id].get("conf_ood", []))
            if len(conf_ood_wo_net) == 0 or len(conf_ood_w_net) == 0:
                continue

            path = save_dir / f"{file_prefix}ood_histogram_net{net_id}.svg"
            plot_ood_histogram(conf_ood_wo_net, conf_ood_w_net, str(path))
            paths[f"ood_histogram_net{net_id}"] = str(path)

            path = save_dir / f"{file_prefix}ood_boxplot_net{net_id}.svg"
            plot_ood_boxplot(conf_ood_wo_net, conf_ood_w_net, str(path))
            paths[f"ood_boxplot_net{net_id}"] = str(path)
    else:
        # Fallback for figure generation from saved JSON only.
        conf_ood_wo_list = ood_wo.get("conf_ood_list", [])
        conf_ood_w_list = ood_w.get("conf_ood_list", [])
        num_pairs = min(len(conf_ood_wo_list), len(conf_ood_w_list))
        for net_id in range(num_pairs):
            conf_ood_wo_net = np.array(conf_ood_wo_list[net_id])
            conf_ood_w_net = np.array(conf_ood_w_list[net_id])
            if len(conf_ood_wo_net) == 0 or len(conf_ood_w_net) == 0:
                continue

            path = save_dir / f"{file_prefix}ood_histogram_net{net_id}.svg"
            plot_ood_histogram(conf_ood_wo_net, conf_ood_w_net, str(path))
            paths[f"ood_histogram_net{net_id}"] = str(path)

            path = save_dir / f"{file_prefix}ood_boxplot_net{net_id}.svg"
            plot_ood_boxplot(conf_ood_wo_net, conf_ood_w_net, str(path))
            paths[f"ood_boxplot_net{net_id}"] = str(path)

    if len(conf_id_wo) > 0 and len(conf_ood_wo) > 0:
        path = save_dir / f"{file_prefix}id_ood_wo.svg"
        plot_id_ood_histogram(conf_id_wo, conf_ood_wo, str(path), title="w/o warmup")
        paths["id_ood_wo"] = str(path)

        path = save_dir / f"{file_prefix}cdf_wo.svg"
        plot_cdf(conf_id_wo, conf_ood_wo, str(path))
        paths["cdf_wo"] = str(path)

    if len(conf_id_w) > 0 and len(conf_ood_w) > 0:
        path = save_dir / f"{file_prefix}id_ood_w.svg"
        plot_id_ood_histogram(conf_id_w, conf_ood_w, str(path), title="w/ warmup")
        paths["id_ood_w"] = str(path)

        path = save_dir / f"{file_prefix}cdf_w.svg"
        plot_cdf(conf_id_w, conf_ood_w, str(path))
        paths["cdf_w"] = str(path)

    roc_wo = ood_wo.get("roc", {})
    roc_w = ood_w.get("roc", {})
    auroc_wo_list = ood_wo.get("auroc_list", [])
    auroc_w_list = ood_w.get("auroc_list", [])
    auroc_wo_plot = ood_wo.get("auroc")
    auroc_w_plot = ood_w.get("auroc")
    if auroc_wo_plot is None and auroc_wo_list:
        auroc_wo_plot = float(np.mean(auroc_wo_list))
    if auroc_w_plot is None and auroc_w_list:
        auroc_w_plot = float(np.mean(auroc_w_list))
    if roc_wo and roc_w:
        path = save_dir / f"{file_prefix}roc_curve.svg"
        plot_roc_curve(
            np.array(roc_wo["fp"]),
            np.array(roc_wo["tp"]),
            np.array(roc_w["fp"]),
            np.array(roc_w["tp"]),
            str(path),
            auroc_wo=auroc_wo_plot,
            auroc_w=auroc_w_plot,
        )
        paths["roc_curve"] = str(path)

    roc_wo_list = ood_wo.get("roc_list", [])
    roc_w_list = ood_w.get("roc_list", [])
    if roc_wo_list and roc_w_list:
        for net_id in range(len(roc_wo_list)):
            roc_wo_net = roc_wo_list[net_id]
            roc_w_net = roc_w_list[net_id]
            auroc_wo_net = auroc_wo_list[net_id] if net_id < len(auroc_wo_list) else None
            auroc_w_net = auroc_w_list[net_id] if net_id < len(auroc_w_list) else None

            path = save_dir / f"{file_prefix}roc_curve_net{net_id}.svg"
            plot_roc_curve(
                np.array(roc_wo_net["fp"]),
                np.array(roc_wo_net["tp"]),
                np.array(roc_w_net["fp"]),
                np.array(roc_w_net["tp"]),
                str(path),
                auroc_wo=auroc_wo_net,
                auroc_w=auroc_w_net,
            )
            paths[f"roc_curve_net{net_id}"] = str(path)

    if "auroc_list" in ood_wo and "auroc_list" in ood_w:
        path = save_dir / f"{file_prefix}auroc_bar.svg"
        plot_auroc_bar(ood_wo["auroc_list"], ood_w["auroc_list"], str(path))
        paths["auroc_bar"] = str(path)

        path = save_dir / f"{file_prefix}auroc_boxplot.svg"
        plot_auroc_boxplot(ood_wo["auroc_list"], ood_w["auroc_list"], str(path))
        paths["auroc_boxplot"] = str(path)

    return paths


def main():
    args = parse_args()

    exp_path = resolve_exp_path(args)
    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {exp_path}")
        sys.exit(1)

    config_dict = load_config(exp_path)
    config = config_from_dict(config_dict, exp_path)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {exp_path}")

    _, test_loader = load_datasets(config)
    ood_loader = load_ood_datasets(config, max_samples=10000)
    print(f"Using {len(ood_loader.dataset)} seeded SVHN test samples for OOD")

    ood_results_wo_list = []
    ood_results_w_list = []

    for net_id in range(config.num_nets):
        print(f"Computing OOD metrics for network {net_id + 1}/{config.num_nets}")
        model_wo = load_model(config, exp_path, net_id, with_warmup=False, device=device)
        model_w = load_model(config, exp_path, net_id, with_warmup=True, device=device)

        ood_wo = evaluate_ood(model_wo, test_loader, ood_loader, device)
        ood_w = evaluate_ood(model_w, test_loader, ood_loader, device)

        ood_results_wo_list.append(ood_wo)
        ood_results_w_list.append(ood_w)

    auroc_wo_list = [r["auroc"] for r in ood_results_wo_list]
    auroc_w_list = [r["auroc"] for r in ood_results_w_list]
    roc_wo_list = [r["roc"] for r in ood_results_wo_list]
    roc_w_list = [r["roc"] for r in ood_results_w_list]
    conf_id_wo_list = [r["conf_id"] for r in ood_results_wo_list]
    conf_id_w_list = [r["conf_id"] for r in ood_results_w_list]
    conf_ood_wo_list = [r["conf_ood"] for r in ood_results_wo_list]
    conf_ood_w_list = [r["conf_ood"] for r in ood_results_w_list]

    ood_results_wo = ood_results_wo_list[0]
    ood_results_wo["auroc_list"] = auroc_wo_list
    ood_results_wo["roc_list"] = roc_wo_list
    ood_results_wo["conf_id_list"] = conf_id_wo_list
    ood_results_wo["conf_ood_list"] = conf_ood_wo_list
    ood_results_wo.pop("auroc", None)

    ood_results_w = ood_results_w_list[0]
    ood_results_w["auroc_list"] = auroc_w_list
    ood_results_w["roc_list"] = roc_w_list
    ood_results_w["conf_id_list"] = conf_id_w_list
    ood_results_w["conf_ood_list"] = conf_ood_w_list
    ood_results_w.pop("auroc", None)

    ood_results = {
        "without_warmup": ood_results_wo,
        "with_warmup": ood_results_w,
    }

    json_logger = JSONLogger(str(exp_path))
    json_logger.ood_results.without_warmup = ood_results["without_warmup"]
    json_logger.ood_results.with_warmup = ood_results["with_warmup"]
    json_logger.save_ood_results()

    no_figures = args.no_figures or bool(config_dict.get("no_figures", False))
    if not no_figures:
        print("\n=== Generating OOD Figures ===")
        save_dir = exp_path / "figures" / "ood_detection"
        figure_paths = generate_ood_figures(
            ood_results,
            save_dir,
            per_net_wo=ood_results_wo_list,
            per_net_w=ood_results_w_list,
        )
        merge_figure_paths(exp_path, figure_paths)

    print(f"\nOOD results saved to: {exp_path / 'ood_detection.json'}")


if __name__ == "__main__":
    main()
