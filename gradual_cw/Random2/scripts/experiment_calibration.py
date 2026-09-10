#!/usr/bin/env python
"""Calibration stage for calibration experiment."""

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
from src.data.dataloader import load_datasets
from src.evaluation.calibration import ece, reliability_diagram
from src.evaluation.inference import inference
from src.logger import JSONLogger
from src.models.mlp import LinearNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibration stage on an existing experiment")
    parser.add_argument("--exp_path", type=str, default=None,
                        help="Absolute or relative path to experiment directory")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment ID under exp_base_path")
    parser.add_argument("--exp_base_path", type=str, default="./exp",
                        help="Base path used with --exp_id (default: ./exp)")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip calibration figure generation")
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


def generate_calibration_figures(calibration: dict, save_dir: Path, file_prefix: str = ""):
    """Generate calibration-related figures."""
    from custom.figure import GraphConfig as c
    from src.visualization.calibration_plots import (
        plot_aggregate_boxplots,
        plot_ece_bar,
        plot_gap_bar,
        plot_reliability_diagram,
    )

    paths = {}

    calib_wo = calibration.get("without_warmup", {})
    calib_w = calibration.get("with_warmup", {})

    if not calib_wo or not calib_w:
        print("Warning: No calibration results found")
        return paths

    save_dir.mkdir(parents=True, exist_ok=True)

    if "reliability_diagram" in calib_wo:
        rd = calib_wo["reliability_diagram"]
        path = save_dir / f"{file_prefix}reliability_diagram_wo.svg"
        plot_reliability_diagram(
            rd["acc"],
            rd["conf_bin"],
            rd["num_sample"],
            str(path),
            title="w/o warmup",
            color=c.color.ORANGE,
        )
        paths["reliability_wo"] = str(path)

    if "reliability_diagram" in calib_w:
        rd = calib_w["reliability_diagram"]
        path = save_dir / f"{file_prefix}reliability_diagram_w.svg"
        plot_reliability_diagram(
            rd["acc"],
            rd["conf_bin"],
            rd["num_sample"],
            str(path),
            title="w/ warmup",
            color=c.color.SKY,
        )
        paths["reliability_w"] = str(path)
    print("  Generated reliability diagrams")

    if "reliability_diagram_list" in calib_wo:
        for net_id, rd in enumerate(calib_wo["reliability_diagram_list"]):
            path = save_dir / f"{file_prefix}reliability_diagram_wo_net{net_id}.svg"
            plot_reliability_diagram(
                rd["acc"],
                rd["conf_bin"],
                rd["num_sample"],
                str(path),
                title=f"w/o warmup (net {net_id})",
                color=c.color.ORANGE,
            )
            paths[f"reliability_wo_net{net_id}"] = str(path)

    if "reliability_diagram_list" in calib_w:
        for net_id, rd in enumerate(calib_w["reliability_diagram_list"]):
            path = save_dir / f"{file_prefix}reliability_diagram_w_net{net_id}.svg"
            plot_reliability_diagram(
                rd["acc"],
                rd["conf_bin"],
                rd["num_sample"],
                str(path),
                title=f"w/ warmup (net {net_id})",
                color=c.color.SKY,
            )
            paths[f"reliability_w_net{net_id}"] = str(path)

    if "ece_list" in calib_wo and "ece_list" in calib_w:
        path = save_dir / f"{file_prefix}ece_bar.svg"
        plot_ece_bar(calib_wo["ece_list"], calib_w["ece_list"], str(path))
        paths["ece_bar"] = str(path)

        path = save_dir / f"{file_prefix}ece_boxplot.svg"
        plot_aggregate_boxplots(
            calib_wo["ece_list"],
            calib_w["ece_list"],
            str(path),
            ylabel="ECE",
        )
        paths["ece_boxplot"] = str(path)

    if "gap_list" in calib_wo and "gap_list" in calib_w:
        path = save_dir / f"{file_prefix}gap_bar.svg"
        plot_gap_bar(calib_wo["gap_list"], calib_w["gap_list"], str(path))
        paths["gap_bar"] = str(path)

        path = save_dir / f"{file_prefix}gap_boxplot.svg"
        plot_aggregate_boxplots(
            calib_wo["gap_list"],
            calib_w["gap_list"],
            str(path),
            ylabel="Acc-Conf Gap",
        )
        paths["gap_boxplot"] = str(path)

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

    reliability_diagram_wo_list = []
    reliability_diagram_w_list = []
    ece_wo_list = []
    ece_w_list = []
    gap_wo_list = []
    gap_w_list = []

    for net_id in range(config.num_nets):
        print(f"Computing calibration for network {net_id + 1}/{config.num_nets}")
        model_wo = load_model(config, exp_path, net_id, with_warmup=False, device=device)
        model_w = load_model(config, exp_path, net_id, with_warmup=True, device=device)

        pred_wo, conf_wo = inference(model_wo, test_loader, device=device)
        pred_w, conf_w = inference(model_w, test_loader, device=device)

        acc_bins_wo, conf_bins_wo, num_samples_wo = reliability_diagram(pred_wo, conf_wo)
        acc_bins_w, conf_bins_w, num_samples_w = reliability_diagram(pred_w, conf_w)

        ece_wo = ece(acc_bins_wo, conf_bins_wo, num_samples_wo)
        ece_w = ece(acc_bins_w, conf_bins_w, num_samples_w)

        acc_wo = float(np.mean(pred_wo))
        acc_w = float(np.mean(pred_w))
        mean_conf_wo = float(np.mean(conf_wo))
        mean_conf_w = float(np.mean(conf_w))

        reliability_diagram_wo_list.append(
            {
                "acc": acc_bins_wo,
                "conf_bin": conf_bins_wo.tolist(),
                "num_sample": num_samples_wo,
            }
        )
        reliability_diagram_w_list.append(
            {
                "acc": acc_bins_w,
                "conf_bin": conf_bins_w.tolist(),
                "num_sample": num_samples_w,
            }
        )

        ece_wo_list.append(float(ece_wo))
        ece_w_list.append(float(ece_w))
        gap_wo_list.append(mean_conf_wo - acc_wo)
        gap_w_list.append(mean_conf_w - acc_w)

    calibration_results = {
        "without_warmup": {
            "reliability_diagram": reliability_diagram_wo_list[0],
            "ece_final": float(np.mean(ece_wo_list)),
            "acc_conf_gap": float(np.mean(gap_wo_list)),
            "ece_list": ece_wo_list,
            "gap_list": gap_wo_list,
            "reliability_diagram_list": reliability_diagram_wo_list,
        },
        "with_warmup": {
            "reliability_diagram": reliability_diagram_w_list[0],
            "ece_final": float(np.mean(ece_w_list)),
            "acc_conf_gap": float(np.mean(gap_w_list)),
            "ece_list": ece_w_list,
            "gap_list": gap_w_list,
            "reliability_diagram_list": reliability_diagram_w_list,
        },
    }

    json_logger = JSONLogger(str(exp_path))
    json_logger.calibration_results.without_warmup = calibration_results["without_warmup"]
    json_logger.calibration_results.with_warmup = calibration_results["with_warmup"]
    json_logger.save_calibration_results()

    no_figures = args.no_figures or bool(config_dict.get("no_figures", False))
    if not no_figures:
        print("\n=== Generating Calibration Figures ===")
        save_dir = exp_path / "figures" / "calibration"
        figure_paths = generate_calibration_figures(calibration_results, save_dir)
        merge_figure_paths(exp_path, figure_paths)

    print(f"\nCalibration results saved to: {exp_path / 'calibration.json'}")


if __name__ == "__main__":
    main()
