#!/usr/bin/env python
"""Training stage for calibration experiment."""

import json
import os
import sys
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.config import ExperimentConfig, parse_args, setup_experiment_dirs
from src.data.dataloader import load_datasets
from src.evaluation.calibration import ece, reliability_diagram
from src.evaluation.inference import inference
from src.logger import ExperimentLogger, JSONLogger
from src.models.mlp import LinearNetwork
from src.training.random_training import random_train, random_validation
from src.training.training import train, validation

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


def _to_list(values):
    """Convert list-like values to a plain Python list."""
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


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


def generate_training_figures(training: dict, save_dir: Path, file_prefix: str = ""):
    """Generate training-related figures."""
    from custom.figure import GraphConfig as c
    from src.visualization.calibration_plots import (
        plot_conf_acc_line,
        plot_conf_acc_line_aggregate,
        plot_conf_acc_scatter,
        plot_ece_curves,
        plot_loss_acc_trajectory,
        plot_reliability_diagram,
    )
    from src.visualization.training_plots import (
        plot_training_acc,
        plot_training_curves_multi,
        plot_training_loss,
        plot_validation_acc,
        plot_validation_loss_all_aggregate,
        plot_validation_loss_all,
        plot_validation_loss,
        plot_warmup_curves,
    )

    paths = {}

    networks_wo = training.get("without_warmup", {}).get("networks", [])
    networks_w = training.get("with_warmup", {}).get("networks", [])

    if not networks_wo or not networks_w:
        print("Warning: No training results found")
        return paths

    save_dir.mkdir(parents=True, exist_ok=True)

    net_wo = networks_wo[0]
    net_w = networks_w[0]

    warmup0 = net_w.get("warmup", {})
    warmup0_train_loss = warmup0.get("train_loss", [])
    warmup0_train_acc = warmup0.get("train_acc", [])
    if warmup0_train_loss and warmup0_train_acc:
        warmup_paths = plot_warmup_curves(
            warmup0_train_loss,
            warmup0_train_acc,
            str(save_dir),
            file_prefix=file_prefix,
        )
        paths.update(warmup_paths)
        print("  Generated warmup curves")

    path = save_dir / f"{file_prefix}training_loss.svg"
    plot_training_loss(net_wo["train_loss"], net_w["train_loss"], str(path))
    paths["training_loss"] = str(path)

    path = save_dir / f"{file_prefix}training_acc.svg"
    plot_training_acc(net_wo["train_acc"], net_w["train_acc"], str(path))
    paths["training_acc"] = str(path)

    path = save_dir / f"{file_prefix}validation_loss.svg"
    plot_validation_loss(net_wo["test_loss"], net_w["test_loss"], str(path))
    paths["validation_loss"] = str(path)

    path = save_dir / f"{file_prefix}validation_acc.svg"
    plot_validation_acc(net_wo["test_acc"], net_w["test_acc"], str(path))
    paths["validation_acc"] = str(path)
    print("  Generated core training/validation curves")

    path = save_dir / f"{file_prefix}ece_curves.svg"
    plot_ece_curves(net_wo["test_ece"], net_w["test_ece"], str(path))
    paths["ece_curves"] = str(path)
    num_pairs = min(len(networks_wo), len(networks_w))
    for net_id in range(num_pairs):
        selected_epochs_wo = networks_wo[net_id].get("selected_epochs")
        selected_epochs_w = networks_w[net_id].get("selected_epochs")

        for selected_epochs, suffix, color in [
            (selected_epochs_wo, "wo", c.color.ORANGE),
            (selected_epochs_w, "w", c.color.SKY),
        ]:
            if not selected_epochs:
                print(f"  Warning: selected_epochs not found for net {net_id} ({suffix}); skipping reliability figures")
                continue

            for point_key in ["final", "acc025"]:
                selected_point = selected_epochs.get(point_key)
                if not selected_point:
                    print(f"  Warning: selected_epochs.{point_key} missing for net {net_id} ({suffix})")
                    continue

                rd = selected_point.get("reliability_diagram", {})
                if not all(k in rd for k in ("acc", "conf_bin", "num_sample")):
                    print(f"  Warning: reliability_diagram missing fields for net {net_id} ({suffix}, {point_key})")
                    continue

                path = save_dir / f"{file_prefix}reliability_{point_key}_{suffix}_net{net_id}.svg"
                plot_reliability_diagram(
                    rd["acc"],
                    rd["conf_bin"],
                    rd["num_sample"],
                    str(path),
                    title=f"{'w/o' if suffix == 'wo' else 'w/'} warmup (net {net_id}, {point_key})",
                    color=color,
                )
                paths[f"reliability_{point_key}_{suffix}_net{net_id}"] = str(path)
    print("  Generated selected-epoch reliability diagrams")

    conf_wo_all = [net.get("test_conf", []) for net in networks_wo[:num_pairs]]
    acc_wo_all = [net.get("test_acc", []) for net in networks_wo[:num_pairs]]
    conf_w_all = [net.get("test_conf", []) for net in networks_w[:num_pairs]]
    acc_w_all = [net.get("test_acc", []) for net in networks_w[:num_pairs]]
    warmup_conf_all = [
        networks_w[i].get("warmup", {}).get("test_conf", [])
        for i in range(num_pairs)
    ]
    warmup_acc_all = [
        networks_w[i].get("warmup", {}).get("test_acc", [])
        for i in range(num_pairs)
    ]
    warmup_loss_all = [
        networks_w[i].get("warmup", {}).get("test_loss", [])
        for i in range(num_pairs)
    ]
    warmup_epochs_global = 0
    for i in range(num_pairs):
        warmup_i = networks_w[i].get("warmup", {})
        warmup_lengths = [
            len(warmup_i.get("test_acc", [])),
            len(warmup_i.get("test_loss", [])),
            len(warmup_i.get("train_acc", [])),
            len(warmup_i.get("train_loss", [])),
        ]
        warmup_epochs_global = max(warmup_epochs_global, max(warmup_lengths) - 1)

    path = save_dir / f"{file_prefix}conf_acc_wo.svg"
    plot_conf_acc_line_aggregate(
        conf_wo_all,
        acc_wo_all,
        str(path),
        color=c.color.ORANGE,
        title="w/o warmup (mean±std)",
        warmup_epochs=warmup_epochs_global,
        reserve_warmup_window=warmup_epochs_global > 0,
    )
    paths["conf_acc_line_wo"] = str(path)

    path = save_dir / f"{file_prefix}conf_acc_w.svg"
    plot_conf_acc_line_aggregate(
        conf_w_all,
        acc_w_all,
        str(path),
        color=c.color.SKY,
        title="w/ warmup (mean±std)",
        warmup_conf_list=warmup_conf_all,
        warmup_acc_list=warmup_acc_all,
        warmup_epochs=warmup_epochs_global,
        reserve_warmup_window=False,
    )
    paths["conf_acc_line_w"] = str(path)

    loss_wo_all = [net.get("test_loss", []) for net in networks_wo[:num_pairs]]
    loss_w_all = [net.get("test_loss", []) for net in networks_w[:num_pairs]]
    path = save_dir / f"{file_prefix}validation_loss_all.svg"
    plot_validation_loss_all_aggregate(
        loss_wo_all,
        loss_w_all,
        str(path),
        warmup_loss_list=warmup_loss_all,
        warmup_epochs=warmup_epochs_global,
    )
    paths["validation_loss_all"] = str(path)

    for net_id in range(num_pairs):
        net_wo_i = networks_wo[net_id]
        net_w_i = networks_w[net_id]
        warmup_i = net_w_i.get("warmup", {})

        warmup_conf = warmup_i.get("test_conf", [])
        warmup_acc = warmup_i.get("test_acc", [])
        warmup_loss = warmup_i.get("test_loss", [])

        warmup_lengths = [
            len(warmup_i.get("test_acc", [])),
            len(warmup_i.get("test_loss", [])),
            len(warmup_i.get("train_acc", [])),
            len(warmup_i.get("train_loss", [])),
        ]
        warmup_epochs = max(max(warmup_lengths) - 1, 0)
        has_warmup_conf_acc = bool(warmup_conf) and bool(warmup_acc)
        has_warmup_loss_acc = bool(warmup_loss) and bool(warmup_acc)

        path = save_dir / f"{file_prefix}validation_loss_all_net{net_id}.svg"
        plot_validation_loss_all(
            net_wo_i.get("test_loss", []),
            net_w_i.get("test_loss", []),
            str(path),
            warmup_loss=warmup_loss if warmup_loss else None,
            warmup_epochs=warmup_epochs,
        )
        paths[f"validation_loss_all_net{net_id}"] = str(path)

        path = save_dir / f"{file_prefix}conf_acc_wo_net{net_id}.svg"
        plot_conf_acc_line(
            net_wo_i.get("test_conf", []),
            net_wo_i.get("test_acc", []),
            str(path),
            color=c.color.ORANGE,
            title=f"w/o warmup (net {net_id})",
            warmup_epochs=warmup_epochs,
            reserve_warmup_window=warmup_epochs > 0,
        )
        paths[f"conf_acc_line_wo_net{net_id}"] = str(path)

        path = save_dir / f"{file_prefix}conf_acc_w_net{net_id}.svg"
        plot_conf_acc_line(
            net_w_i.get("test_conf", []),
            net_w_i.get("test_acc", []),
            str(path),
            color=c.color.SKY,
            title=f"w/ warmup (net {net_id})",
            warmup_conf=warmup_conf if has_warmup_conf_acc else None,
            warmup_acc=warmup_acc if has_warmup_conf_acc else None,
            warmup_epochs=warmup_epochs,
            reserve_warmup_window=False,
        )
        paths[f"conf_acc_line_w_net{net_id}"] = str(path)

        conf_acc_scatter_paths = [save_dir / f"{file_prefix}conf_acc_scatter_net{net_id}.svg"]
        if net_id == 0:
            conf_acc_scatter_paths.append(save_dir / f"{file_prefix}conf_acc_scatter.svg")
        for path in conf_acc_scatter_paths:
            plot_conf_acc_scatter(
                net_wo_i.get("test_conf", []),
                net_wo_i.get("test_acc", []),
                net_w_i.get("test_conf", []),
                net_w_i.get("test_acc", []),
                str(path),
                include_warmup_phase=has_warmup_conf_acc,
                warmup_conf=warmup_conf if has_warmup_conf_acc else None,
                warmup_acc=warmup_acc if has_warmup_conf_acc else None,
            )
        paths[f"conf_acc_scatter_net{net_id}"] = str(conf_acc_scatter_paths[0])
        if net_id == 0:
            paths["conf_acc_scatter"] = str(conf_acc_scatter_paths[1])

        loss_acc_val_paths = [save_dir / f"{file_prefix}loss_acc_val_net{net_id}.svg"]
        if net_id == 0:
            loss_acc_val_paths.append(save_dir / f"{file_prefix}loss_acc_val.svg")
        for path in loss_acc_val_paths:
            plot_loss_acc_trajectory(
                net_wo_i.get("test_loss", []),
                net_wo_i.get("test_acc", []),
                net_w_i.get("test_loss", []),
                net_w_i.get("test_acc", []),
                str(path),
                include_warmup=has_warmup_loss_acc,
                warmup_loss=warmup_loss if has_warmup_loss_acc else None,
                warmup_acc=warmup_acc if has_warmup_loss_acc else None,
                chance_acc=0.25,
                chance_acc_secondary=0.1,
                chance_linewidth=0.5,
                add_markers=True,
            )
        paths[f"loss_acc_val_net{net_id}"] = str(loss_acc_val_paths[0])
        if net_id == 0:
            paths["loss_acc_val"] = str(loss_acc_val_paths[1])
    print("  Generated per-network confidence/loss calibration figures")

    if len(networks_wo) > 1 and len(networks_w) > 1:
        multi_paths = plot_training_curves_multi(
            networks_wo,
            networks_w,
            str(save_dir),
            file_prefix=f"{file_prefix}multi_",
        )
        paths.update({f"multi_{k}": v for k, v in multi_paths.items()})
        print("  Generated multi-network training curves")

    return paths


def train_network(
    net_id: int,
    config: ExperimentConfig,
    train_loader,
    test_loader,
    device: torch.device,
    with_warmup: bool,
    logger: ExperimentLogger,
):
    """Train a single network with or without random warmup."""
    num_hidden_list = [config.width] * (config.depth - 1) + [config.num_classes]
    model = LinearNetwork(
        in_features=3072,
        num_layers=config.depth,
        num_hidden_list=num_hidden_list,
        mode=config.mode,
        w_seed=net_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    metrics = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "train_ece": [], "train_conf": [],
        "test_ece": [], "test_conf": [],
        "selected_epochs": {},
    }
    warmup_metrics = None
    test_reliability_history = []

    if with_warmup:
        warmup_metrics = {
            "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [],
            "test_conf": [], "test_ece": [],
        }
        optimizer_warmup = torch.optim.Adam(
            model.parameters(),
            lr=config.lr_noise,
            weight_decay=config.weight_decay_noise,
        )

        input_shape = (3, 32, 32)
        print(f"  [Net {net_id}] Starting random noise warmup...")

        for epoch in range(config.epochs_noise + 1):
            if epoch == 0:
                loss, acc = random_validation(
                    model,
                    criterion,
                    input_shape,
                    config.num_noise,
                    config.batch_size,
                    config.num_classes,
                    device=device,
                )
            else:
                loss, acc = random_train(
                    model,
                    optimizer_warmup,
                    criterion,
                    input_shape,
                    config.num_noise,
                    config.batch_size,
                    config.num_classes,
                    device=device,
                )

            test_loss, test_acc = validation(model, test_loader, criterion, device=device)
            test_pred, test_conf_scores = inference(model, test_loader, device=device)
            test_acc_bins, test_conf_bins, test_num_samples = reliability_diagram(test_pred, test_conf_scores)
            test_ece = ece(test_acc_bins, test_conf_bins, test_num_samples)
            test_conf = float(np.mean(test_conf_scores))

            warmup_metrics["train_loss"].append(loss)
            warmup_metrics["train_acc"].append(acc)
            warmup_metrics["test_loss"].append(test_loss)
            warmup_metrics["test_acc"].append(test_acc)
            warmup_metrics["test_conf"].append(test_conf)
            warmup_metrics["test_ece"].append(test_ece)

            logger.log_metrics(
                {
                    "loss": loss,
                    "acc": acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_conf": test_conf,
                    "test_ece": test_ece,
                },
                step=epoch,
                prefix=f"warmup/net{net_id}/",
            )

            if epoch % 5 == 0 or epoch == config.epochs_noise:
                print(
                    f"    Warmup Epoch {epoch}/{config.epochs_noise}: "
                    f"Loss={loss:.4f}, Acc={acc:.4f}, "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
                )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    prefix_type = "train_w" if with_warmup else "train_wo"
    print(
        f"  [Net {net_id}] Starting main training "
        f"({'with' if with_warmup else 'without'} warmup)..."
    )

    for epoch in range(config.epochs + 1):
        if epoch == 0:
            train_loss, train_acc = validation(model, train_loader, criterion, device=device)
        else:
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device=device)

        test_loss, test_acc = validation(model, test_loader, criterion, device=device)

        train_pred, train_conf_scores = inference(model, train_loader, device=device)
        test_pred, test_conf_scores = inference(model, test_loader, device=device)

        train_acc_bins, train_conf_bins, train_num_samples = reliability_diagram(train_pred, train_conf_scores)
        test_acc_bins, test_conf_bins, test_num_samples = reliability_diagram(test_pred, test_conf_scores)

        train_ece = ece(train_acc_bins, train_conf_bins, train_num_samples)
        test_ece = ece(test_acc_bins, test_conf_bins, test_num_samples)

        train_conf = float(np.mean(train_conf_scores))
        test_conf = float(np.mean(test_conf_scores))

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)
        metrics["train_ece"].append(train_ece)
        metrics["train_conf"].append(train_conf)
        metrics["test_ece"].append(test_ece)
        metrics["test_conf"].append(test_conf)
        test_reliability_history.append(
            {
                "acc": _to_list(test_acc_bins),
                "conf_bin": _to_list(test_conf_bins),
                "num_sample": _to_list(test_num_samples),
            }
        )

        logger.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "train_ece": train_ece,
                "train_conf": train_conf,
                "test_ece": test_ece,
                "test_conf": test_conf,
            },
            step=epoch,
            prefix=f"{prefix_type}/net{net_id}/",
        )

        if epoch % 5 == 0 or epoch == config.epochs:
            print(
                f"    Epoch {epoch}/{config.epochs}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Test Acc={test_acc:.4f}, Test ECE={test_ece:.4f}"
            )

    final_epoch = len(metrics["test_acc"]) - 1
    acc025_epoch = int(np.argmin(np.abs(np.array(metrics["test_acc"]) - 0.25)))

    def build_selected_epoch(epoch_idx: int) -> dict:
        return {
            "epoch": int(epoch_idx),
            "acc": float(metrics["test_acc"][epoch_idx]),
            "ece": float(metrics["test_ece"][epoch_idx]),
            "reliability_diagram": test_reliability_history[epoch_idx],
        }

    metrics["selected_epochs"] = {
        "final": build_selected_epoch(final_epoch),
        "acc025": build_selected_epoch(acc025_epoch),
    }

    suffix = "w" if with_warmup else "wo"
    model_path = logger.get_checkpoints_path() / f"final_{suffix}_{net_id}.pt"
    torch.save(model.state_dict(), model_path)

    return metrics, warmup_metrics


def main():
    """Training stage runner."""
    config = parse_args()

    setup_experiment_dirs(config)
    print(f"Experiment ID: {config.exp_id}")
    print(f"Output directory: {config.exp_path}")

    config.save()

    logger = ExperimentLogger(
        config=config.to_dict(),
        exp_path=config.exp_path,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        no_wandb=config.no_wandb,
    )

    logger.define_metrics_for_training(config.num_nets)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_loader, test_loader = load_datasets(config)

    results_w = []
    results_wo = []

    for net_id in range(config.num_nets):
        print(f"\n=== Training Network {net_id + 1}/{config.num_nets} ===")

        metrics_wo, _ = train_network(
            net_id,
            config,
            train_loader,
            test_loader,
            device,
            with_warmup=False,
            logger=logger,
        )
        results_wo.append(metrics_wo)
        logger.add_network_result(net_id=net_id, with_warmup=False, **metrics_wo)

        metrics_w, warmup_metrics = train_network(
            net_id,
            config,
            train_loader,
            test_loader,
            device,
            with_warmup=True,
            logger=logger,
        )
        results_w.append(metrics_w)
        logger.add_network_result(
            net_id=net_id,
            with_warmup=True,
            warmup_metrics=warmup_metrics,
            **metrics_w,
        )

    final_metrics = {
        "test_acc_wo": float(np.mean([r["test_acc"][-1] for r in results_wo])),
        "test_acc_w": float(np.mean([r["test_acc"][-1] for r in results_w])),
        "test_ece_wo": float(np.mean([r["test_ece"][-1] for r in results_wo])),
        "test_ece_w": float(np.mean([r["test_ece"][-1] for r in results_w])),
        "acc_conf_gap_wo": float(np.mean([r["test_conf"][-1] - r["test_acc"][-1] for r in results_wo])),
        "acc_conf_gap_w": float(np.mean([r["test_conf"][-1] - r["test_acc"][-1] for r in results_w])),
    }
    logger.log_final_metrics(final_metrics)

    logger.json_logger.save_training_results()

    if not config.no_figures:
        print("\n=== Generating Training Figures ===")
        training_results = JSONLogger.load_json(str(Path(config.exp_path) / "training_results.json"))
        save_dir = Path(config.exp_path) / "figures" / "training"
        figure_paths = generate_training_figures(training_results, save_dir)
        merge_figure_paths(Path(config.exp_path), figure_paths)
        for name, path in figure_paths.items():
            logger.log_figure(name, path)

    logger.wandb_logger.finish()

    print(f"\nResults saved to: {config.exp_path}")
    print(f"EXP_ID={config.exp_id}")
    print(f"EXP_PATH={config.exp_path}")


if __name__ == "__main__":
    main()
