#!/usr/bin/env python3
"""Initial overconfidence experiment runner.

Studies how neural networks exhibit overconfidence at initialization by:
- Exp1: Varying output classes (2-10) at fixed depth, showing confidence ~ 1/k
- Exp2: Varying depth (2-10) at fixed outputs, showing depth affects confidence and logit variance
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from src.models.mlp import LinearNetwork
from src.training.random_training import random_train
from src.data.dataloader import get_cifar10_transform
from src.evaluation.inference import get_predictions_and_confidence
from src.logger import NumpyEncoder
from src.visualization.overconfidence_plots import (
    plot_confidence_distribution,
    plot_probability_distribution,
    plot_logit_distribution,
    plot_logit_boxplot,
    plot_logit_vs_probability,
    plot_mean_confidence_vs_outputs,
    plot_mean_confidence_vs_depth,
    plot_logit_variance_vs_depth,
)
from custom.figure import GraphConfig as c

import torchvision

RAW_JSON_ROUND_DIGITS = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Initial overconfidence experiment")
    parser.add_argument("--experiment", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Which experiment to run")
    parser.add_argument("--num_net", type=int, default=10,
                        help="Number of networks per configuration")
    parser.add_argument("--num_hidden", type=int, default=100,
                        help="Hidden layer width")
    parser.add_argument("--depth", type=int, default=2,
                        help="Fixed depth for exp1")
    parser.add_argument("--num_outputs", type=int, default=2,
                        help="Fixed number of outputs for exp2")
    parser.add_argument("--num_output_list", type=str, default="2,3,4,5,6,7,8,9,10",
                        help="Comma-separated output class sweep for exp1")
    parser.add_argument("--depth_list", type=str, default="2,3,4,5,6,7,8,9,10",
                        help="Comma-separated depth sweep for exp2")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Random training epochs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_noise", type=int, default=10000,
                        help="Number of random noise samples per epoch")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="./exp")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation")

    args = parser.parse_args()
    args.num_output_list = [int(x) for x in args.num_output_list.split(",")]
    args.depth_list = [int(x) for x in args.depth_list.split(",")]
    return args


def create_model(depth, num_hidden, num_outputs, device):
    """Create LinearNetwork with appropriate num_hidden_list."""
    num_hidden_list = [num_hidden] * (depth - 1) + [num_outputs]
    model = LinearNetwork(
        in_features=32 * 32 * 3,
        num_layers=depth,
        num_hidden_list=num_hidden_list,
        mode='BP',
    ).to(device)
    return model


def measure_model(model, loader, device):
    """Measure model predictions, confidence, probabilities, and logits."""
    pred_correct, conf, pred_labels, true_labels, probs, logits = \
        get_predictions_and_confidence(model, loader, device,
                                       return_probs=True, return_logits=True)
    return pred_labels, conf, probs, logits


def run_single_experiment(depth, num_outputs, loader, device, args):
    """Run a single experiment: create model, measure untrained, train, measure trained."""
    print(f"  depth={depth}, num_outputs={num_outputs}")

    model = create_model(depth, args.num_hidden, num_outputs, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    # Measure untrained model
    untrained_pred, untrained_conf, untrained_prob, untrained_logit = \
        measure_model(model, loader, device)

    # Train on random noise
    training_loss = []
    training_acc = []
    for epoch in range(args.epochs):
        loss, acc = random_train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            input_shape=(32, 32, 3),
            num_image=args.num_noise,
            batch_size=args.batch_size,
            output_size=num_outputs,
            device=device,
        )
        training_loss.append(loss)
        training_acc.append(acc)

    # Measure trained model
    pretrained_pred, pretrained_conf, pretrained_prob, pretrained_logit = \
        measure_model(model, loader, device)

    return {
        'depth': depth,
        'num_outputs': num_outputs,
        'training_loss': training_loss,
        'training_acc': training_acc,
        'untrained_conf': untrained_conf,
        'untrained_prob': untrained_prob,
        'untrained_logit': untrained_logit,
        'pretrained_conf': pretrained_conf,
        'pretrained_prob': pretrained_prob,
        'pretrained_logit': pretrained_logit,
    }


def run_exp1(args, loader, device):
    """Exp1: Vary output classes, fixed depth."""
    print(f"Running Exp1: depth={args.depth}, outputs={args.num_output_list}")
    results = []
    for num_output in args.num_output_list:
        config_results = []
        for net_idx in range(args.num_net):
            print(f"  [Exp1] output={num_output}, net={net_idx+1}/{args.num_net}")
            info = run_single_experiment(args.depth, num_output, loader, device, args)
            config_results.append(info)
        results.append(config_results)
    return results


def run_exp2(args, loader, device):
    """Exp2: Vary depth, fixed outputs."""
    print(f"Running Exp2: depths={args.depth_list}, outputs={args.num_outputs}")
    results = []
    for depth in args.depth_list:
        config_results = []
        for net_idx in range(args.num_net):
            print(f"  [Exp2] depth={depth}, net={net_idx+1}/{args.num_net}")
            info = run_single_experiment(depth, args.num_outputs, loader, device, args)
            config_results.append(info)
        results.append(config_results)
    return results


def compute_confidence_stats(results, num_configs, num_net):
    """Compute per-network mean confidence for untrained and pretrained models."""
    conf_untrained = np.array([
        [np.mean(results[cfg_idx][net_idx]['untrained_conf'])
         for net_idx in range(num_net)]
        for cfg_idx in range(num_configs)
    ])
    conf_pretrained = np.array([
        [np.mean(results[cfg_idx][net_idx]['pretrained_conf'])
         for net_idx in range(num_net)]
        for cfg_idx in range(num_configs)
    ])
    return conf_untrained, conf_pretrained


def compute_logit_variance_stats(results, num_configs, num_net):
    """Compute per-network logit variance for untrained and pretrained models."""
    logit_untrained = np.array([
        [np.var(results[cfg_idx][net_idx]['untrained_logit'].flatten())
         for net_idx in range(num_net)]
        for cfg_idx in range(num_configs)
    ])
    logit_pretrained = np.array([
        [np.var(results[cfg_idx][net_idx]['pretrained_logit'].flatten())
         for net_idx in range(num_net)]
        for cfg_idx in range(num_configs)
    ])
    return logit_untrained, logit_pretrained


def _serialize_raw_value(value, digits=RAW_JSON_ROUND_DIGITS):
    """Convert values to JSON-safe objects with rounded floating-point numbers."""
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.integer):
            return value.tolist()
        return np.round(value.astype(np.float64), digits).tolist()
    if isinstance(value, (np.floating, float)):
        return float(np.round(float(value), digits))
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: _serialize_raw_value(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_raw_value(v, digits) for v in value]
    return value


def save_raw_json_exp1_output2(exp1_results, args, exp_path, exp_id):
    """Save figure-generation raw JSON for Exp1 at output=2 (or fallback first output)."""
    if exp1_results is None:
        return

    if 2 in args.num_output_list:
        selected_output = 2
        fallback_used = False
    else:
        selected_output = args.num_output_list[0]
        fallback_used = True

    selected_output_idx = args.num_output_list.index(selected_output)
    selected_results = exp1_results[selected_output_idx]

    raw_root = (
        Path(exp_path)
        / "raw_json"
        / "overconfidence"
        / f"exp1_output_{selected_output}"
    )
    raw_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "exp_id": exp_id,
        "selected_output": selected_output,
        "requested_output_list": args.num_output_list,
        "selected_output_index": selected_output_idx,
        "fallback_used": fallback_used,
        "num_net": args.num_net,
        "depth": args.depth,
        "round_digits": RAW_JSON_ROUND_DIGITS,
    }
    with open(raw_root / "metadata.json", "w") as f:
        json.dump(_serialize_raw_value(metadata), f, indent=2)

    for net_id, info in enumerate(selected_results):
        net_dir = raw_root / f"net_{net_id}"
        net_dir.mkdir(parents=True, exist_ok=True)

        net_data = {
            "net_id": net_id,
            "depth": info["depth"],
            "num_outputs": info["num_outputs"],
            "training_loss": info["training_loss"],
            "training_acc": info["training_acc"],
            "untrained_conf": info["untrained_conf"],
            "pretrained_conf": info["pretrained_conf"],
            "untrained_prob": info["untrained_prob"],
            "pretrained_prob": info["pretrained_prob"],
            "untrained_logit": info["untrained_logit"],
            "pretrained_logit": info["pretrained_logit"],
        }
        with open(net_dir / "data.json", "w") as f:
            json.dump(_serialize_raw_value(net_data), f, indent=2)

    print(f"Raw JSON saved to {raw_root}")


def generate_figures(exp1_results, exp2_results, args, figure_path):
    """Generate all figures from experiment results."""
    os.makedirs(figure_path, exist_ok=True)

    # --- Exp1 per-network distribution plots ---
    if exp1_results is not None:
        # Use output=2 if present (for comparability), otherwise fallback to first output.
        if 2 in args.num_output_list:
            selected_output = 2
        else:
            selected_output = args.num_output_list[0]
        num_output_idx = args.num_output_list.index(selected_output)

        for net_idx in range(args.num_net):
            sample = exp1_results[num_output_idx][net_idx]
            net_suffix = f"_net{net_idx}.svg"

            plot_confidence_distribution(
                sample['untrained_conf'],
                os.path.join(figure_path, f'exp1_untrained_conf{net_suffix}'),
                color=c.color.DARKGRAY, title='Untrained')
            plot_confidence_distribution(
                sample['pretrained_conf'],
                os.path.join(figure_path, f'exp1_pretrained_conf{net_suffix}'),
                color=c.color.SKY, title='Randomly pretrained')

            plot_probability_distribution(
                sample['untrained_prob'],
                os.path.join(figure_path, f'exp1_untrained_prob{net_suffix}'),
                color=c.color.DARKGRAY, title='Untrained')
            plot_probability_distribution(
                sample['pretrained_prob'],
                os.path.join(figure_path, f'exp1_pretrained_prob{net_suffix}'),
                color=c.color.SKY, title='Randomly pretrained')

            plot_logit_distribution(
                sample['untrained_logit'],
                os.path.join(figure_path, f'exp1_untrained_logit{net_suffix}'),
                color=c.color.DARKGRAY, title='Untrained')
            plot_logit_distribution(
                sample['pretrained_logit'],
                os.path.join(figure_path, f'exp1_pretrained_logit{net_suffix}'),
                color=c.color.SKY, title='Randomly pretrained')

            plot_logit_boxplot(
                sample['untrained_logit'],
                os.path.join(figure_path, f'exp1_untrained_logit_box{net_suffix}'),
                color=c.color.DARKGRAY, title='Untrained')
            plot_logit_boxplot(
                sample['pretrained_logit'],
                os.path.join(figure_path, f'exp1_pretrained_logit_box{net_suffix}'),
                color=c.color.SKY, title='Randomly pretrained')

            plot_logit_vs_probability(
                sample['untrained_logit'], sample['untrained_prob'],
                os.path.join(figure_path, f'exp1_untrained_logit_prob{net_suffix}'),
                color=c.color.DARKGRAY, title='Untrained')
            plot_logit_vs_probability(
                sample['pretrained_logit'], sample['pretrained_prob'],
                os.path.join(figure_path, f'exp1_pretrained_logit_prob{net_suffix}'),
                color=c.color.SKY, title='Randomly pretrained')

        # Exp1 aggregate: mean confidence vs output count
        conf_untrained, conf_pretrained = compute_confidence_stats(
            exp1_results, len(args.num_output_list), args.num_net)
        plot_mean_confidence_vs_outputs(
            args.num_output_list, conf_untrained, conf_pretrained,
            os.path.join(figure_path, 'exp1_mean_confidence.svg'))

    # --- Exp2 aggregate plots ---
    if exp2_results is not None:
        conf_untrained, conf_pretrained = compute_confidence_stats(
            exp2_results, len(args.depth_list), args.num_net)
        plot_mean_confidence_vs_depth(
            args.depth_list, conf_untrained, conf_pretrained,
            os.path.join(figure_path, 'exp2_mean_confidence.svg'))

        logit_untrained, logit_pretrained = compute_logit_variance_stats(
            exp2_results, len(args.depth_list), args.num_net)
        plot_logit_variance_vs_depth(
            args.depth_list, logit_untrained, logit_pretrained,
            os.path.join(figure_path, 'exp2_logit_variance.svg'))

    print(f"Figures saved to {figure_path}")


def results_to_serializable(results):
    """Convert experiment results to JSON-serializable format."""
    serializable = []
    for config_results in results:
        config_list = []
        for info in config_results:
            entry = {
                'depth': info['depth'],
                'num_outputs': info['num_outputs'],
                'training_loss': info['training_loss'],
                'training_acc': info['training_acc'],
                'untrained_conf_mean': float(np.mean(info['untrained_conf'])),
                'pretrained_conf_mean': float(np.mean(info['pretrained_conf'])),
                'untrained_logit_var': float(np.var(info['untrained_logit'].flatten())),
                'pretrained_logit_var': float(np.var(info['pretrained_logit'].flatten())),
            }
            config_list.append(entry)
        serializable.append(config_list)
    return serializable


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate experiment ID
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    exp_id = f"{timestamp}_overconf_h{args.num_hidden}_ep{args.epochs}_n{args.num_net}"
    exp_path = os.path.join(args.output_path, exp_id)
    figure_path = os.path.join(exp_path, "figures")
    os.makedirs(exp_path, exist_ok=True)

    # Save config
    config_dict = vars(args)
    config_dict['exp_id'] = exp_id
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Load CIFAR-10 (full training set for measurement)
    transform = get_cifar10_transform()
    cifar10_train = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform)
    cifar10_loader = torch.utils.data.DataLoader(
        cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Run experiments
    exp1_results = None
    exp2_results = None

    if args.experiment in ("1", "both"):
        exp1_results = run_exp1(args, cifar10_loader, device)
        with open(os.path.join(exp_path, 'exp1_results.json'), 'w') as f:
            json.dump({
                'config': {
                    'num_net': args.num_net,
                    'depth': args.depth,
                    'num_output_list': args.num_output_list,
                },
                'results': results_to_serializable(exp1_results),
            }, f, indent=2, cls=NumpyEncoder)
        save_raw_json_exp1_output2(exp1_results, args, exp_path, exp_id)

    if args.experiment in ("2", "both"):
        exp2_results = run_exp2(args, cifar10_loader, device)
        with open(os.path.join(exp_path, 'exp2_results.json'), 'w') as f:
            json.dump({
                'config': {
                    'num_net': args.num_net,
                    'depth_list': args.depth_list,
                    'num_output': args.num_outputs,
                },
                'results': results_to_serializable(exp2_results),
            }, f, indent=2, cls=NumpyEncoder)

    # Generate figures
    if not args.no_figures:
        generate_figures(exp1_results, exp2_results, args, figure_path)

    print(f"\nExperiment complete: {exp_id}")
    print(f"Results saved to: {exp_path}")


if __name__ == "__main__":
    main()
