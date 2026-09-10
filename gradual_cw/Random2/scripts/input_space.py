#!/usr/bin/env python3
"""Input space (2D) uncertainty experiment runner.

Compares untrained vs. randomly pretrained networks on a 2D input space,
visualizing confidence maps and prediction balance.
"""

import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.models.mlp import LinearNetwork
from src.training.random_training import random_train
from src.evaluation.inference import get_predictions_and_confidence
from src.logger import NumpyEncoder
from src.visualization.input_space_plots import (
    plot_confidence_map,
    plot_confidence_boxplot,
    plot_prediction_std_boxplot,
    plot_warmup_loss,
)

RAW_JSON_ROUND_DIGITS = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Input space uncertainty experiment")
    parser.add_argument("--model_structure", type=str, default="2,10,10,2",
                        help="Comma-separated network structure (e.g. 2,10,10,2)")
    parser.add_argument("--num_net", type=int, default=10,
                        help="Number of networks")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Random pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_noise", type=int, default=1000,
                        help="Random samples per epoch")
    parser.add_argument("--num_test", type=int, default=1000,
                        help="Test points for confidence map")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--input_range", type=float, default=10,
                        help="Input space limit (symmetric: [-range, range])")
    parser.add_argument("--output_path", type=str, default="./exp")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation")

    args = parser.parse_args()
    args.model_structure = [int(x) for x in args.model_structure.split(",")]
    return args


def create_model(model_structure, device, w_seed=-1):
    """Create LinearNetwork from model structure list."""
    in_features = model_structure[0]
    num_hidden_list = model_structure[1:]
    num_layers = len(num_hidden_list)
    model = LinearNetwork(
        in_features=in_features,
        num_layers=num_layers,
        num_hidden_list=num_hidden_list,
        mode='BP',
        w_seed=w_seed,
        b_seed=1,
    ).to(device)
    return model


def generate_test_data(num_test, input_range, device):
    """Generate uniform random 2D test points with dummy labels."""
    x = np.random.uniform(-input_range, input_range, (num_test, 2))
    y = np.random.randint(0, 2, num_test)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    return x, loader


def evaluate_model(model, loader, device):
    """Evaluate model and return predictions and confidence."""
    _, conf, pred_labels, _ = get_predictions_and_confidence(model, loader, device)
    return pred_labels, conf


def pretrain_on_noise(model, args, device):
    """Pretrain model on random uniform noise."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    output_size = args.model_structure[-1]
    uniform_std = args.input_range / np.sqrt(3)

    losses = []
    for epoch in range(args.epochs):
        loss, acc = random_train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            input_shape=(args.model_structure[0],),
            num_image=args.num_noise,
            batch_size=args.batch_size,
            output_size=output_size,
            mean=0,
            std=uniform_std,
            mode="uniform",
            device=device,
        )
        losses.append(loss)
    return losses


def compute_prediction_std(pred_labels, num_classes=2):
    """Compute std of class prediction proportions."""
    proportions = []
    for c in range(num_classes):
        proportions.append(np.sum(pred_labels == c) / len(pred_labels))
    return float(np.std(proportions))


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


def save_raw_json(results, args, exp_path, exp_id):
    """Save figure-generation raw data as per-network JSON files."""
    raw_root = Path(exp_path) / "raw_json" / "input_space"
    raw_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "exp_id": exp_id,
        "num_net": args.num_net,
        "model_structure": args.model_structure,
        "num_test": args.num_test,
        "input_range": args.input_range,
        "round_digits": RAW_JSON_ROUND_DIGITS,
    }
    with open(raw_root / "metadata.json", "w") as f:
        json.dump(_serialize_raw_value(metadata), f, indent=2)

    for net_id, net_result in enumerate(results):
        net_dir = raw_root / f"net_{net_id}"
        net_dir.mkdir(parents=True, exist_ok=True)
        test_x = net_result["test_x"]

        net_data = {
            "net_id": net_id,
            "test_xy": {
                "x": test_x[:, 0],
                "y": test_x[:, 1],
            },
            "warmup_losses": net_result["warmup_losses"],
            "untrained_conf": net_result["untrained_conf"],
            "pretrained_conf": net_result["pretrained_conf"],
            "untrained_pred": net_result["untrained_pred"],
            "pretrained_pred": net_result["pretrained_pred"],
            "untrained_pred_std": net_result["untrained_pred_std"],
            "pretrained_pred_std": net_result["pretrained_pred_std"],
        }
        with open(net_dir / "data.json", "w") as f:
            json.dump(_serialize_raw_value(net_data), f, indent=2)

    print(f"Raw JSON saved to {raw_root}")


def generate_figures(results, args, figure_path):
    """Generate all figures from experiment results."""
    os.makedirs(figure_path, exist_ok=True)
    xlim = (-args.input_range, args.input_range)
    ylim = (-args.input_range, args.input_range)

    for i, net_result in enumerate(results):
        # Confidence maps
        plot_confidence_map(
            net_result['test_x'], net_result['untrained_conf'],
            os.path.join(figure_path, f'confidence_map_untrained_{i}.svg'),
            title='Untrained', xlim=xlim, ylim=ylim)
        plot_confidence_map(
            net_result['test_x'], net_result['pretrained_conf'],
            os.path.join(figure_path, f'confidence_map_pretrained_{i}.svg'),
            title='Pretrained', xlim=xlim, ylim=ylim)

        # Per-network confidence boxplot
        plot_confidence_boxplot(
            net_result['untrained_conf'], net_result['pretrained_conf'],
            os.path.join(figure_path, f'confidence_boxplot_{i}.svg'))

        # Warmup loss
        plot_warmup_loss(
            net_result['warmup_losses'],
            os.path.join(figure_path, f'warmup_loss_{i}.svg'))

    # Aggregate summary plots
    all_conf_untrained = np.concatenate([r['untrained_conf'] for r in results])
    all_conf_pretrained = np.concatenate([r['pretrained_conf'] for r in results])
    plot_confidence_boxplot(
        all_conf_untrained, all_conf_pretrained,
        os.path.join(figure_path, 'summary_confidence.svg'))

    std_untrained = [r['untrained_pred_std'] for r in results]
    std_pretrained = [r['pretrained_pred_std'] for r in results]
    plot_prediction_std_boxplot(
        std_untrained, std_pretrained,
        os.path.join(figure_path, 'summary_prediction_std.svg'))

    print(f"Figures saved to {figure_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate experiment ID
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    structure_str = "x".join(str(s) for s in args.model_structure)
    exp_id = f"{timestamp}_input_space_{structure_str}_ep{args.epochs}_n{args.num_net}"
    exp_path = os.path.join(args.output_path, exp_id)
    figure_path = os.path.join(exp_path, "figures")
    os.makedirs(exp_path, exist_ok=True)

    # Save config
    config_dict = vars(args)
    config_dict['exp_id'] = exp_id
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Generate test data
    test_x, test_loader = generate_test_data(args.num_test, args.input_range, device)

    # Run experiments
    results = []
    for net_idx in range(args.num_net):
        print(f"\n--- Network {net_idx+1}/{args.num_net} ---")

        # Create untrained model
        model_untrained = create_model(args.model_structure, device, w_seed=net_idx)

        # Create pretrained model (same init)
        model_pretrained = create_model(args.model_structure, device, w_seed=net_idx)

        # Pretrain on random noise
        print("  Pretraining on random noise...")
        warmup_losses = pretrain_on_noise(model_pretrained, args, device)
        print(f"  Final warmup loss: {warmup_losses[-1]:.4f}")

        # Evaluate both models
        print("  Evaluating...")
        untrained_pred, untrained_conf = evaluate_model(model_untrained, test_loader, device)
        pretrained_pred, pretrained_conf = evaluate_model(model_pretrained, test_loader, device)

        # Compute prediction balance
        untrained_std = compute_prediction_std(untrained_pred, args.model_structure[-1])
        pretrained_std = compute_prediction_std(pretrained_pred, args.model_structure[-1])
        print(f"  Untrained pred std: {untrained_std:.4f}, Pretrained pred std: {pretrained_std:.4f}")
        print(f"  Untrained mean conf: {np.mean(untrained_conf):.4f}, "
              f"Pretrained mean conf: {np.mean(pretrained_conf):.4f}")

        results.append({
            'test_x': test_x,
            'warmup_losses': warmup_losses,
            'untrained_conf': untrained_conf,
            'pretrained_conf': pretrained_conf,
            'untrained_pred': untrained_pred,
            'pretrained_pred': pretrained_pred,
            'untrained_pred_std': untrained_std,
            'pretrained_pred_std': pretrained_std,
        })

    # Save results (summary only, not raw arrays)
    summary = []
    for i, r in enumerate(results):
        summary.append({
            'net_id': i,
            'warmup_final_loss': r['warmup_losses'][-1],
            'untrained_conf_mean': float(np.mean(r['untrained_conf'])),
            'pretrained_conf_mean': float(np.mean(r['pretrained_conf'])),
            'untrained_pred_std': r['untrained_pred_std'],
            'pretrained_pred_std': r['pretrained_pred_std'],
        })
    with open(os.path.join(exp_path, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Save figure-generation raw data
    save_raw_json(results, args, exp_path, exp_id)

    # Generate figures
    if not args.no_figures:
        generate_figures(results, args, figure_path)

    print(f"\nExperiment complete: {exp_id}")
    print(f"Results saved to: {exp_path}")


if __name__ == "__main__":
    main()
