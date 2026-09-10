"""Training visualization plots."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Import custom figure module
from custom.figure import to_mm, GraphConfig as c, plot_error


def _build_xticks(total_end: int, warmup_epochs: int, include_boundary: bool) -> List[int]:
    """Build ticks with warmup/main-phase aware positions."""
    ticks = [0, total_end]

    if include_boundary and 0 < warmup_epochs < total_end:
        ticks.append(warmup_epochs)
        main_span = total_end - warmup_epochs
        if main_span > 1:
            ticks.append(warmup_epochs + (main_span // 2))
    elif total_end > 2:
        ticks.append(total_end // 2)

    return sorted(set(int(t) for t in ticks))


def _build_xtick_labels(ticks: List[int], warmup_epochs: int) -> List[str]:
    """Format labels as warmup epoch, then main-training epoch."""
    labels = []
    for tick in ticks:
        if tick == warmup_epochs and warmup_epochs > 0:
            labels.append(f"{warmup_epochs}/0")
        elif tick > warmup_epochs and warmup_epochs > 0:
            labels.append(str(int(tick - warmup_epochs)))
        else:
            labels.append(str(int(tick)))
    return labels


def _stack_series(series_list: List[List[float]]) -> Optional[np.ndarray]:
    """Stack 1D series after trimming to the minimum shared length."""
    if not series_list:
        return None
    arrays = [np.asarray(v, dtype=float) for v in series_list if v is not None and len(v) > 0]
    if not arrays:
        return None
    min_len = min(arr.size for arr in arrays)
    if min_len == 0:
        return None
    return np.stack([arr[:min_len] for arr in arrays], axis=0)


def plot_random_noise_samples(
    samples: np.ndarray,
    save_path: str,
    rows: int = 3,
    cols: int = 10,
) -> str:
    """
    Plot grid of random noise input samples.

    Args:
        samples: Array of shape (N, C, H, W) or (N, H, W, C)
        save_path: Path to save the figure
        rows: Number of rows in grid
        cols: Number of columns in grid

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    axes = axes.flatten()

    for idx in range(min(rows * cols, len(samples))):
        img = samples[idx]
        # Handle both (C, H, W) and (H, W, C) formats
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # Normalize to [0, 1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[idx].imshow(img)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_warmup_curves(
    train_loss: List[float],
    train_acc: List[float],
    save_dir: str,
    file_prefix: str = "",
) -> Dict[str, str]:
    """
    Plot warmup training loss and accuracy curves.

    Args:
        train_loss: List of training losses per epoch
        train_acc: List of training accuracies per epoch
        save_dir: Directory to save figures
        file_prefix: Prefix for figure filenames

    Returns:
        Dict mapping figure names to paths
    """
    paths = {}
    save_dir = Path(save_dir)

    # Loss curve
    plt.figure(figsize=(to_mm(40), to_mm(30)))
    plt.plot(train_loss, color=c.color.SKY)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Random Warmup Loss")
    loss_path = save_dir / f"{file_prefix}warmup_loss.svg"
    plt.savefig(str(loss_path))
    plt.close()
    paths["warmup_loss"] = str(loss_path)

    # Accuracy curve
    plt.figure(figsize=(to_mm(40), to_mm(30)))
    plt.plot(train_acc, color=c.color.SKY)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Random Warmup Accuracy")
    acc_path = save_dir / f"{file_prefix}warmup_acc.svg"
    plt.savefig(str(acc_path))
    plt.close()
    paths["warmup_acc"] = str(acc_path)

    return paths


def plot_training_loss(
    loss_wo: List[float],
    loss_w: List[float],
    save_path: str,
    epochs_warmup: int = 0,
) -> str:
    """
    Plot training loss comparison (with vs without warmup).

    Args:
        loss_wo: Training loss without warmup
        loss_w: Training loss with warmup
        save_path: Path to save figure
        epochs_warmup: Number of warmup epochs (for x-axis offset)

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    # Plot with warmup offset
    x_wo = np.arange(len(loss_wo))
    x_w = np.arange(len(loss_w))

    plt.plot(x_wo, loss_wo, color=c.color.ORANGE, label="w/o warmup")
    plt.plot(x_w, loss_w, color=c.color.SKY, label="w/ warmup")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_training_acc(
    acc_wo: List[float],
    acc_w: List[float],
    save_path: str,
    epochs_warmup: int = 0,
) -> str:
    """
    Plot training accuracy comparison (with vs without warmup).

    Args:
        acc_wo: Training accuracy without warmup
        acc_w: Training accuracy with warmup
        save_path: Path to save figure
        epochs_warmup: Number of warmup epochs (for x-axis offset)

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    x_wo = np.arange(len(acc_wo))
    x_w = np.arange(len(acc_w))

    plt.plot(x_wo, acc_wo, color=c.color.ORANGE, label="w/o warmup")
    plt.plot(x_w, acc_w, color=c.color.SKY, label="w/ warmup")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_validation_loss(
    loss_wo: List[float],
    loss_w: List[float],
    save_path: str,
) -> str:
    """
    Plot validation loss comparison (with vs without warmup).

    Args:
        loss_wo: Validation loss without warmup
        loss_w: Validation loss with warmup
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    plt.plot(loss_wo, color=c.color.ORANGE, label="w/o warmup")
    plt.plot(loss_w, color=c.color.SKY, label="w/ warmup")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_validation_loss_all(
    loss_wo: List[float],
    loss_w: List[float],
    save_path: str,
    warmup_loss: Optional[List[float]] = None,
    warmup_epochs: int = 0,
) -> str:
    """
    Plot validation loss with warmup test-loss window.

    Args:
        loss_wo: Validation loss without warmup (main training phase)
        loss_w: Validation loss with warmup (main training phase)
        save_path: Path to save figure
        warmup_loss: Warmup-phase CIFAR test loss (optional)
        warmup_epochs: Number of warmup epochs for boundary marker

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    loss_wo_arr = np.asarray(loss_wo, dtype=float)
    loss_w_arr = np.asarray(loss_w, dtype=float)
    warmup_loss_arr = np.asarray(warmup_loss, dtype=float) if warmup_loss is not None else np.array([])

    x_offset = int(max(0, warmup_epochs))
    x_wo = x_offset + np.arange(loss_wo_arr.size)
    x_w = x_offset + np.arange(loss_w_arr.size)

    plt.plot(x_wo, loss_wo_arr, color=c.color.ORANGE, label="w/o warmup")
    plt.plot(x_w, loss_w_arr, color=c.color.SKY, label="w/ warmup")

    if warmup_loss_arr.size > 0:
        x_warmup = np.arange(warmup_loss_arr.size)
        plt.plot(x_warmup, warmup_loss_arr, color=c.color.SKY, linewidth=0.8, alpha=0.5)

    total_end = 0
    if x_wo.size > 0:
        total_end = max(total_end, int(x_wo[-1]))
    if x_w.size > 0:
        total_end = max(total_end, int(x_w[-1]))
    if warmup_loss_arr.size > 0:
        total_end = max(total_end, int(warmup_loss_arr.size - 1))
    total_end = max(total_end, 1)

    if warmup_epochs > 0:
        plt.axvline(x=warmup_epochs, color="black", linestyle="--", linewidth=0.5)

    ticks = _build_xticks(total_end, warmup_epochs, include_boundary=warmup_epochs > 0)
    plt.xticks(ticks, _build_xtick_labels(ticks, warmup_epochs))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.xlim(0, total_end)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_validation_loss_all_aggregate(
    loss_wo_list: List[List[float]],
    loss_w_list: List[List[float]],
    save_path: str,
    warmup_loss_list: Optional[List[List[float]]] = None,
    warmup_epochs: int = 0,
) -> str:
    """
    Plot network-averaged validation loss with std shading and warmup window.

    Args:
        loss_wo_list: Validation loss series for w/o warmup networks
        loss_w_list: Validation loss series for w/ warmup networks
        save_path: Path to save figure
        warmup_loss_list: Optional warmup CIFAR test-loss series per network
        warmup_epochs: Number of warmup epochs for boundary marker

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    loss_wo_data = _stack_series(loss_wo_list)
    loss_w_data = _stack_series(loss_w_list)
    warmup_data = _stack_series(warmup_loss_list or [])

    if loss_wo_data is None or loss_w_data is None:
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.savefig(save_path)
        plt.close()
        return save_path

    series_len = min(loss_wo_data.shape[1], loss_w_data.shape[1])
    loss_wo_data = loss_wo_data[:, :series_len]
    loss_w_data = loss_w_data[:, :series_len]

    x_offset = int(max(0, warmup_epochs))
    x_wo = x_offset + np.arange(series_len)
    x_w = x_offset + np.arange(series_len)

    plot_error(loss_wo_data, c.color.ORANGE, label="w/o warmup", xrange=x_wo)
    plot_error(loss_w_data, c.color.SKY, label="w/ warmup", xrange=x_w)

    if warmup_data is not None:
        x_warmup = np.arange(warmup_data.shape[1])
        plot_error(warmup_data, c.color.SKY, xrange=x_warmup)

    total_end = int(max(x_wo[-1] if x_wo.size else 0, x_w[-1] if x_w.size else 0))
    if warmup_data is not None and warmup_data.shape[1] > 0:
        total_end = max(total_end, int(warmup_data.shape[1] - 1))
    total_end = max(total_end, 1)

    if warmup_epochs > 0:
        plt.axvline(x=warmup_epochs, color="black", linestyle="--", linewidth=0.5)

    ticks = _build_xticks(total_end, warmup_epochs, include_boundary=warmup_epochs > 0)
    plt.xticks(ticks, _build_xtick_labels(ticks, warmup_epochs))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.xlim(0, total_end)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_validation_acc(
    acc_wo: List[float],
    acc_w: List[float],
    save_path: str,
) -> str:
    """
    Plot validation accuracy comparison (with vs without warmup).

    Args:
        acc_wo: Validation accuracy without warmup
        acc_w: Validation accuracy with warmup
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    plt.plot(acc_wo, color=c.color.ORANGE, label="w/o warmup")
    plt.plot(acc_w, color=c.color.SKY, label="w/ warmup")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_training_curves_multi(
    results_wo: List[Dict[str, List[float]]],
    results_w: List[Dict[str, List[float]]],
    save_dir: str,
    file_prefix: str = "",
) -> Dict[str, str]:
    """
    Plot training curves with error bands across multiple networks.

    Args:
        results_wo: List of result dicts without warmup
        results_w: List of result dicts with warmup
        save_dir: Directory to save figures
        file_prefix: Prefix for figure filenames

    Returns:
        Dict mapping figure names to paths
    """
    paths = {}
    save_dir = Path(save_dir)

    # Collect metrics
    metrics = ["train_loss", "train_acc", "test_loss", "test_acc"]
    titles = ["Training Loss", "Training Accuracy", "Test Loss", "Test Accuracy"]
    ylabels = ["Loss", "Accuracy", "Loss", "Accuracy"]
    filenames = ["training_loss", "training_acc", "validation_loss", "validation_acc"]

    for metric, title, ylabel, fname in zip(metrics, titles, ylabels, filenames):
        data_wo = np.array([r[metric] for r in results_wo])
        data_w = np.array([r[metric] for r in results_w])

        plt.figure(figsize=(to_mm(40), to_mm(30)))

        if len(data_wo) > 1:
            plot_error(data_wo, c.color.ORANGE, label="w/o warmup")
            plot_error(data_w, c.color.SKY, label="w/ warmup")
        else:
            plt.plot(data_wo[0], color=c.color.ORANGE, label="w/o warmup")
            plt.plot(data_w[0], color=c.color.SKY, label="w/ warmup")

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)

        save_path = save_dir / f"{file_prefix}{fname}.svg"
        plt.savefig(str(save_path))
        plt.close()

        paths[fname] = str(save_path)

    return paths
