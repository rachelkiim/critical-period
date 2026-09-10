"""Input space (2D) uncertainty visualization plots."""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from custom.figure import to_mm, GraphConfig as c


def plot_confidence_map(
    x: np.ndarray,
    conf: np.ndarray,
    save_path: str,
    title: str = "",
    xlim: Tuple[float, float] = (-10, 10),
    ylim: Tuple[float, float] = (-10, 10),
) -> str:
    """
    Plot 2D scatter colored by confidence.

    Args:
        x: 2D input points (N, 2)
        conf: Confidence scores (N,)
        save_path: Path to save figure
        title: Plot title
        xlim: X-axis limits
        ylim: Y-axis limits

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.scatter(x[:, 0], x[:, 1], c=conf, s=1, cmap='coolwarm')
    plt.clim(0.5, 1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_confidence_boxplot(
    conf_untrained: np.ndarray,
    conf_pretrained: np.ndarray,
    save_path: str,
) -> str:
    """
    Plot boxplot comparing untrained vs pretrained confidence.

    Args:
        conf_untrained: Confidence scores for untrained model (N,)
        conf_pretrained: Confidence scores for pretrained model (N,)
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(20), to_mm(30)))
    plt.boxplot([conf_untrained, conf_pretrained], positions=[0, 1], widths=0.6)
    plt.axhline(0.5, color='k', linestyle='--', linewidth=0.5)
    plt.xlim(-1, 2)
    plt.ylim(0.4, 1)
    plt.xticks([0, 1], ['Untrained', 'Trained'])
    plt.ylabel('Confidence')
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_prediction_std_boxplot(
    std_untrained: List[float],
    std_pretrained: List[float],
    save_path: str,
) -> str:
    """
    Plot boxplot comparing prediction balance std across networks.

    Args:
        std_untrained: Prediction balance std per untrained network
        std_pretrained: Prediction balance std per pretrained network
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(15), to_mm(30)))
    plt.boxplot([std_untrained, std_pretrained], positions=[0, 1], widths=0.6)
    plt.xlim(-1, 2)
    plt.ylim(0, 0.6)
    plt.xticks([0, 1], ['Untrained', 'Trained'])
    plt.ylabel('Prediction std')
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_warmup_loss(
    train_loss: List[float],
    save_path: str,
) -> str:
    """
    Plot pretraining loss curve.

    Args:
        train_loss: Per-epoch training losses
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.plot(train_loss, color=c.color.SKY)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Warmup loss')
    plt.savefig(save_path)
    plt.close()
    return save_path
