"""Overconfidence analysis visualization plots (ported from Random2's
src/visualization/overconfidence_plots.py; only the color/size helper
differs, since this repo's custom.figure exposes `mm` + a lowercase
`color` dict instead of Random2's `to_mm`/`GraphConfig`)."""

import numpy as np
import matplotlib.pyplot as plt

from custom.figure import mm


def plot_confidence_distribution(conf: np.ndarray, color: str, title: str = "", chance_level: float = None) -> None:
    """Plot histogram of confidence scores (max softmax prob per sample)."""
    plt.figure(figsize=(15 * mm, 30 * mm))
    plt.hist(conf, bins=100, range=(0, 1),
             density=True, alpha=0.5, color=color, orientation='horizontal')
    if chance_level is not None:
        plt.axhline(y=chance_level, color='k', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.xlabel('Density')
    plt.ylabel('Confidence')
    plt.title(title)
    plt.show()


def plot_logit_distribution(logit: np.ndarray, color: str, title: str = "") -> None:
    """Plot histogram of flattened logit values (N, C) -> flattened."""
    plt.figure(figsize=(30 * mm, 10 * mm))
    plt.hist(logit.flatten(), bins=60, density=True, alpha=0.5, color=color)
    plt.xlim(-10, 10)
    plt.ylim(0, 0.6)
    plt.xlabel('Logit')
    plt.ylabel('Density')
    plt.title(title)
    plt.show()
