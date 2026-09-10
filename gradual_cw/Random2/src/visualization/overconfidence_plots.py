"""Overconfidence analysis visualization plots."""

from typing import List
import numpy as np
import matplotlib.pyplot as plt

from custom.figure import to_mm, GraphConfig as c


def plot_confidence_distribution(
    conf: np.ndarray,
    save_path: str,
    color: str,
    title: str = "",
) -> str:
    """
    Plot horizontal histogram of confidence scores.

    Args:
        conf: Confidence scores array (N,)
        save_path: Path to save figure
        color: Histogram color
        title: Plot title

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(15), to_mm(30)))
    plt.hist(conf, bins=30, range=(0.5, 1),
             density=True, alpha=0.5, color=color, orientation='horizontal')
    plt.ylim(0.5, 1)
    plt.yticks([0.5, 0.75, 1])
    plt.xlabel('Density')
    plt.ylabel('Confidence')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_probability_distribution(
    prob: np.ndarray,
    save_path: str,
    color: str,
    title: str = "",
) -> str:
    """
    Plot horizontal histogram of flattened probability values.

    Args:
        prob: Probability matrix (N, C) — will be flattened
        save_path: Path to save figure
        color: Histogram color
        title: Plot title

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(10), to_mm(30)))
    plt.hist(prob.flatten(), bins=60, range=(0, 1),
             density=True, alpha=0.5, color=color, orientation='horizontal')
    plt.axhline(y=0.5, color='k', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1)
    plt.xlabel('Density')
    plt.ylabel('Probability')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_logit_distribution(
    logit: np.ndarray,
    save_path: str,
    color: str,
    title: str = "",
) -> str:
    """
    Plot histogram of flattened logit values.

    Args:
        logit: Logit matrix (N, C) — will be flattened
        save_path: Path to save figure
        color: Histogram color
        title: Plot title

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(10)))
    plt.hist(logit.flatten(), bins=60, density=True, alpha=0.5, color=color)
    plt.xlim(-10, 10)
    plt.ylim(0, 0.6)
    plt.xlabel('Logit')
    plt.ylabel('Density')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_logit_boxplot(
    logit: np.ndarray,
    save_path: str,
    color: str,
    title: str = "",
) -> str:
    """
    Plot horizontal boxplot of flattened logit values.

    Args:
        logit: Logit matrix (N, C) — will be flattened
        save_path: Path to save figure
        color: Box color
        title: Plot title

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(10)))
    plt.boxplot(logit.flatten(), widths=0.5, positions=[0], patch_artist=True,
                boxprops=dict(facecolor=color), orientation='horizontal')
    plt.ylim(-1, 1)
    plt.xlim(-10, 10)
    plt.yticks([])
    plt.xlabel('Logit')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_logit_vs_probability(
    logit: np.ndarray,
    prob: np.ndarray,
    save_path: str,
    color: str,
    title: str = "",
    num_points: int = 2000,
) -> str:
    """
    Plot scatter of logit vs probability with sigmoid overlay.

    Args:
        logit: Logit matrix (N, C) — will be flattened
        prob: Probability matrix (N, C) — will be flattened
        save_path: Path to save figure
        color: Scatter point color
        title: Plot title
        num_points: Number of points to display

    Returns:
        Path to saved figure
    """
    x = np.linspace(-10, 10, 100)
    softmax_max = 1 / (1 + np.exp(-x))

    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.scatter(logit.flatten()[:num_points],
                prob.flatten()[:num_points],
                s=1, alpha=0.2, c=color)
    plt.plot(x, softmax_max, color='k', linewidth=0.5, linestyle='--')
    plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.xlabel('Logit')
    plt.ylabel('Probability')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_mean_confidence_vs_outputs(
    num_output_list: List[int],
    conf_untrained: np.ndarray,
    conf_pretrained: np.ndarray,
    save_path: str,
) -> str:
    """
    Plot mean confidence vs number of output classes with 1/k theoretical curve.

    Args:
        num_output_list: List of output class counts
        conf_untrained: (num_configs, num_nets) mean confidence per untrained network
        conf_pretrained: (num_configs, num_nets) mean confidence per trained network
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    mean_untrained = np.mean(conf_untrained, axis=1)
    std_untrained = np.std(conf_untrained, axis=1)
    mean_pretrained = np.mean(conf_pretrained, axis=1)
    std_pretrained = np.std(conf_pretrained, axis=1)

    x_data = np.arange(num_output_list[0] - 1, num_output_list[-1] + 1, 0.1)
    y_data = 1 / x_data

    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.errorbar(num_output_list, mean_untrained, yerr=std_untrained,
                 fmt='o', color=c.color.DARKGRAY, label='Untrained',
                 capsize=0, elinewidth=1, markeredgewidth=1, markersize=3)
    plt.errorbar(num_output_list, mean_pretrained, yerr=std_pretrained,
                 fmt='o', color=c.color.SKY, label='Randomly pretrained',
                 capsize=0, elinewidth=1, markeredgewidth=1, markersize=3)
    plt.plot(x_data, y_data, color='k', linewidth=0.5, linestyle='--', label='1/k')
    plt.xticks(num_output_list)
    plt.xlabel('Number of outputs')
    plt.ylabel('Mean confidence')
    plt.xlim(min(num_output_list) - 1, max(num_output_list) + 1)
    plt.ylim(0, 1)
    plt.title('Mean confidence vs \n Number of outputs')
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_mean_confidence_vs_depth(
    depth_list: List[int],
    conf_untrained: np.ndarray,
    conf_pretrained: np.ndarray,
    save_path: str,
) -> str:
    """
    Plot mean confidence vs network depth.

    Args:
        depth_list: List of depth values
        conf_untrained: (num_configs, num_nets) mean confidence per untrained network
        conf_pretrained: (num_configs, num_nets) mean confidence per trained network
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    mean_untrained = np.mean(conf_untrained, axis=1)
    std_untrained = np.std(conf_untrained, axis=1)
    mean_pretrained = np.mean(conf_pretrained, axis=1)
    std_pretrained = np.std(conf_pretrained, axis=1)

    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.errorbar(depth_list, mean_untrained, yerr=std_untrained,
                 fmt='o', color=c.color.DARKGRAY, label='Untrained',
                 capsize=0, elinewidth=1, markeredgewidth=1, markersize=3)
    plt.errorbar(depth_list, mean_pretrained, yerr=std_pretrained,
                 fmt='o', color=c.color.SKY, label='Randomly pretrained',
                 capsize=0, elinewidth=1, markeredgewidth=1, markersize=3)
    plt.xticks(depth_list)
    plt.xlabel('Depth')
    plt.ylabel('Mean confidence')
    plt.xlim(depth_list[0] - 0.5, depth_list[-1] + 0.5)
    plt.title('Mean confidence vs \n Depth')
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_logit_variance_vs_depth(
    depth_list: List[int],
    logit_untrained: np.ndarray,
    logit_pretrained: np.ndarray,
    save_path: str,
) -> str:
    """
    Plot logit variance vs network depth.

    Args:
        depth_list: List of depth values
        logit_untrained: (num_configs, num_nets) logit variance per untrained network
        logit_pretrained: (num_configs, num_nets) logit variance per trained network
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    mean_untrained = np.mean(logit_untrained, axis=1)
    std_untrained = np.std(logit_untrained, axis=1)
    mean_pretrained = np.mean(logit_pretrained, axis=1)
    std_pretrained = np.std(logit_pretrained, axis=1)

    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.errorbar(depth_list, mean_untrained, yerr=std_untrained,
                 fmt='o', color=c.color.DARKGRAY, label='Untrained',
                 capsize=0, elinewidth=1, markeredgewidth=1, markersize=3)
    plt.errorbar(depth_list, mean_pretrained, yerr=std_pretrained,
                 fmt='o', color=c.color.SKY, label='Randomly pretrained',
                 capsize=0, elinewidth=1, markeredgewidth=1, markersize=3)
    plt.xticks(depth_list)
    plt.xlabel('Depth')
    plt.ylabel('Logit variance')
    plt.xlim(depth_list[0] - 0.5, depth_list[-1] + 0.5)
    plt.title('Logit variance vs \n Depth')
    plt.savefig(save_path)
    plt.close()
    return save_path
