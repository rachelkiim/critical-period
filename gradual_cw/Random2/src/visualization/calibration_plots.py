"""Calibration visualization plots."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from custom.figure import to_mm, GraphConfig as c, plot_error, bar_error


def ceil_0p1(value: float) -> float:
    """Round up to the nearest 0.1 with a minimum of 0.1."""
    return max(0.1, float(np.ceil(float(value) * 10.0) / 10.0))


def conf_acc_limit(*series: Optional[List[float]]) -> float:
    """Compute shared axis upper bound for confidence/accuracy plots."""
    max_value = 0.0
    for values in series:
        if values is None:
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        max_value = max(max_value, float(np.max(arr)))
    return ceil_0p1(max_value)


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
    """Stack equally-sized 1D series after trimming to the minimum length."""
    if not series_list:
        return None
    arrays = [np.asarray(v, dtype=float) for v in series_list if v is not None and len(v) > 0]
    if not arrays:
        return None
    min_len = min(arr.size for arr in arrays)
    if min_len == 0:
        return None
    trimmed = [arr[:min_len] for arr in arrays]
    return np.stack(trimmed, axis=0)


def plot_ece_curves(
    ece_wo: List[float],
    ece_w: List[float],
    save_path: str,
) -> str:
    """
    Plot ECE comparison over epochs.

    Args:
        ece_wo: ECE values without warmup per epoch
        ece_w: ECE values with warmup per epoch
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    plt.plot(ece_wo, label="w/o warmup", color=c.color.ORANGE)
    plt.plot(ece_w, label="w/ warmup", color=c.color.SKY)

    plt.xlabel("Epoch")
    plt.ylabel("ECE")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_conf_acc_line(
    conf: List[float],
    acc: List[float],
    save_path: str,
    color: str = None,
    title: str = "",
    warmup_conf: Optional[List[float]] = None,
    warmup_acc: Optional[List[float]] = None,
    warmup_epochs: int = 0,
    reserve_warmup_window: bool = False,
) -> str:
    """
    Plot confidence vs accuracy as line plot over epochs.

    Args:
        conf: Confidence values per epoch
        acc: Accuracy values per epoch
        save_path: Path to save figure
        color: Color for the confidence line
        title: Plot title
        warmup_conf: Optional warmup confidence values (test CIFAR)
        warmup_acc: Optional warmup accuracy values (test CIFAR)
        warmup_epochs: Number of warmup epochs for boundary marker
        reserve_warmup_window: Reserve warmup window on x-axis without plotting
            warmup curves (used for w/o warmup condition)

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))

    if color is None:
        color = c.color.ORANGE

    warmup_conf_arr = np.asarray(warmup_conf, dtype=float) if warmup_conf is not None else np.array([])
    warmup_acc_arr = np.asarray(warmup_acc, dtype=float) if warmup_acc is not None else np.array([])
    include_warmup = warmup_conf_arr.size > 0 and warmup_acc_arr.size > 0

    conf_main = np.asarray(conf, dtype=float)
    acc_main = np.asarray(acc, dtype=float)

    if reserve_warmup_window or warmup_epochs > 0:
        x_offset = int(max(0, warmup_epochs))
    else:
        x_offset = 0

    x_main = x_offset + np.arange(conf_main.size)
    plt.plot(x_main, conf_main, color=color, label="Confidence")
    plt.plot(x_main, acc_main, color=c.color.DARKGRAY, label="Accuracy")

    if include_warmup:
        x_warmup = np.arange(warmup_conf_arr.size)
        plt.plot(x_warmup, warmup_conf_arr, color=color, linewidth=0.8, alpha=0.5)
        plt.plot(x_warmup, warmup_acc_arr, color=c.color.DARKGRAY, linewidth=0.8, alpha=0.5)

    total_end = int(max(x_main[-1] if x_main.size else 0, warmup_epochs + conf_main.size - 1))
    if include_warmup:
        total_end = max(total_end, int(warmup_conf_arr.size - 1))
    total_end = max(total_end, 1)

    if warmup_epochs > 0:
        plt.axvline(x=warmup_epochs, color="black", linestyle="--", linewidth=0.5)

    ticks = _build_xticks(total_end, warmup_epochs, include_boundary=warmup_epochs > 0)
    plt.xticks(ticks, _build_xtick_labels(ticks, warmup_epochs))

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.xlim(0, total_end)
    ylim_max = conf_acc_limit(conf, acc, warmup_conf if include_warmup else None, warmup_acc if include_warmup else None)
    plt.ylim(0, ylim_max)
    if title:
        plt.title(title)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_conf_acc_line_aggregate(
    conf_list: List[List[float]],
    acc_list: List[List[float]],
    save_path: str,
    color: str = None,
    title: str = "",
    warmup_conf_list: Optional[List[List[float]]] = None,
    warmup_acc_list: Optional[List[List[float]]] = None,
    warmup_epochs: int = 0,
    reserve_warmup_window: bool = False,
) -> str:
    """
    Plot network-averaged confidence/accuracy with std shaded area.

    Args:
        conf_list: List of confidence series for each network
        acc_list: List of accuracy series for each network
        save_path: Path to save figure
        color: Color for confidence mean/std
        title: Plot title
        warmup_conf_list: Optional warmup confidence series for each network
        warmup_acc_list: Optional warmup accuracy series for each network
        warmup_epochs: Number of warmup epochs for boundary marker
        reserve_warmup_window: Reserve warmup window on x-axis without plotting
            warmup curves (used for w/o warmup condition)

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))

    if color is None:
        color = c.color.ORANGE

    conf_data = _stack_series(conf_list)
    acc_data = _stack_series(acc_list)
    if conf_data is None or acc_data is None:
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        if title:
            plt.title(title)
        plt.savefig(save_path)
        plt.close()
        return save_path

    series_len = min(conf_data.shape[1], acc_data.shape[1])
    conf_data = conf_data[:, :series_len]
    acc_data = acc_data[:, :series_len]

    warmup_conf_data = _stack_series(warmup_conf_list or [])
    warmup_acc_data = _stack_series(warmup_acc_list or [])
    include_warmup = warmup_conf_data is not None and warmup_acc_data is not None
    if include_warmup:
        warmup_len = min(warmup_conf_data.shape[1], warmup_acc_data.shape[1])
        warmup_conf_data = warmup_conf_data[:, :warmup_len]
        warmup_acc_data = warmup_acc_data[:, :warmup_len]

    if reserve_warmup_window or warmup_epochs > 0:
        x_offset = int(max(0, warmup_epochs))
    else:
        x_offset = 0

    x_main = x_offset + np.arange(series_len)
    plot_error(conf_data, color, label="Confidence", xrange=x_main)
    plot_error(acc_data, c.color.DARKGRAY, label="Accuracy", xrange=x_main)

    if include_warmup:
        x_warmup = np.arange(warmup_conf_data.shape[1])
        plot_error(warmup_conf_data, color, xrange=x_warmup)
        plot_error(warmup_acc_data, c.color.DARKGRAY, xrange=x_warmup)

    total_end = int(max(x_main[-1] if x_main.size else 0, warmup_epochs + series_len - 1))
    if include_warmup:
        total_end = max(total_end, int(warmup_conf_data.shape[1] - 1))
    total_end = max(total_end, 1)

    if warmup_epochs > 0:
        plt.axvline(x=warmup_epochs, color="black", linestyle="--", linewidth=0.5)

    ticks = _build_xticks(total_end, warmup_epochs, include_boundary=warmup_epochs > 0)
    plt.xticks(ticks, _build_xtick_labels(ticks, warmup_epochs))

    flat_conf = conf_data.reshape(-1)
    flat_acc = acc_data.reshape(-1)
    if include_warmup:
        flat_warmup_conf = warmup_conf_data.reshape(-1)
        flat_warmup_acc = warmup_acc_data.reshape(-1)
    else:
        flat_warmup_conf = None
        flat_warmup_acc = None

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.xlim(0, total_end)
    ylim_max = conf_acc_limit(flat_conf, flat_acc, flat_warmup_conf, flat_warmup_acc)
    plt.ylim(0, ylim_max)
    if title:
        plt.title(title)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_conf_acc_scatter(
    conf_wo: List[float],
    acc_wo: List[float],
    conf_w: List[float],
    acc_w: List[float],
    save_path: str,
    include_warmup_phase: bool = False,
    warmup_conf: Optional[List[float]] = None,
    warmup_acc: Optional[List[float]] = None,
) -> str:
    """
    Plot confidence vs accuracy as scatter plot.

    Args:
        conf_wo: Confidence values without warmup
        acc_wo: Accuracy values without warmup
        conf_w: Confidence values with warmup
        acc_w: Accuracy values with warmup
        save_path: Path to save figure
        include_warmup_phase: Whether to include warmup phase data
        warmup_conf: Confidence during warmup phase
        warmup_acc: Accuracy during warmup phase

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))

    conf_wo_arr = np.asarray(conf_wo, dtype=float)
    acc_wo_arr = np.asarray(acc_wo, dtype=float)
    conf_w_arr = np.asarray(conf_w, dtype=float)
    acc_w_arr = np.asarray(acc_w, dtype=float)

    plt.scatter(conf_wo_arr, acc_wo_arr, label="w/o warmup", color=c.color.ORANGE, s=1, alpha=1)
    plt.scatter(conf_w_arr, acc_w_arr, label="w/ warmup", color=c.color.SKY, s=1, alpha=1)

    if include_warmup_phase and warmup_conf is not None and warmup_acc is not None:
        warmup_conf_arr = np.asarray(warmup_conf, dtype=float)
        warmup_acc_arr = np.asarray(warmup_acc, dtype=float)
        plt.scatter(warmup_conf_arr, warmup_acc_arr, label="warmup phase",
                    color=c.color.SKY, s=1, alpha=0.5)
    else:
        warmup_conf_arr = None
        warmup_acc_arr = None

    # Diagonal line (perfect calibration)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.5)
    # Chance-level accuracy
    plt.axhline(y=0.1, color="black", linestyle="--", linewidth=0.5, alpha=0.7)

    lim = conf_acc_limit(
        conf_wo_arr,
        acc_wo_arr,
        conf_w_arr,
        acc_w_arr,
        warmup_conf_arr,
        warmup_acc_arr,
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_loss_acc_trajectory(
    loss_wo: List[float],
    acc_wo: List[float],
    loss_w: List[float],
    acc_w: List[float],
    save_path: str,
    include_warmup: bool = False,
    warmup_loss: Optional[List[float]] = None,
    warmup_acc: Optional[List[float]] = None,
    chance_acc: Optional[float] = 0.25,
    chance_acc_secondary: Optional[float] = None,
    chance_linewidth: float = 0.5,
    add_markers: bool = True,
) -> str:
    """
    Plot loss vs accuracy trajectory.

    Args:
        loss_wo: Loss values without warmup
        acc_wo: Accuracy values without warmup
        loss_w: Loss values with warmup
        acc_w: Accuracy values with warmup
        save_path: Path to save figure
        include_warmup: Whether to include warmup phase
        warmup_loss: Loss during warmup (test CIFAR)
        warmup_acc: Accuracy during warmup (test CIFAR)
        chance_acc: Optional horizontal dashed line y-value
        chance_acc_secondary: Optional second horizontal dashed line y-value
        chance_linewidth: Line width for dashed reference line
        add_markers: Add square marker near acc=0.25 and circle at final epoch

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))

    loss_wo_arr = np.asarray(loss_wo, dtype=float)
    acc_wo_arr = np.asarray(acc_wo, dtype=float)
    loss_w_arr = np.asarray(loss_w, dtype=float)
    acc_w_arr = np.asarray(acc_w, dtype=float)

    plt.scatter(loss_wo_arr, acc_wo_arr, label="w/o warmup", color=c.color.ORANGE, s=1)
    plt.scatter(loss_w_arr, acc_w_arr, label="w/ warmup", color=c.color.SKY, s=1)

    if include_warmup and warmup_loss is not None and warmup_acc is not None:
        plt.scatter(warmup_loss, warmup_acc, label="warmup phase",
                    color=c.color.SKY, s=1, alpha=0.5)

    if chance_acc is not None:
        plt.axhline(y=chance_acc, color="black", linestyle="--", linewidth=chance_linewidth, alpha=0.7)
    if chance_acc_secondary is not None:
        plt.axhline(y=chance_acc_secondary, color="black", linestyle="--", linewidth=chance_linewidth, alpha=0.7)

    if add_markers:
        def _add_markers(losses: np.ndarray, accs: np.ndarray, marker_color: str):
            if losses.size == 0 or accs.size == 0:
                return
            target_acc = 0.25 if chance_acc is None else float(chance_acc)
            idx_acc025 = int(np.argmin(np.abs(accs - target_acc)))
            idx_final = int(accs.size - 1)

            plt.scatter(
                losses[idx_acc025],
                accs[idx_acc025],
                marker="s",
                s=10,
                color="w",
                edgecolors=marker_color,
                zorder=4,
            )
            plt.scatter(
                losses[idx_final],
                accs[idx_final],
                marker="o",
                s=10,
                color="w",
                edgecolors=marker_color,
                zorder=4,
            )

        _add_markers(loss_wo_arr, acc_wo_arr, c.color.ORANGE)
        _add_markers(loss_w_arr, acc_w_arr, c.color.SKY)

    plt.xlabel("Loss")
    plt.ylabel("Accuracy")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_reliability_diagram(
    acc_bins: List[float],
    conf_bins: List[float],
    num_samples: List[int],
    save_path: str,
    title: str = "",
    color: str = None,
) -> str:
    """
    Plot reliability diagram (calibration curve).

    Args:
        acc_bins: Accuracy per bin
        conf_bins: Confidence per bin (bin centers)
        num_samples: Number of samples per bin
        save_path: Path to save figure
        title: Plot title
        color: Bar color

    Returns:
        Path to saved figure
    """
    if color is None:
        color = c.color.ORANGE

    plt.figure(figsize=(to_mm(30), to_mm(30)))

    num_bin = len(acc_bins)
    bin_width = 1.0 / num_bin

    # Bar plot for accuracy
    plt.bar(conf_bins, acc_bins, width=bin_width, color=color, alpha=0.7)

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=0.5)

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if title:
        plt.title(title)

    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_ece_bar(
    ece_wo: List[float],
    ece_w: List[float],
    save_path: str,
) -> str:
    """
    Plot ECE comparison bar chart.

    Args:
        ece_wo: ECE values without warmup (across networks)
        ece_w: ECE values with warmup (across networks)
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(20), to_mm(30)))

    ece_wo = np.array(ece_wo)
    ece_w = np.array(ece_w)

    bar_error([0], ece_wo.reshape(-1, 1), c.color.ORANGE)
    bar_error([1], ece_w.reshape(-1, 1), c.color.SKY)

    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("ECE")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_gap_bar(
    gap_wo: List[float],
    gap_w: List[float],
    save_path: str,
) -> str:
    """
    Plot accuracy-confidence gap comparison bar chart.

    Args:
        gap_wo: Gap values without warmup
        gap_w: Gap values with warmup
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(20), to_mm(30)))

    gap_wo = np.array(gap_wo)
    gap_w = np.array(gap_w)

    bar_error([0], gap_wo.reshape(-1, 1), c.color.ORANGE)
    bar_error([1], gap_w.reshape(-1, 1), c.color.SKY)

    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("Acc-Conf Gap")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_aggregate_boxplots(
    data_wo: List[float],
    data_w: List[float],
    save_path: str,
    ylabel: str = "Value",
    title: str = "",
) -> str:
    """
    Plot boxplot comparison of aggregate metrics.

    Args:
        data_wo: Data without warmup
        data_w: Data with warmup
        save_path: Path to save figure
        ylabel: Y-axis label
        title: Plot title

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(15), to_mm(30)))

    bp = plt.boxplot(
        [data_wo, data_w],
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
    )

    # Color the boxes
    bp['boxes'][0].set_facecolor(c.color.ORANGE)
    bp['boxes'][1].set_facecolor(c.color.SKY)

    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel(ylabel)

    if title:
        plt.title(title)

    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_confidence_histogram(
    conf: np.ndarray,
    save_path: str,
    title: str = "",
    color: str = None,
    bins: int = 20,
    show_mean: bool = True,
) -> str:
    """
    Plot confidence score histogram.

    Args:
        conf: Confidence scores
        save_path: Path to save figure
        title: Plot title
        color: Histogram color
        bins: Number of bins
        show_mean: Whether to show mean line

    Returns:
        Path to saved figure
    """
    if color is None:
        color = c.color.ORANGE

    plt.figure(figsize=(to_mm(30), to_mm(30)))

    plt.hist(conf, bins=bins, range=(0, 1), color=color, alpha=0.7, density=True)

    if show_mean:
        mean_conf = np.mean(conf)
        plt.axvline(mean_conf, color='k', linestyle='--', linewidth=0.5,
                    label=f"Mean: {mean_conf:.2f}")

    plt.xlabel("Confidence")
    plt.ylabel("Density")

    if title:
        plt.title(title)

    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_calibration_comparison(
    results_wo: Dict[str, Any],
    results_w: Dict[str, Any],
    save_dir: str,
    file_prefix: str = "",
) -> Dict[str, str]:
    """
    Generate all calibration comparison plots.

    Args:
        results_wo: Calibration results without warmup
        results_w: Calibration results with warmup
        save_dir: Directory to save figures
        file_prefix: Prefix for filenames

    Returns:
        Dict mapping figure names to paths
    """
    paths = {}
    save_dir = Path(save_dir)

    # Reliability diagrams
    rd_wo = results_wo.get("reliability_diagram", {})
    rd_w = results_w.get("reliability_diagram", {})

    if rd_wo:
        path = save_dir / f"{file_prefix}reliability_diagram_wo.svg"
        plot_reliability_diagram(
            rd_wo["acc"], rd_wo["conf_bin"], rd_wo["num_sample"],
            str(path), title="w/o warmup", color=c.color.ORANGE
        )
        paths["reliability_wo"] = str(path)

    if rd_w:
        path = save_dir / f"{file_prefix}reliability_diagram_w.svg"
        plot_reliability_diagram(
            rd_w["acc"], rd_w["conf_bin"], rd_w["num_sample"],
            str(path), title="w/ warmup", color=c.color.SKY
        )
        paths["reliability_w"] = str(path)

    # ECE bar
    if "ece_list" in results_wo and "ece_list" in results_w:
        path = save_dir / f"{file_prefix}ece_bar.svg"
        plot_ece_bar(results_wo["ece_list"], results_w["ece_list"], str(path))
        paths["ece_bar"] = str(path)

    # Gap bar
    if "gap_list" in results_wo and "gap_list" in results_w:
        path = save_dir / f"{file_prefix}gap_bar.svg"
        plot_gap_bar(results_wo["gap_list"], results_w["gap_list"], str(path))
        paths["gap_bar"] = str(path)

    return paths
