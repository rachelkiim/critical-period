"""OOD (Out-of-Distribution) detection visualization plots."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from custom.figure import to_mm, GraphConfig as c, plot_error, bar_error


def plot_ood_histogram(
    conf_ood_wo: np.ndarray,
    conf_ood_w: np.ndarray,
    save_path: str,
    bins: int = 20,
) -> str:
    """
    Plot OOD confidence histogram comparison.

    Args:
        conf_ood_wo: OOD confidence scores without warmup
        conf_ood_w: OOD confidence scores with warmup
        save_path: Path to save figure
        bins: Number of bins

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))

    plt.hist(conf_ood_wo, bins=bins, range=(0, 1), alpha=0.5,
             color=c.color.ORANGE, density=True, label="w/o warmup")
    plt.hist(conf_ood_w, bins=bins, range=(0, 1), alpha=0.5,
             color=c.color.SKY, density=True, label="w/ warmup")

    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_ood_boxplot(
    conf_ood_wo: np.ndarray,
    conf_ood_w: np.ndarray,
    save_path: str,
) -> str:
    """
    Plot OOD confidence boxplot comparison.

    Args:
        conf_ood_wo: OOD confidence scores without warmup
        conf_ood_w: OOD confidence scores with warmup
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(10), to_mm(30)))

    bp = plt.boxplot(
        [conf_ood_wo, conf_ood_w],
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
    )

    bp['boxes'][0].set_facecolor(c.color.ORANGE)
    bp['boxes'][1].set_facecolor(c.color.SKY)

    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("OOD Confidence")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_id_ood_histogram(
    conf_id: np.ndarray,
    conf_ood: np.ndarray,
    save_path: str,
    title: str = "",
    bins: int = 20,
) -> str:
    """
    Plot ID vs OOD confidence histogram.

    Args:
        conf_id: In-distribution confidence scores
        conf_ood: Out-of-distribution confidence scores
        save_path: Path to save figure
        title: Plot title
        bins: Number of bins

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(20)))

    plt.hist(conf_id, bins=bins, range=(0, 1), alpha=0.5,
             color=c.color.DARKGRAY, density=True, label="ID")
    plt.hist(conf_ood, bins=bins, range=(0, 1), alpha=0.5,
             color=c.color.ORANGE, density=True, label="OOD")

    plt.xlabel("Confidence")
    plt.ylabel("Density")

    if title:
        plt.title(title)

    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_cdf(
    conf_id: np.ndarray,
    conf_ood: np.ndarray,
    save_path: str,
    bins: int = 20,
) -> str:
    """
    Plot CDF comparison of ID vs OOD confidence.

    Args:
        conf_id: In-distribution confidence scores
        conf_ood: Out-of-distribution confidence scores
        save_path: Path to save figure
        bins: Number of bins

    Returns:
        Path to saved figure
    """
    from src.evaluation.ood_detection import cdf

    plt.figure(figsize=(to_mm(30), to_mm(10)))

    cdf_id, edges_id = cdf(conf_id, bins=bins)
    cdf_ood, edges_ood = cdf(conf_ood, bins=bins)

    # Use bin centers for x-axis
    centers_id = (edges_id[:-1] + edges_id[1:]) / 2
    centers_ood = (edges_ood[:-1] + edges_ood[1:]) / 2

    plt.plot(centers_id, cdf_id, color=c.color.DARKGRAY, label="ID")
    plt.plot(centers_ood, cdf_ood, color=c.color.ORANGE, label="OOD")

    plt.xlabel("Confidence")
    plt.ylabel("CDF")
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_roc_curve(
    fp_wo: np.ndarray,
    tp_wo: np.ndarray,
    fp_w: np.ndarray,
    tp_w: np.ndarray,
    save_path: str,
    auroc_wo: Optional[float] = None,
    auroc_w: Optional[float] = None,
) -> str:
    """
    Plot ROC curve comparison.

    Args:
        fp_wo: False positive rates without warmup
        tp_wo: True positive rates without warmup
        fp_w: False positive rates with warmup
        tp_w: True positive rates with warmup
        save_path: Path to save figure
        auroc_wo: AUROC without warmup (for legend)
        auroc_w: AUROC with warmup (for legend)

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(30), to_mm(30)))

    label_wo = "w/o warmup"
    label_w = "w/ warmup"

    if auroc_wo is not None:
        label_wo += f" (AUC={auroc_wo:.3f})"
    if auroc_w is not None:
        label_w += f" (AUC={auroc_w:.3f})"

    plt.plot(fp_wo, tp_wo, color=c.color.ORANGE, label=label_wo)
    plt.plot(fp_w, tp_w, color=c.color.SKY, label=label_w)

    # Diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_auroc_bar(
    auroc_wo: List[float],
    auroc_w: List[float],
    save_path: str,
) -> str:
    """
    Plot AUROC comparison bar chart.

    Args:
        auroc_wo: AUROC values without warmup (across networks)
        auroc_w: AUROC values with warmup (across networks)
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(20), to_mm(30)))

    auroc_wo = np.array(auroc_wo)
    auroc_w = np.array(auroc_w)

    bar_error([0], auroc_wo.reshape(-1, 1), c.color.ORANGE)
    bar_error([1], auroc_w.reshape(-1, 1), c.color.SKY)

    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("AUROC")
    plt.ylim(0.5, 1.0)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_auroc_boxplot(
    auroc_wo: List[float],
    auroc_w: List[float],
    save_path: str,
) -> str:
    """
    Plot AUROC comparison boxplot.

    Args:
        auroc_wo: AUROC values without warmup
        auroc_w: AUROC values with warmup
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plt.figure(figsize=(to_mm(15), to_mm(30)))

    bp = plt.boxplot(
        [auroc_wo, auroc_w],
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
    )

    bp['boxes'][0].set_facecolor(c.color.ORANGE)
    bp['boxes'][1].set_facecolor(c.color.SKY)

    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("AUROC")
    plt.ylim(0.5, 1.0)
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_ood_comparison(
    results_wo: Dict[str, Any],
    results_w: Dict[str, Any],
    save_dir: str,
    file_prefix: str = "",
) -> Dict[str, str]:
    """
    Generate all OOD detection comparison plots.

    Args:
        results_wo: OOD results without warmup
        results_w: OOD results with warmup
        save_dir: Directory to save figures
        file_prefix: Prefix for filenames

    Returns:
        Dict mapping figure names to paths
    """
    paths = {}
    save_dir = Path(save_dir)

    conf_id_wo = np.array(results_wo.get("conf_id", []))
    conf_id_w = np.array(results_w.get("conf_id", []))
    conf_ood_wo = np.array(results_wo.get("conf_ood", []))
    conf_ood_w = np.array(results_w.get("conf_ood", []))

    # OOD histogram comparison
    if len(conf_ood_wo) > 0 and len(conf_ood_w) > 0:
        path = save_dir / f"{file_prefix}ood_histogram.svg"
        plot_ood_histogram(conf_ood_wo, conf_ood_w, str(path))
        paths["ood_histogram"] = str(path)

        # OOD boxplot
        path = save_dir / f"{file_prefix}ood_boxplot.svg"
        plot_ood_boxplot(conf_ood_wo, conf_ood_w, str(path))
        paths["ood_boxplot"] = str(path)

    # ID vs OOD histograms
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

    # ROC curves
    roc_wo = results_wo.get("roc", {})
    roc_w = results_w.get("roc", {})
    auroc_wo_plot = results_wo.get("auroc")
    auroc_w_plot = results_w.get("auroc")
    if auroc_wo_plot is None and "auroc_list" in results_wo:
        auroc_wo_plot = float(np.mean(results_wo["auroc_list"]))
    if auroc_w_plot is None and "auroc_list" in results_w:
        auroc_w_plot = float(np.mean(results_w["auroc_list"]))
    if roc_wo and roc_w:
        path = save_dir / f"{file_prefix}roc_curve.svg"
        plot_roc_curve(
            np.array(roc_wo["fp"]), np.array(roc_wo["tp"]),
            np.array(roc_w["fp"]), np.array(roc_w["tp"]),
            str(path),
            auroc_wo=auroc_wo_plot,
            auroc_w=auroc_w_plot,
        )
        paths["roc_curve"] = str(path)

    # AUROC bar
    if "auroc_list" in results_wo and "auroc_list" in results_w:
        path = save_dir / f"{file_prefix}auroc_bar.svg"
        plot_auroc_bar(results_wo["auroc_list"], results_w["auroc_list"], str(path))
        paths["auroc_bar"] = str(path)

        # AUROC boxplot
        path = save_dir / f"{file_prefix}auroc_boxplot.svg"
        plot_auroc_boxplot(results_wo["auroc_list"], results_w["auroc_list"], str(path))
        paths["auroc_boxplot"] = str(path)

    return paths
