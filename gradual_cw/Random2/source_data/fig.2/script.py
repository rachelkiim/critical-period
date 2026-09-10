from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure `src` package import works when running:
# python3 source_data/fig.2/script.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom.figure import GraphConfig as c, plot_error, to_mm
from src.evaluation.stats import wilcoxon_signed_rank

INPUT_D = Path(__file__).resolve().parent / "d.json"
INPUT_E = Path(__file__).resolve().parent / "e.json"
INPUT_E_INSET = Path(__file__).resolve().parent / "e_inset.json"
INPUT_F = Path(__file__).resolve().parent / "f.json"
INPUT_G_H = Path(__file__).resolve().parent / "g-h.json"

OUT_D = Path(__file__).resolve().parent / "d.svg"
OUT_E = Path(__file__).resolve().parent / "e.svg"
OUT_E_INSET = Path(__file__).resolve().parent / "e_inset.svg"
OUT_F_LEFT = Path(__file__).resolve().parent / "f_left.svg"
OUT_F_RIGHT = Path(__file__).resolve().parent / "f_right.svg"
OUT_G = Path(__file__).resolve().parent / "g.svg"
OUT_H = Path(__file__).resolve().parent / "h.svg"
OUT_STATS = Path(__file__).resolve().parent / "stats.json"

COLOR_WITHOUT = "#EFB1A3"
COLOR_WITH = "#84C4FF"
COLOR_IDEAL = "#C1C1C1"

HEATMAP_CMAP = "coolwarm"
HEATMAP_CLIM = (0.0, 0.25)
HEATMAP_VMIN_VMAX = (0.0, 0.25)

GH_POSITIONS = [0, 1, 2.5, 3.5]
GH_LABELS = ["Task only", "Random + Task", "Pretraining + Task", "Pretraining + Random + Task"]
GH_COLORS = [c.color.ORANGE, c.color.SKY, "#D6342B", "#6AB518"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be object: {path}")
    return payload


def require_keys(obj: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise KeyError(f"Missing keys in {context}: {missing}")


def as_1d_float_array(values: Any, context: str, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{context} must not be empty.")
    return arr


def as_2d_float_array(values: Any, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{context} must be a non-empty 2D numeric array.")
    return arr


def _build_xticks(total_end: int, warmup_epochs: int, include_boundary: bool) -> list[int]:
    ticks = [0, total_end]
    if include_boundary and 0 < warmup_epochs < total_end:
        ticks.append(warmup_epochs)
        main_span = total_end - warmup_epochs
        if main_span > 1:
            ticks.append(warmup_epochs + (main_span // 2))
    elif total_end > 2:
        ticks.append(total_end // 2)
    return sorted(set(int(t) for t in ticks))


def _build_xtick_labels(ticks: list[int], warmup_epochs: int) -> list[str]:
    labels: list[str] = []
    for tick in ticks:
        if tick == warmup_epochs and warmup_epochs > 0:
            labels.append(f"{warmup_epochs}/0")
        elif tick > warmup_epochs and warmup_epochs > 0:
            labels.append(str(int(tick - warmup_epochs)))
        else:
            labels.append(str(int(tick)))
    return labels


def extract_reliability(entry: dict[str, Any], context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_keys(entry, ["conf_bin", "acc", "num_sample"], context)
    conf_bin = as_1d_float_array(entry["conf_bin"], f"{context}.conf_bin")
    acc = as_1d_float_array(entry["acc"], f"{context}.acc")
    num_sample = as_1d_float_array(entry["num_sample"], f"{context}.num_sample")
    if not (conf_bin.size == acc.size == num_sample.size):
        raise ValueError(f"{context}: conf_bin/acc/num_sample length mismatch.")
    return conf_bin, acc, num_sample


def build_heatmap_matrices(
    f_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[dict[str, Any]]]:
    settings = []
    for key, value in f_data.items():
        if key == "config":
            continue
        if not isinstance(value, dict):
            raise TypeError(f"f.json[{key}] must be object.")
        require_keys(value, ["depth", "num_image", "without_warmup", "with_warmup"], f"f.json[{key}]")
        wo = value["without_warmup"]
        w = value["with_warmup"]
        if not isinstance(wo, dict) or not isinstance(w, dict):
            raise TypeError(f"f.json[{key}].with/without_warmup must be objects.")
        require_keys(wo, ["ece_list"], f"f.json[{key}].without_warmup")
        require_keys(w, ["ece_list"], f"f.json[{key}].with_warmup")
        wo_ece = as_1d_float_array(wo["ece_list"], f"f.json[{key}].without_warmup.ece_list")
        w_ece = as_1d_float_array(w["ece_list"], f"f.json[{key}].with_warmup.ece_list")
        if wo_ece.size != w_ece.size:
            raise ValueError(f"f.json[{key}]: with/without ece_list length mismatch.")
        settings.append(
            {
                "run_key": key,
                "depth": int(value["depth"]),
                "num_image": int(value["num_image"]),
                "wo_ece": wo_ece,
                "w_ece": w_ece,
            }
        )

    if len(settings) == 0:
        raise ValueError("f.json contains no setting entries.")

    depths = sorted({s["depth"] for s in settings})
    num_images = sorted({s["num_image"] for s in settings})
    depth_to_idx = {depth: i for i, depth in enumerate(depths)}
    num_to_idx = {num: i for i, num in enumerate(num_images)}

    matrix_wo = np.full((len(num_images), len(depths)), np.nan, dtype=float)
    matrix_w = np.full((len(num_images), len(depths)), np.nan, dtype=float)

    for s in settings:
        row = num_to_idx[s["num_image"]]
        col = depth_to_idx[s["depth"]]
        if not np.isnan(matrix_wo[row, col]) or not np.isnan(matrix_w[row, col]):
            raise ValueError(
                f"Duplicate depth/num_image entry in f.json for depth={s['depth']}, num_image={s['num_image']}"
            )
        matrix_wo[row, col] = float(np.mean(s["wo_ece"]))
        matrix_w[row, col] = float(np.mean(s["w_ece"]))

    if np.isnan(matrix_wo).any() or np.isnan(matrix_w).any():
        raise ValueError("f.json is missing at least one depth/num_image combination.")

    settings.sort(key=lambda x: (x["depth"], x["num_image"]))
    return matrix_wo, matrix_w, depths, num_images, settings


def load_data() -> dict[str, Any]:
    d_data = load_json(INPUT_D)
    e_data = load_json(INPUT_E)
    e_inset_data = load_json(INPUT_E_INSET)
    f_data = load_json(INPUT_F)
    g_h_data = load_json(INPUT_G_H)

    require_keys(d_data, ["without_warmup", "with_warmup", "warmup"], "d.json")
    wo_loss = d_data["without_warmup"]
    w_loss = d_data["with_warmup"]
    warmup_loss = d_data["warmup"]
    if not isinstance(wo_loss, dict) or not isinstance(w_loss, dict) or not isinstance(warmup_loss, dict):
        raise TypeError("d.json with/without/warmup sections must be objects.")
    require_keys(wo_loss, ["test_loss_list"], "d.json.without_warmup")
    require_keys(w_loss, ["test_loss_list"], "d.json.with_warmup")
    require_keys(warmup_loss, ["test_loss_list"], "d.json.warmup")
    d_wo = as_2d_float_array(wo_loss["test_loss_list"], "d.json.without_warmup.test_loss_list")
    d_w = as_2d_float_array(w_loss["test_loss_list"], "d.json.with_warmup.test_loss_list")
    d_warmup = as_2d_float_array(warmup_loss["test_loss_list"], "d.json.warmup.test_loss_list")

    require_keys(e_data, ["with_warmup", "without_warmup"], "e.json")
    if not isinstance(e_data["with_warmup"], dict) or not isinstance(e_data["without_warmup"], dict):
        raise TypeError("e.json with/without_warmup sections must be objects.")
    e_w_conf, e_w_acc, _ = extract_reliability(e_data["with_warmup"], "e.json.with_warmup")
    e_wo_conf, e_wo_acc, _ = extract_reliability(e_data["without_warmup"], "e.json.without_warmup")

    require_keys(e_inset_data, ["with_warmup", "without_warmup"], "e_inset.json")
    if not isinstance(e_inset_data["with_warmup"], dict) or not isinstance(e_inset_data["without_warmup"], dict):
        raise TypeError("e_inset.json with/without_warmup sections must be objects.")
    require_keys(e_inset_data["with_warmup"], ["ece_list"], "e_inset.json.with_warmup")
    require_keys(e_inset_data["without_warmup"], ["ece_list"], "e_inset.json.without_warmup")
    e_inset_w = as_1d_float_array(e_inset_data["with_warmup"]["ece_list"], "e_inset.json.with_warmup.ece_list")
    e_inset_wo = as_1d_float_array(
        e_inset_data["without_warmup"]["ece_list"], "e_inset.json.without_warmup.ece_list"
    )
    if e_inset_w.size != e_inset_wo.size:
        raise ValueError("e_inset.json: with/without ece_list length mismatch.")

    f_wo_matrix, f_w_matrix, f_depths, f_num_images, f_settings = build_heatmap_matrices(f_data)

    require_keys(g_h_data, ["scratch", "finetuning"], "g-h.json")
    scratch = g_h_data["scratch"]
    finetuning = g_h_data["finetuning"]
    if not isinstance(scratch, dict) or not isinstance(finetuning, dict):
        raise TypeError("g-h.json scratch/finetuning sections must be objects.")

    def _extract_gh_group(
        root: dict[str, Any],
        root_name: str,
        warmup_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        require_keys(root, [warmup_key], f"g-h.json.{root_name}")
        section = root[warmup_key]
        if not isinstance(section, dict):
            raise TypeError(f"g-h.json.{root_name}.{warmup_key} must be object.")
        require_keys(section, ["ece_list", "acc_list"], f"g-h.json.{root_name}.{warmup_key}")
        ece = as_1d_float_array(section["ece_list"], f"g-h.json.{root_name}.{warmup_key}.ece_list")
        acc = as_1d_float_array(section["acc_list"], f"g-h.json.{root_name}.{warmup_key}.acc_list")
        return ece, acc

    scratch_wo_ece, scratch_wo_acc = _extract_gh_group(scratch, "scratch", "without_warmup")
    scratch_w_ece, scratch_w_acc = _extract_gh_group(scratch, "scratch", "with_warmup")
    finetuning_wo_ece, finetuning_wo_acc = _extract_gh_group(finetuning, "finetuning", "without_warmup")
    finetuning_w_ece, finetuning_w_acc = _extract_gh_group(finetuning, "finetuning", "with_warmup")

    ece_lengths = {
        scratch_wo_ece.size,
        scratch_w_ece.size,
        finetuning_wo_ece.size,
        finetuning_w_ece.size,
    }
    if len(ece_lengths) != 1:
        raise ValueError(
            "g-h.json ece_list length mismatch among four groups: "
            "scratch.without_warmup / scratch.with_warmup / "
            "finetuning.without_warmup / finetuning.with_warmup"
        )

    acc_lengths = {
        scratch_wo_acc.size,
        scratch_w_acc.size,
        finetuning_wo_acc.size,
        finetuning_w_acc.size,
    }
    if len(acc_lengths) != 1:
        raise ValueError(
            "g-h.json acc_list length mismatch among four groups: "
            "scratch.without_warmup / scratch.with_warmup / "
            "finetuning.without_warmup / finetuning.with_warmup"
        )

    return {
        "d_wo": d_wo,
        "d_w": d_w,
        "d_warmup": d_warmup,
        "e_w_conf": e_w_conf,
        "e_w_acc": e_w_acc,
        "e_wo_conf": e_wo_conf,
        "e_wo_acc": e_wo_acc,
        "e_inset_w": e_inset_w,
        "e_inset_wo": e_inset_wo,
        "f_wo_matrix": f_wo_matrix,
        "f_w_matrix": f_w_matrix,
        "f_depths": f_depths,
        "f_num_images": f_num_images,
        "f_settings": f_settings,
        "g_task_only_ece": scratch_wo_ece,
        "g_random_task_ece": scratch_w_ece,
        "g_pretrain_task_ece": finetuning_wo_ece,
        "g_pretrain_random_task_ece": finetuning_w_ece,
        "h_task_only_acc": scratch_wo_acc,
        "h_random_task_acc": scratch_w_acc,
        "h_pretrain_task_acc": finetuning_wo_acc,
        "h_pretrain_random_task_acc": finetuning_w_acc,
    }


def plot_d(loss_wo: np.ndarray, loss_w: np.ndarray, warmup_loss: np.ndarray, save_path: Path) -> dict[str, Any]:
    plt.figure(figsize=(to_mm(40), to_mm(30)))

    series_len = min(loss_wo.shape[1], loss_w.shape[1])
    wo_data = loss_wo[:, :series_len]
    w_data = loss_w[:, :series_len]

    warmup_epochs = max(0, int(warmup_loss.shape[1] - 1))
    x_offset = warmup_epochs
    x_main = x_offset + np.arange(series_len)

    plot_error(wo_data, c.color.ORANGE, xrange=x_main)
    plot_error(w_data, c.color.SKY, xrange=x_main)

    if warmup_loss.shape[1] > 0:
        x_warmup = np.arange(warmup_loss.shape[1])
        plot_error(warmup_loss, c.color.SKY, xrange=x_warmup)

    total_end = int(x_main[-1]) if x_main.size > 0 else 1
    if warmup_loss.shape[1] > 0:
        total_end = max(total_end, int(warmup_loss.shape[1] - 1))
    total_end = max(total_end, 1)

    if warmup_epochs > 0:
        plt.axvline(x=warmup_epochs, color="black", linestyle="--", linewidth=0.5)

    ticks = _build_xticks(total_end, warmup_epochs, include_boundary=warmup_epochs > 0)
    plt.xticks(ticks, _build_xtick_labels(ticks, warmup_epochs))
    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.xlim(0, total_end)
    plt.ylim(1.5, 3.5)
    plt.savefig(save_path)
    plt.close()

    # Extract final test loss values for statistical test
    final_loss_wo = loss_wo[:, -1]
    final_loss_w = loss_w[:, -1]

    # Perform Wilcoxon signed rank test
    stats = wilcoxon_signed_rank(
        final_loss_wo,
        final_loss_w,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="approx",
    )

    return {
        "test_result": stats,
        "summary": {
            "without_warmup": {
                "mean": float(np.mean(final_loss_wo)),
                "median": float(np.median(final_loss_wo)),
                "std": float(np.std(final_loss_wo)),
            },
            "with_warmup": {
                "mean": float(np.mean(final_loss_w)),
                "median": float(np.median(final_loss_w)),
                "std": float(np.std(final_loss_w)),
            },
        },
        "test_config": {
            "test": "wilcoxon_signed_rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "correction": False,
            "method": "approx",
        },
    }


def plot_e(
    conf_w: np.ndarray,
    acc_w: np.ndarray,
    conf_wo: np.ndarray,
    acc_wo: np.ndarray,
    save_path: Path,
) -> None:
    plt.figure(figsize=(to_mm(30), to_mm(30)))

    bin_width_w = 1.0 / conf_w.size
    bin_width_wo = 1.0 / conf_wo.size

    plt.bar(conf_w, acc_w, width=bin_width_w, color=COLOR_WITH, zorder=1.0)
    plt.bar(conf_wo, acc_wo, width=bin_width_wo, color=COLOR_WITHOUT, zorder=1.1)
    plt.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=0.5, zorder=2.0)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.savefig(save_path)
    plt.close()


def plot_e_inset(ece_wo: np.ndarray, ece_w: np.ndarray, save_path: Path) -> dict[str, Any]:
    wo_pct = 100.0 * ece_wo
    w_pct = 100.0 * ece_w

    plt.figure(figsize=(to_mm(7), to_mm(15)))

    bar_means = [float(np.mean(wo_pct)), float(np.mean(w_pct))]
    plt.bar([0, 1], bar_means, width=0.5, color=[c.color.ORANGE, c.color.SKY], alpha=0.2, zorder=0)

    rng = np.random.default_rng(0)
    for pos, values in zip([0, 1], [wo_pct, w_pct], strict=True):
        jitter = rng.normal(0.0, 0.1, size=values.size)
        plt.scatter(
            pos + jitter,
            values,
            color=c.color.DARKGRAY,
            s=10,
            alpha=0.5,
            edgecolors="none",
            linewidths=0,
            zorder=1,
        )

    bp = plt.boxplot([wo_pct, w_pct], positions=[0, 1], widths=0.5, patch_artist=True, zorder=2)
    bp["boxes"][0].set_facecolor(c.color.ORANGE)
    bp["boxes"][1].set_facecolor(c.color.SKY)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_alpha(0.7)

    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("Error (%)")
    plt.ylim(0, 20)
    plt.yticks([0, 20])
    plt.savefig(save_path)
    plt.close()

    stats = wilcoxon_signed_rank(
        ece_wo,
        ece_w,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="approx",
    )
    return {
        "test_result": stats,
        "summary": {
            "without_warmup_percent": {
                "mean": float(np.mean(wo_pct)),
                "median": float(np.median(wo_pct)),
                "std": float(np.std(wo_pct)),
            },
            "with_warmup_percent": {
                "mean": float(np.mean(w_pct)),
                "median": float(np.median(w_pct)),
                "std": float(np.std(w_pct)),
            },
        },
        "test_config": {
            "test": "wilcoxon_signed_rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "correction": False,
            "method": "approx",
        },
    }


def plot_heatmap_common(matrix: np.ndarray, depths: list[int], num_images: list[int], save_path: Path) -> None:
    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.imshow(
        matrix,
        origin="lower",
        cmap=HEATMAP_CMAP,
        vmin=HEATMAP_VMIN_VMAX[0],
        vmax=HEATMAP_VMIN_VMAX[1],
        aspect="auto",
    )
    plt.clim(*HEATMAP_CLIM)
    plt.xlabel("Network depth")
    plt.ylabel("Size of training data")
    ax = plt.gca()
    ax.set_xticks(np.arange(len(depths)), depths)
    ax.set_yticks(np.arange(len(num_images)), [f"{num / 1000:g}" for num in num_images])
    ax.text(-0.05, 1.01, "$×10³$", transform=ax.transAxes, ha="left", va="bottom")
    plt.savefig(save_path)
    plt.close()


def plot_f_left(matrix_wo: np.ndarray, depths: list[int], num_images: list[int], save_path: Path) -> None:
    plot_heatmap_common(matrix_wo, depths, num_images, save_path)


def plot_f_right(matrix_w: np.ndarray, depths: list[int], num_images: list[int], save_path: Path) -> None:
    plot_heatmap_common(matrix_w, depths, num_images, save_path)


def plot_horizontal_four_group_boxplot(
    data_groups: list[np.ndarray],
    xlabel: str,
    xlim: tuple[float, float],
    xticks: list[float] | None,
    save_path: Path,
) -> None:
    plt.figure(figsize=(to_mm(20), to_mm(30)))

    rng = np.random.default_rng(0)
    for pos, values in zip(GH_POSITIONS, data_groups, strict=True):
        jitter = rng.normal(0.0, 0.1, size=values.size)
        plt.scatter(
            values,
            pos + jitter,
            color=c.color.DARKGRAY,
            s=10,
            alpha=0.5,
            edgecolors="none",
            linewidths=0,
            zorder=1,
        )

    bp = plt.boxplot(
        data_groups,
        positions=GH_POSITIONS,
        vert=False,
        patch_artist=True,
        zorder=2,
    )
    for box, color in zip(bp["boxes"], GH_COLORS, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    plt.yticks(GH_POSITIONS, GH_LABELS)
    plt.ylim(-0.8, 4.3)
    plt.xlabel(xlabel)
    plt.xlim(*xlim)
    if xticks is not None:
        plt.xticks(xticks)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.close()


def summarize_groups(
    task_only: np.ndarray,
    random_task: np.ndarray,
    pretraining_task: np.ndarray,
    pretraining_random_task: np.ndarray,
) -> dict[str, Any]:
    return {
        "task_only": {
            "mean": float(np.mean(task_only)),
            "median": float(np.median(task_only)),
            "std": float(np.std(task_only)),
        },
        "random_task": {
            "mean": float(np.mean(random_task)),
            "median": float(np.median(random_task)),
            "std": float(np.std(random_task)),
        },
        "pretraining_task": {
            "mean": float(np.mean(pretraining_task)),
            "median": float(np.median(pretraining_task)),
            "std": float(np.std(pretraining_task)),
        },
        "pretraining_random_task": {
            "mean": float(np.mean(pretraining_random_task)),
            "median": float(np.median(pretraining_random_task)),
            "std": float(np.std(pretraining_random_task)),
        },
    }


def run_two_pair_wilcoxon(
    task_only: np.ndarray,
    random_task: np.ndarray,
    pretraining_task: np.ndarray,
    pretraining_random_task: np.ndarray,
) -> dict[str, Any]:
    return {
        "task_only_vs_random_task": wilcoxon_signed_rank(
            task_only,
            random_task,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="approx",
        ),
        "pretraining_task_vs_pretraining_random_task": wilcoxon_signed_rank(
            pretraining_task,
            pretraining_random_task,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="approx",
        ),
    }


def plot_g(
    task_only_ece: np.ndarray,
    random_task_ece: np.ndarray,
    pretraining_task_ece: np.ndarray,
    pretraining_random_task_ece: np.ndarray,
    save_path: Path,
) -> dict[str, Any]:
    task_only_pct = 100.0 * task_only_ece
    random_task_pct = 100.0 * random_task_ece
    pretraining_task_pct = 100.0 * pretraining_task_ece
    pretraining_random_task_pct = 100.0 * pretraining_random_task_ece

    plot_horizontal_four_group_boxplot(
        [task_only_pct, random_task_pct, pretraining_task_pct, pretraining_random_task_pct],
        xlabel="Error (%)",
        xlim=(0, 8),
        xticks=[0, 2, 4, 6, 8],
        save_path=save_path,
    )

    return {
        "test_result": run_two_pair_wilcoxon(
            task_only_pct,
            random_task_pct,
            pretraining_task_pct,
            pretraining_random_task_pct,
        ),
        "summary": summarize_groups(
            task_only_pct,
            random_task_pct,
            pretraining_task_pct,
            pretraining_random_task_pct,
        ),
        "test_config": {
            "test": "wilcoxon_signed_rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "correction": False,
            "method": "approx",
            "num_tests": 2,
        },
    }


def plot_h(
    task_only_acc: np.ndarray,
    random_task_acc: np.ndarray,
    pretraining_task_acc: np.ndarray,
    pretraining_random_task_acc: np.ndarray,
    save_path: Path,
) -> dict[str, Any]:
    plot_horizontal_four_group_boxplot(
        [task_only_acc, random_task_acc, pretraining_task_acc, pretraining_random_task_acc],
        xlabel="Accuracy",
        xlim=(0.87, 0.98),
        xticks=[0.9, 0.94, 0.98],
        save_path=save_path,
    )

    return {
        "test_result": run_two_pair_wilcoxon(
            task_only_acc,
            random_task_acc,
            pretraining_task_acc,
            pretraining_random_task_acc,
        ),
        "summary": summarize_groups(
            task_only_acc,
            random_task_acc,
            pretraining_task_acc,
            pretraining_random_task_acc,
        ),
        "test_config": {
            "test": "wilcoxon_signed_rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "correction": False,
            "method": "approx",
            "num_tests": 2,
        },
    }


def compute_f_wilcoxon_by_setting(settings: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for s in settings:
        test_result = wilcoxon_signed_rank(
            s["wo_ece"],
            s["w_ece"],
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="approx",
        )
        results[s["run_key"]] = {
            "depth": s["depth"],
            "num_image": s["num_image"],
            "test_result": test_result,
        }
    return results


def write_stats(stats_payload: dict[str, Any]) -> None:
    with OUT_STATS.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)
        f.write("\n")


def main() -> None:
    data = load_data()

    d_info = plot_d(data["d_wo"], data["d_w"], data["d_warmup"], OUT_D)
    plot_e(data["e_w_conf"], data["e_w_acc"], data["e_wo_conf"], data["e_wo_acc"], OUT_E)
    e_inset_info = plot_e_inset(data["e_inset_wo"], data["e_inset_w"], OUT_E_INSET)
    plot_f_left(data["f_wo_matrix"], data["f_depths"], data["f_num_images"], OUT_F_LEFT)
    plot_f_right(data["f_w_matrix"], data["f_depths"], data["f_num_images"], OUT_F_RIGHT)
    g_info = plot_g(
        data["g_task_only_ece"],
        data["g_random_task_ece"],
        data["g_pretrain_task_ece"],
        data["g_pretrain_random_task_ece"],
        OUT_G,
    )
    h_info = plot_h(
        data["h_task_only_acc"],
        data["h_random_task_acc"],
        data["h_pretrain_task_acc"],
        data["h_pretrain_random_task_acc"],
        OUT_H,
    )

    f_pair_tests = compute_f_wilcoxon_by_setting(data["f_settings"])

    stats_payload = {
        "d_wilcoxon_signed_rank": d_info["test_result"],
        "e_inset_wilcoxon_signed_rank": e_inset_info["test_result"],
        "f_wilcoxon_signed_rank_by_setting": f_pair_tests,
        "g_wilcoxon_signed_rank": g_info["test_result"],
        "h_wilcoxon_signed_rank": h_info["test_result"],
        "summary": {
            "d_final_test_loss": d_info["summary"],
            "e_inset": e_inset_info["summary"],
            "f_heatmap_cell_mean": {
                "without_warmup": {
                    "mean": float(np.mean(data["f_wo_matrix"])),
                    "median": float(np.median(data["f_wo_matrix"])),
                    "std": float(np.std(data["f_wo_matrix"])),
                },
                "with_warmup": {
                    "mean": float(np.mean(data["f_w_matrix"])),
                    "median": float(np.median(data["f_w_matrix"])),
                    "std": float(np.std(data["f_w_matrix"])),
                },
            },
            "g_ece_percent": g_info["summary"],
            "h_accuracy": h_info["summary"],
        },
        "config": {
            "input_files": {
                "d": str(INPUT_D),
                "e": str(INPUT_E),
                "e_inset": str(INPUT_E_INSET),
                "f": str(INPUT_F),
                "g_h": str(INPUT_G_H),
            },
            "output_files": {
                "d": str(OUT_D),
                "e": str(OUT_E),
                "e_inset": str(OUT_E_INSET),
                "f_left": str(OUT_F_LEFT),
                "f_right": str(OUT_F_RIGHT),
                "g": str(OUT_G),
                "h": str(OUT_H),
                "stats": str(OUT_STATS),
            },
            "tests": {
                "d": d_info["test_config"],
                "e_inset": e_inset_info["test_config"],
                "f_by_setting": {
                    "test": "wilcoxon_signed_rank",
                    "alternative": "two-sided",
                    "zero_method": "wilcox",
                    "correction": False,
                    "method": "approx",
                    "num_tests": len(f_pair_tests),
                },
                "g": g_info["test_config"],
                "h": h_info["test_config"],
            },
            "colors": {
                "without_warmup": COLOR_WITHOUT,
                "with_warmup_reliability": COLOR_WITH,
                "ideal": COLOR_IDEAL,
                "learning_curve_without_warmup": c.color.ORANGE,
                "learning_curve_with_warmup": c.color.SKY,
                "g_h_boxplot": GH_COLORS,
            },
            "heatmap": {
                "aggregation": "mean(ece_list)",
                "cmap": HEATMAP_CMAP,
                "vmin_vmax": list(HEATMAP_VMIN_VMAX),
                "clim": list(HEATMAP_CLIM),
            },
        },
    }
    write_stats(stats_payload)

    print(f"Saved: {OUT_D}")
    print(f"Saved: {OUT_E}")
    print(f"Saved: {OUT_E_INSET}")
    print(f"Saved: {OUT_F_LEFT}")
    print(f"Saved: {OUT_F_RIGHT}")
    print(f"Saved: {OUT_G}")
    print(f"Saved: {OUT_H}")
    print(f"Saved: {OUT_STATS}")


if __name__ == "__main__":
    main()
