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
# python3 source_data/fig.4/script.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom.figure import GraphConfig as c, to_mm
from src.evaluation.stats import wilcoxon_signed_rank

INPUT_A = Path(__file__).resolve().parent / "a.json"
INPUT_B_LEFT = Path(__file__).resolve().parent / "b_left.json"
INPUT_B_RIGHT = Path(__file__).resolve().parent / "b_right.json"
INPUT_C_LEFT = Path(__file__).resolve().parent / "c_left.json"
INPUT_C_RIGHT = Path(__file__).resolve().parent / "c_right.json"
INPUT_D_F = Path(__file__).resolve().parent / "d-f.json"

OUT_A = Path(__file__).resolve().parent / "a.svg"
OUT_B_LEFT = Path(__file__).resolve().parent / "b_left.svg"
OUT_B_RIGHT = Path(__file__).resolve().parent / "b_right.svg"
OUT_C_LEFT = Path(__file__).resolve().parent / "c_left.svg"
OUT_C_RIGHT = Path(__file__).resolve().parent / "c_right.svg"
OUT_D = Path(__file__).resolve().parent / "d.svg"
OUT_E = Path(__file__).resolve().parent / "e.svg"
OUT_F = Path(__file__).resolve().parent / "f.svg"
OUT_STATS = Path(__file__).resolve().parent / "stats.json"

COLOR_WITHOUT = "#EFB1A3"
COLOR_WITH = "#84C4FF"
PANEL_A_D_DOT_SIZE = 3.0


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


def as_int(value: Any, context: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{context} must be int.")
    return int(value)


def extract_reliability(entry: dict[str, Any], context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_keys(entry, ["conf_bin", "acc", "num_sample"], context)
    conf_bin = as_1d_float_array(entry["conf_bin"], f"{context}.conf_bin")
    acc = as_1d_float_array(entry["acc"], f"{context}.acc")
    num_sample = as_1d_float_array(entry["num_sample"], f"{context}.num_sample")
    if not (conf_bin.size == acc.size == num_sample.size):
        raise ValueError(f"{context}: conf_bin/acc/num_sample length mismatch.")
    return conf_bin, acc, num_sample


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


def load_data() -> dict[str, Any]:
    a_data = load_json(INPUT_A)
    b_left_data = load_json(INPUT_B_LEFT)
    b_right_data = load_json(INPUT_B_RIGHT)
    c_left_data = load_json(INPUT_C_LEFT)
    c_right_data = load_json(INPUT_C_RIGHT)
    d_f_data = load_json(INPUT_D_F)

    require_keys(a_data, ["without_warmup", "with_warmup"], "a.json")
    wo_a = a_data["without_warmup"]
    w_a = a_data["with_warmup"]
    if not isinstance(wo_a, dict) or not isinstance(w_a, dict):
        raise TypeError("a.json without_warmup/with_warmup must be objects.")
    require_keys(wo_a, ["test_loss", "test_acc", "selected_epochs"], "a.json.without_warmup")
    require_keys(w_a, ["test_loss", "test_acc", "warmup", "selected_epochs"], "a.json.with_warmup")
    if not isinstance(wo_a["selected_epochs"], dict) or not isinstance(w_a["selected_epochs"], dict):
        raise TypeError("a.json selected_epochs must be objects.")
    require_keys(wo_a["selected_epochs"], ["acc025", "final"], "a.json.without_warmup.selected_epochs")
    require_keys(w_a["selected_epochs"], ["acc025", "final"], "a.json.with_warmup.selected_epochs")
    if not isinstance(w_a["warmup"], dict):
        raise TypeError("a.json.with_warmup.warmup must be object.")
    require_keys(w_a["warmup"], ["test_loss", "test_acc"], "a.json.with_warmup.warmup")

    a_wo_loss = as_1d_float_array(wo_a["test_loss"], "a.json.without_warmup.test_loss")
    a_wo_acc = as_1d_float_array(wo_a["test_acc"], "a.json.without_warmup.test_acc")
    a_w_loss = as_1d_float_array(w_a["test_loss"], "a.json.with_warmup.test_loss")
    a_w_acc = as_1d_float_array(w_a["test_acc"], "a.json.with_warmup.test_acc")
    a_warmup_loss = as_1d_float_array(w_a["warmup"]["test_loss"], "a.json.with_warmup.warmup.test_loss")
    a_warmup_acc = as_1d_float_array(w_a["warmup"]["test_acc"], "a.json.with_warmup.warmup.test_acc")
    if a_wo_loss.size != a_wo_acc.size:
        raise ValueError("a.json: without_warmup test_loss/test_acc length mismatch.")
    if a_w_loss.size != a_w_acc.size:
        raise ValueError("a.json: with_warmup test_loss/test_acc length mismatch.")
    if a_warmup_loss.size != a_warmup_acc.size:
        raise ValueError("a.json: with_warmup.warmup test_loss/test_acc length mismatch.")
    a_wo_acc025_epoch = as_int(wo_a["selected_epochs"]["acc025"], "a.json.without_warmup.selected_epochs.acc025")
    a_wo_final_epoch = as_int(wo_a["selected_epochs"]["final"], "a.json.without_warmup.selected_epochs.final")
    a_w_acc025_epoch = as_int(w_a["selected_epochs"]["acc025"], "a.json.with_warmup.selected_epochs.acc025")
    a_w_final_epoch = as_int(w_a["selected_epochs"]["final"], "a.json.with_warmup.selected_epochs.final")

    require_keys(b_left_data, ["without_warmup", "with_warmup"], "b_left.json")
    require_keys(c_left_data, ["without_warmup", "with_warmup"], "c_left.json")
    b_wo_conf, b_wo_acc, _ = extract_reliability(b_left_data["without_warmup"], "b_left.json.without_warmup")
    b_w_conf, b_w_acc, _ = extract_reliability(b_left_data["with_warmup"], "b_left.json.with_warmup")
    c_wo_conf, c_wo_acc, _ = extract_reliability(c_left_data["without_warmup"], "c_left.json.without_warmup")
    c_w_conf, c_w_acc, _ = extract_reliability(c_left_data["with_warmup"], "c_left.json.with_warmup")

    require_keys(b_right_data, ["without_warmup", "with_warmup"], "b_right.json")
    require_keys(c_right_data, ["without_warmup", "with_warmup"], "c_right.json")
    require_keys(b_right_data["without_warmup"], ["ece_list"], "b_right.json.without_warmup")
    require_keys(b_right_data["with_warmup"], ["ece_list"], "b_right.json.with_warmup")
    require_keys(c_right_data["without_warmup"], ["ece_list"], "c_right.json.without_warmup")
    require_keys(c_right_data["with_warmup"], ["ece_list"], "c_right.json.with_warmup")
    b_wo_ece = as_1d_float_array(b_right_data["without_warmup"]["ece_list"], "b_right.json.without_warmup.ece_list")
    b_w_ece = as_1d_float_array(b_right_data["with_warmup"]["ece_list"], "b_right.json.with_warmup.ece_list")
    c_wo_ece = as_1d_float_array(c_right_data["without_warmup"]["ece_list"], "c_right.json.without_warmup.ece_list")
    c_w_ece = as_1d_float_array(c_right_data["with_warmup"]["ece_list"], "c_right.json.with_warmup.ece_list")
    if b_wo_ece.size != b_w_ece.size:
        raise ValueError("b_right.json: with/without ece_list length mismatch.")
    if c_wo_ece.size != c_w_ece.size:
        raise ValueError("c_right.json: with/without ece_list length mismatch.")

    require_keys(d_f_data, ["without_warmup", "with_warmup"], "d-f.json")
    wo_df = d_f_data["without_warmup"]
    w_df = d_f_data["with_warmup"]
    if not isinstance(wo_df, dict) or not isinstance(w_df, dict):
        raise TypeError("d-f.json with/without_warmup must be objects.")
    require_keys(wo_df, ["test_conf", "test_acc"], "d-f.json.without_warmup")
    require_keys(w_df, ["test_conf", "test_acc", "warmup"], "d-f.json.with_warmup")
    if not isinstance(w_df["warmup"], dict):
        raise TypeError("d-f.json.with_warmup.warmup must be object.")
    require_keys(w_df["warmup"], ["test_conf", "test_acc"], "d-f.json.with_warmup.warmup")

    d_wo_conf = as_1d_float_array(wo_df["test_conf"], "d-f.json.without_warmup.test_conf")
    d_wo_acc = as_1d_float_array(wo_df["test_acc"], "d-f.json.without_warmup.test_acc")
    d_w_conf = as_1d_float_array(w_df["test_conf"], "d-f.json.with_warmup.test_conf")
    d_w_acc = as_1d_float_array(w_df["test_acc"], "d-f.json.with_warmup.test_acc")
    d_warmup_conf = as_1d_float_array(w_df["warmup"]["test_conf"], "d-f.json.with_warmup.warmup.test_conf")
    d_warmup_acc = as_1d_float_array(w_df["warmup"]["test_acc"], "d-f.json.with_warmup.warmup.test_acc")
    if d_wo_conf.size != d_wo_acc.size:
        raise ValueError("d-f.json: without_warmup test_conf/test_acc length mismatch.")
    if d_w_conf.size != d_w_acc.size:
        raise ValueError("d-f.json: with_warmup test_conf/test_acc length mismatch.")
    if d_warmup_conf.size != d_warmup_acc.size:
        raise ValueError("d-f.json: with_warmup.warmup test_conf/test_acc length mismatch.")

    return {
        "a_wo_loss": a_wo_loss,
        "a_wo_acc": a_wo_acc,
        "a_w_loss": a_w_loss,
        "a_w_acc": a_w_acc,
        "a_warmup_loss": a_warmup_loss,
        "a_warmup_acc": a_warmup_acc,
        "a_wo_acc025_epoch": a_wo_acc025_epoch,
        "a_wo_final_epoch": a_wo_final_epoch,
        "a_w_acc025_epoch": a_w_acc025_epoch,
        "a_w_final_epoch": a_w_final_epoch,
        "b_wo_conf": b_wo_conf,
        "b_wo_acc": b_wo_acc,
        "b_w_conf": b_w_conf,
        "b_w_acc": b_w_acc,
        "b_wo_ece": b_wo_ece,
        "b_w_ece": b_w_ece,
        "c_wo_conf": c_wo_conf,
        "c_wo_acc": c_wo_acc,
        "c_w_conf": c_w_conf,
        "c_w_acc": c_w_acc,
        "c_wo_ece": c_wo_ece,
        "c_w_ece": c_w_ece,
        "d_wo_conf": d_wo_conf,
        "d_wo_acc": d_wo_acc,
        "d_w_conf": d_w_conf,
        "d_w_acc": d_w_acc,
        "d_warmup_conf": d_warmup_conf,
        "d_warmup_acc": d_warmup_acc,
    }


def plot_a(
    wo_loss: np.ndarray,
    wo_acc: np.ndarray,
    w_loss: np.ndarray,
    w_acc: np.ndarray,
    warmup_loss: np.ndarray,
    warmup_acc: np.ndarray,
    wo_acc025_epoch: int,
    wo_final_epoch: int,
    w_acc025_epoch: int,
    w_final_epoch: int,
    save_path: Path,
) -> None:
    plt.figure(figsize=(to_mm(35), to_mm(35)))

    plt.scatter(wo_loss, wo_acc, color=c.color.ORANGE, s=PANEL_A_D_DOT_SIZE)
    plt.scatter(w_loss, w_acc, color=c.color.SKY, s=PANEL_A_D_DOT_SIZE)
    plt.scatter(warmup_loss, warmup_acc, color=c.color.SKY, s=PANEL_A_D_DOT_SIZE)

    plt.axhline(y=0.25, color="black", linestyle="--", linewidth=0.5)
    plt.axhline(y=0.1, color="black", linestyle="--", linewidth=0.5)

    if 0 <= wo_acc025_epoch < wo_loss.size and 0 <= wo_acc025_epoch < wo_acc.size:
        plt.scatter(
            wo_loss[wo_acc025_epoch],
            wo_acc[wo_acc025_epoch],
            marker="s",
            s=10,
            color="w",
            edgecolors=c.color.ORANGE,
            zorder=4,
        )
    if 0 <= wo_final_epoch < wo_loss.size and 0 <= wo_final_epoch < wo_acc.size:
        plt.scatter(
            wo_loss[wo_final_epoch],
            wo_acc[wo_final_epoch],
            marker="o",
            s=10,
            color="w",
            edgecolors=c.color.ORANGE,
            zorder=4,
        )
    if 0 <= w_acc025_epoch < w_loss.size and 0 <= w_acc025_epoch < w_acc.size:
        plt.scatter(
            w_loss[w_acc025_epoch],
            w_acc[w_acc025_epoch],
            marker="s",
            s=10,
            color="w",
            edgecolors=c.color.SKY,
            zorder=4,
        )
    if 0 <= w_final_epoch < w_loss.size and 0 <= w_final_epoch < w_acc.size:
        plt.scatter(
            w_loss[w_final_epoch],
            w_acc[w_final_epoch],
            marker="o",
            s=10,
            color="w",
            edgecolors=c.color.SKY,
            zorder=4,
        )

    plt.xlabel("Test loss")
    plt.ylabel("Test accuracy")
    plt.ylim(0.02, 0.35)
    plt.xticks([2, 3, 4])
    plt.yticks([0.1, 0.2, 0.3])
    plt.savefig(save_path)
    plt.close()


def plot_reliability_compare(
    conf_wo: np.ndarray,
    acc_wo: np.ndarray,
    conf_w: np.ndarray,
    acc_w: np.ndarray,
    save_path: Path,
    width_mm: int,
    height_mm: int,
    wo_color: str,
    w_color: str,
    bar_alpha: float,
    with_bar_zorder: float = 2.0,
    without_bar_zorder: float = 3.0,
    ref_line_zorder: float = 4.0,
) -> None:
    plt.figure(figsize=(to_mm(width_mm), to_mm(height_mm)))

    wo_bin_width = 1.0 / conf_wo.size
    w_bin_width = 1.0 / conf_w.size

    plt.bar(conf_w, acc_w, width=w_bin_width, color=w_color, alpha=bar_alpha, zorder=with_bar_zorder)
    plt.bar(conf_wo, acc_wo, width=wo_bin_width, color=wo_color, alpha=bar_alpha, zorder=without_bar_zorder)
    plt.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=0.5, zorder=ref_line_zorder)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.savefig(save_path)
    plt.close()


def plot_b_left(conf_wo: np.ndarray, acc_wo: np.ndarray, conf_w: np.ndarray, acc_w: np.ndarray, save_path: Path) -> None:
    plot_reliability_compare(
        conf_wo=conf_wo,
        acc_wo=acc_wo,
        conf_w=conf_w,
        acc_w=acc_w,
        save_path=save_path,
        width_mm=17,
        height_mm=17,
        wo_color=c.color.ORANGE,
        w_color=c.color.SKY,
        bar_alpha=0.5,
    )


def plot_c_left(conf_wo: np.ndarray, acc_wo: np.ndarray, conf_w: np.ndarray, acc_w: np.ndarray, save_path: Path) -> None:
    plot_reliability_compare(
        conf_wo=conf_wo,
        acc_wo=acc_wo,
        conf_w=conf_w,
        acc_w=acc_w,
        save_path=save_path,
        width_mm=17,
        height_mm=17,
        wo_color=COLOR_WITHOUT,
        w_color=COLOR_WITH,
        bar_alpha=1,
        with_bar_zorder=1.0,
        without_bar_zorder=1.1,
        ref_line_zorder=2.0,
    )


def plot_ece_inset(
    ece_wo: np.ndarray,
    ece_w: np.ndarray,
    save_path: Path,
) -> dict[str, Any]:
    wo_pct = 100.0 * ece_wo
    w_pct = 100.0 * ece_w

    plt.figure(figsize=(to_mm(7), to_mm(17)))

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
    plt.yticks([0, 20])
    plt.ylim(0, 20)
    plt.ylabel("Calibration error (%)")
    plt.savefig(save_path)
    plt.close()

    test_result = wilcoxon_signed_rank(
        ece_wo,
        ece_w,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="approx",
    )
    return {
        "test_result": test_result,
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


def plot_b_right(ece_wo: np.ndarray, ece_w: np.ndarray, save_path: Path) -> dict[str, Any]:
    return plot_ece_inset(ece_wo, ece_w, save_path)


def plot_c_right(ece_wo: np.ndarray, ece_w: np.ndarray, save_path: Path) -> dict[str, Any]:
    return plot_ece_inset(ece_wo, ece_w, save_path)


def plot_d(
    conf_wo: np.ndarray,
    acc_wo: np.ndarray,
    conf_w: np.ndarray,
    acc_w: np.ndarray,
    warmup_conf: np.ndarray,
    warmup_acc: np.ndarray,
    save_path: Path,
) -> None:
    plt.figure(figsize=(to_mm(35), to_mm(35)))

    plt.scatter(conf_wo, acc_wo, color=c.color.ORANGE, s=PANEL_A_D_DOT_SIZE)
    plt.scatter(conf_w, acc_w, color=c.color.SKY, s=PANEL_A_D_DOT_SIZE)
    plt.scatter(warmup_conf, warmup_acc, color=c.color.SKY, s=PANEL_A_D_DOT_SIZE)

    plt.plot([0, 0.5], [0, 0.5], color="k", linestyle="--", linewidth=0.5)
    plt.axhline(y=0.1, color="k", linestyle="--", linewidth=0.5)

    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel("Averaged confidence")
    plt.ylabel("Test accuracy")
    plt.savefig(save_path)
    plt.close()


def plot_conf_acc_line_single(
    conf: np.ndarray,
    acc: np.ndarray,
    save_path: Path,
    color: str,
    warmup_epochs: int,
    reserve_warmup_window: bool,
    warmup_conf: np.ndarray | None = None,
    warmup_acc: np.ndarray | None = None,
) -> None:
    plt.figure(figsize=(to_mm(21), to_mm(16)))

    if reserve_warmup_window or warmup_epochs > 0:
        x_offset = int(max(0, warmup_epochs))
    else:
        x_offset = 0

    x_main = x_offset + np.arange(conf.size)
    plt.plot(x_main, conf, color=color, linewidth=1.5)
    plt.plot(x_main, acc, color=c.color.DARKGRAY, linewidth=1.5)

    if warmup_conf is not None and warmup_acc is not None and warmup_conf.size > 0 and warmup_acc.size > 0:
        x_warmup = np.arange(warmup_conf.size)
        plt.plot(x_warmup, warmup_conf, color=color, linewidth=1.5)
        plt.plot(x_warmup, warmup_acc, color=c.color.DARKGRAY, linewidth=1.5)

    total_end = int(x_main[-1]) if x_main.size > 0 else 1
    if warmup_conf is not None and warmup_conf.size > 0:
        total_end = max(total_end, int(warmup_conf.size - 1))
    total_end = max(total_end, 1)

    if warmup_epochs > 0:
        plt.axvline(x=warmup_epochs, color="k", linestyle="--", linewidth=0.5)

    ticks = _build_xticks(total_end, warmup_epochs, include_boundary=warmup_epochs > 0)
    plt.xticks(ticks, _build_xtick_labels(ticks, warmup_epochs))
    plt.xlabel("Epoch")
    plt.xlim(0, total_end)

    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_ticklabels = ["0", "", "", "", "", "0.5"]
    plt.ylim(0, 0.5)
    plt.yticks(y_ticks, y_ticklabels)
    plt.savefig(save_path)
    plt.close()


def plot_e(
    conf_w: np.ndarray,
    acc_w: np.ndarray,
    warmup_conf: np.ndarray,
    warmup_acc: np.ndarray,
    save_path: Path,
) -> None:
    warmup_epochs = max(0, int(warmup_conf.size - 1))
    plot_conf_acc_line_single(
        conf=conf_w,
        acc=acc_w,
        save_path=save_path,
        color=c.color.SKY,
        warmup_epochs=warmup_epochs,
        reserve_warmup_window=False,
        warmup_conf=warmup_conf,
        warmup_acc=warmup_acc,
    )


def plot_f(conf_wo: np.ndarray, acc_wo: np.ndarray, warmup_epochs: int, save_path: Path) -> None:
    plot_conf_acc_line_single(
        conf=conf_wo,
        acc=acc_wo,
        save_path=save_path,
        color=c.color.ORANGE,
        warmup_epochs=warmup_epochs,
        reserve_warmup_window=warmup_epochs > 0,
    )


def write_stats(payload: dict[str, Any]) -> None:
    with OUT_STATS.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    data = load_data()

    plot_a(
        data["a_wo_loss"],
        data["a_wo_acc"],
        data["a_w_loss"],
        data["a_w_acc"],
        data["a_warmup_loss"],
        data["a_warmup_acc"],
        data["a_wo_acc025_epoch"],
        data["a_wo_final_epoch"],
        data["a_w_acc025_epoch"],
        data["a_w_final_epoch"],
        OUT_A,
    )
    plot_b_left(data["b_wo_conf"], data["b_wo_acc"], data["b_w_conf"], data["b_w_acc"], OUT_B_LEFT)
    b_right_info = plot_b_right(data["b_wo_ece"], data["b_w_ece"], OUT_B_RIGHT)
    plot_c_left(data["c_wo_conf"], data["c_wo_acc"], data["c_w_conf"], data["c_w_acc"], OUT_C_LEFT)
    c_right_info = plot_c_right(data["c_wo_ece"], data["c_w_ece"], OUT_C_RIGHT)
    plot_d(
        data["d_wo_conf"],
        data["d_wo_acc"],
        data["d_w_conf"],
        data["d_w_acc"],
        data["d_warmup_conf"],
        data["d_warmup_acc"],
        OUT_D,
    )

    warmup_epochs = max(0, int(data["d_warmup_conf"].size - 1))
    plot_f(data["d_wo_conf"], data["d_wo_acc"], warmup_epochs, OUT_E)
    plot_e(data["d_w_conf"], data["d_w_acc"], data["d_warmup_conf"], data["d_warmup_acc"], OUT_F)

    stats_payload = {
        "b_right_wilcoxon_signed_rank": b_right_info["test_result"],
        "c_right_wilcoxon_signed_rank": c_right_info["test_result"],
        "summary": {
            "b_right": b_right_info["summary"],
            "c_right": c_right_info["summary"],
        },
        "config": {
            "input_files": {
                "a": str(INPUT_A),
                "b_left": str(INPUT_B_LEFT),
                "b_right": str(INPUT_B_RIGHT),
                "c_left": str(INPUT_C_LEFT),
                "c_right": str(INPUT_C_RIGHT),
                "d_f": str(INPUT_D_F),
            },
            "output_files": {
                "a": str(OUT_A),
                "b_left": str(OUT_B_LEFT),
                "b_right": str(OUT_B_RIGHT),
                "c_left": str(OUT_C_LEFT),
                "c_right": str(OUT_C_RIGHT),
                "d": str(OUT_D),
                "e": str(OUT_E),
                "f": str(OUT_F),
                "stats": str(OUT_STATS),
            },
            "tests": {
                "b_right": b_right_info["test_config"],
                "c_right": c_right_info["test_config"],
            },
        },
    }
    write_stats(stats_payload)

    print(f"Saved: {OUT_A}")
    print(f"Saved: {OUT_B_LEFT}")
    print(f"Saved: {OUT_B_RIGHT}")
    print(f"Saved: {OUT_C_LEFT}")
    print(f"Saved: {OUT_C_RIGHT}")
    print(f"Saved: {OUT_D}")
    print(f"Saved: {OUT_E}")
    print(f"Saved: {OUT_F}")
    print(f"Saved: {OUT_STATS}")


if __name__ == "__main__":
    main()
