from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats as scipy_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure `src` package import works when running:
# python3 source_data/fig.5/script.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom.figure import GraphConfig as c, to_mm
from src.evaluation.stats import mannwhitneyu, wilcoxon_signed_rank

INPUT_B = Path(__file__).resolve().parent / "b.json"
INPUT_D_LEFT = Path(__file__).resolve().parent / "d_left.json"
INPUT_D_RIGHT = Path(__file__).resolve().parent / "d_right.json"

OUT_B_LEFT = Path(__file__).resolve().parent / "b_left.svg"
OUT_B_RIGHT = Path(__file__).resolve().parent / "b_right.svg"
OUT_D_LEFT = Path(__file__).resolve().parent / "d_left.svg"
OUT_D_RIGHT = Path(__file__).resolve().parent / "d_right.svg"
OUT_STATS = Path(__file__).resolve().parent / "stats.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_keys(obj: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise KeyError(f"Missing keys in {context}: {missing}")


def as_1d_float_array(values: Any, context: str, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{context} must not be empty.")
    return arr


def load_data() -> dict[str, np.ndarray]:
    b_data = load_json(INPUT_B)
    d_left_data = load_json(INPUT_D_LEFT)
    d_right_data = load_json(INPUT_D_RIGHT)

    require_keys(b_data, ["with_warmup", "without_warmup"], "b.json")
    require_keys(d_left_data, ["with_warmup", "without_warmup"], "d_left.json")
    require_keys(d_right_data, ["with_warmup", "without_warmup"], "d_right.json")

    require_keys(b_data["with_warmup"], ["id_confidence", "ood_confidence"], "b.json.with_warmup")
    require_keys(
        b_data["without_warmup"],
        ["id_confidence", "ood_confidence"],
        "b.json.without_warmup",
    )

    require_keys(d_left_data["with_warmup"], ["fp", "tp"], "d_left.json.with_warmup")
    require_keys(d_left_data["without_warmup"], ["fp", "tp"], "d_left.json.without_warmup")

    require_keys(d_right_data["with_warmup"], ["auroc_list"], "d_right.json.with_warmup")
    require_keys(d_right_data["without_warmup"], ["auroc_list"], "d_right.json.without_warmup")

    b_wo_ood = as_1d_float_array(
        b_data["without_warmup"]["ood_confidence"], "b.json.without_warmup.ood_confidence"
    )
    b_w_ood = as_1d_float_array(
        b_data["with_warmup"]["ood_confidence"], "b.json.with_warmup.ood_confidence"
    )

    # Validate presence as requested input schema.
    as_1d_float_array(b_data["without_warmup"]["id_confidence"], "b.json.without_warmup.id_confidence")
    as_1d_float_array(b_data["with_warmup"]["id_confidence"], "b.json.with_warmup.id_confidence")

    d_left_fp_wo = as_1d_float_array(d_left_data["without_warmup"]["fp"], "d_left.json.without_warmup.fp")
    d_left_tp_wo = as_1d_float_array(d_left_data["without_warmup"]["tp"], "d_left.json.without_warmup.tp")
    d_left_fp_w = as_1d_float_array(d_left_data["with_warmup"]["fp"], "d_left.json.with_warmup.fp")
    d_left_tp_w = as_1d_float_array(d_left_data["with_warmup"]["tp"], "d_left.json.with_warmup.tp")

    if d_left_fp_wo.size != d_left_tp_wo.size:
        raise ValueError("d_left.json: without_warmup fp/tp length mismatch.")
    if d_left_fp_w.size != d_left_tp_w.size:
        raise ValueError("d_left.json: with_warmup fp/tp length mismatch.")

    d_right_wo = as_1d_float_array(d_right_data["without_warmup"]["auroc_list"], "d_right.json.without_warmup.auroc_list")
    d_right_w = as_1d_float_array(d_right_data["with_warmup"]["auroc_list"], "d_right.json.with_warmup.auroc_list")
    if d_right_wo.size != d_right_w.size:
        raise ValueError("d_right.json: AUROC list length mismatch for paired Wilcoxon test.")

    return {
        "b_wo_ood": b_wo_ood,
        "b_w_ood": b_w_ood,
        "d_left_fp_wo": d_left_fp_wo,
        "d_left_tp_wo": d_left_tp_wo,
        "d_left_fp_w": d_left_fp_w,
        "d_left_tp_w": d_left_tp_w,
        "d_right_wo": d_right_wo,
        "d_right_w": d_right_w,
    }


def plot_b_left(wo_ood: np.ndarray, w_ood: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.hist(
        wo_ood,
        bins=40,
        range=(0, 1),
        alpha=0.5,
        color=c.color.ORANGE,
        density=True,
        label="w/o warmup",
        orientation="horizontal",
    )
    plt.hist(
        w_ood,
        bins=40,
        range=(0, 1),
        alpha=0.5,
        color=c.color.SKY,
        density=True,
        label="w/ warmup",
        orientation="horizontal",
    )
    plt.axhline(0.1, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel("Density")
    plt.xticks([0, 10, 20])
    plt.ylabel("Confidence")
    plt.yticks([0, 0.3, 0.6])
    plt.ylim(0, 0.6)
    plt.savefig(save_path)
    plt.close()


def plot_b_right(wo_ood: np.ndarray, w_ood: np.ndarray, save_path: Path) -> dict[str, Any]:
    plt.figure(figsize=(to_mm(10), to_mm(30)))
    bp = plt.boxplot([wo_ood, w_ood], positions=[0, 1], widths=0.5, patch_artist=True)
    bp["boxes"][0].set_facecolor(c.color.ORANGE)
    bp["boxes"][1].set_facecolor(c.color.SKY)
    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("Confidence")
    plt.yticks([0, 0.3, 0.6])
    plt.ylim(0, 0.6)
    plt.savefig(save_path)
    plt.close()

    return {
        "test_result": mannwhitneyu(wo_ood, w_ood, alternative="two-sided", method="asymptotic"),
        "summary": {
            "without_warmup": {
                "mean": float(np.mean(wo_ood)),
                "median": float(np.median(wo_ood)),
                "std": float(np.std(wo_ood)),
            },
            "with_warmup": {
                "mean": float(np.mean(w_ood)),
                "median": float(np.median(w_ood)),
                "std": float(np.std(w_ood)),
            },
        },
        "test_config": {
            "test": "mannwhitneyu",
            "alternative": "two-sided",
            "method": "asymptotic",
        },
    }


def plot_d_left(fp_wo: np.ndarray, tp_wo: np.ndarray, fp_w: np.ndarray, tp_w: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.plot(fp_wo, tp_wo, color=c.color.ORANGE, label="w/o warmup")
    plt.plot(fp_w, tp_w, color=c.color.SKY, label="w/ warmup")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.xticks([0, 0.5, 1])
    plt.ylim(0, 1)
    plt.yticks([0, 0.5, 1])
    plt.savefig(save_path)
    plt.close()


def plot_d_right(auroc_wo: np.ndarray, auroc_w: np.ndarray, save_path: Path) -> dict[str, Any]:
    plt.figure(figsize=(to_mm(10), to_mm(30)))

    rng = np.random.default_rng(0)
    for pos, values in zip([0, 1], [auroc_wo, auroc_w], strict=True):
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

    bp = plt.boxplot([auroc_wo, auroc_w], positions=[0, 1], widths=0.5, patch_artist=True, zorder=2)
    bp["boxes"][0].set_facecolor(c.color.ORANGE)
    bp["boxes"][1].set_facecolor(c.color.SKY)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_alpha(0.7)
    plt.xticks([0, 1], ["w/o", "w/"])
    plt.ylabel("AUROC")
    plt.ylim(0.5, 1.05)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.savefig(save_path)
    plt.close()

    d_right_stats = wilcoxon_signed_rank(
        auroc_wo,
        auroc_w,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="approx",
    )

    # ranksum_z = float(scipy_stats.ranksums(auroc_wo, auroc_w).statistic)
    # d_right_stats["r"] = float(ranksum_z / np.sqrt(auroc_wo.size))

    return {
        "test_result": d_right_stats,
        "summary": {
            "without_warmup": {
                "mean": float(np.mean(auroc_wo)),
                "median": float(np.median(auroc_wo)),
                "std": float(np.std(auroc_wo)),
            },
            "with_warmup": {
                "mean": float(np.mean(auroc_w)),
                "median": float(np.median(auroc_w)),
                "std": float(np.std(auroc_w)),
            },
        },
        "test_config": {
            "test": "wilcoxon_signed_rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "correction": False,
            "method": "auto",
            "effect_size_r_source": "ranksum",
        },
    }


def main() -> None:
    data = load_data()

    plot_b_left(data["b_wo_ood"], data["b_w_ood"], OUT_B_LEFT)
    b_right_info = plot_b_right(data["b_wo_ood"], data["b_w_ood"], OUT_B_RIGHT)
    plot_d_left(
        data["d_left_fp_wo"],
        data["d_left_tp_wo"],
        data["d_left_fp_w"],
        data["d_left_tp_w"],
        OUT_D_LEFT,
    )
    d_right_info = plot_d_right(data["d_right_wo"], data["d_right_w"], OUT_D_RIGHT)

    stats_payload = {
        "b_right_mannwhitneyu": b_right_info["test_result"],
        "d_right_wilcoxon_signed_rank": d_right_info["test_result"],
        "summary": {
            "b_right": b_right_info["summary"],
            "d_right": d_right_info["summary"],
        },
        "config": {
            "input_files": {
                "b": str(INPUT_B),
                "d_left": str(INPUT_D_LEFT),
                "d_right": str(INPUT_D_RIGHT),
            },
            "output_files": {
                "b_left": str(OUT_B_LEFT),
                "b_right": str(OUT_B_RIGHT),
                "d_left": str(OUT_D_LEFT),
                "d_right": str(OUT_D_RIGHT),
                "stats": str(OUT_STATS),
            },
            "tests": {
                "b_right": b_right_info["test_config"],
                "d_right": d_right_info["test_config"],
            },
            "colors": {
                "without_warmup": c.color.ORANGE,
                "with_warmup": c.color.SKY,
                "aux": c.color.DARKGRAY,
            },
        },
    }

    with OUT_STATS.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    print(f"Saved: {OUT_B_LEFT}")
    print(f"Saved: {OUT_B_RIGHT}")
    print(f"Saved: {OUT_D_LEFT}")
    print(f"Saved: {OUT_D_RIGHT}")
    print(f"Saved: {OUT_STATS}")


if __name__ == "__main__":
    main()
