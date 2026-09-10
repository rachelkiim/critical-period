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
# python3 source_data/fig.1/script.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom.figure import to_mm

INPUT_C = Path(__file__).resolve().parent / "c.json"
INPUT_D_INSET = Path(__file__).resolve().parent / "d_inset.json"
INPUT_D = Path(__file__).resolve().parent / "d.json"

OUT_C = Path(__file__).resolve().parent / "c.svg"
OUT_D_INSET_1 = Path(__file__).resolve().parent / "d_inset_1.svg"
OUT_D_INSET_2 = Path(__file__).resolve().parent / "d_inset_2.svg"
OUT_D_INSET_3 = Path(__file__).resolve().parent / "d_inset_3.svg"
OUT_D_INSET_4 = Path(__file__).resolve().parent / "d_inset_4.svg"
OUT_D = Path(__file__).resolve().parent / "d.svg"
OUT_STATS = Path(__file__).resolve().parent / "stats.json"

WITHOUT_COLOR = "#EFB1A3"
IDEAL_COLOR = "#C1C1C1"

INSET_1_RUN = "d2_nd4000"
INSET_2_RUN = "d6_nd4000"
INSET_3_RUN = "d6_nd32000"
INSET_4_RUN = "d6_nd500"

EXPECTED_DEPTHS = [2, 3, 4, 5, 6]
EXPECTED_NUM_IMAGES = [500, 1000, 2000, 4000, 8000, 16000, 32000]


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


def extract_without_reliability(entry: dict[str, Any], context: str) -> tuple[np.ndarray, np.ndarray]:
    require_keys(entry, ["without_warmup"], context)
    wo = entry["without_warmup"]
    if not isinstance(wo, dict):
        raise TypeError(f"{context}.without_warmup must be object.")

    require_keys(wo, ["conf_bin", "acc", "num_sample"], f"{context}.without_warmup")
    conf_bin = as_1d_float_array(wo["conf_bin"], f"{context}.without_warmup.conf_bin")
    acc = as_1d_float_array(wo["acc"], f"{context}.without_warmup.acc")
    num_sample = as_1d_float_array(wo["num_sample"], f"{context}.without_warmup.num_sample")

    if conf_bin.size != acc.size or conf_bin.size != num_sample.size:
        raise ValueError(f"{context}.without_warmup length mismatch: conf_bin/acc/num_sample")

    return conf_bin, acc


def extract_inset_reliability(d_inset_data: dict[str, Any], run_key: str) -> tuple[np.ndarray, np.ndarray]:
    if run_key not in d_inset_data:
        raise KeyError(f"Missing key in d_inset.json: {run_key}")
    if not isinstance(d_inset_data[run_key], dict):
        raise TypeError(f"d_inset.json[{run_key}] must be object.")
    return extract_without_reliability(d_inset_data[run_key], f"d_inset.json[{run_key}]")


def build_heatmap_matrix(d_data: dict[str, Any]) -> tuple[np.ndarray, list[int], list[int]]:
    depth_to_idx = {depth: i for i, depth in enumerate(EXPECTED_DEPTHS)}
    num_img_to_idx = {num_image: i for i, num_image in enumerate(EXPECTED_NUM_IMAGES)}
    matrix = np.full((len(EXPECTED_NUM_IMAGES), len(EXPECTED_DEPTHS)), np.nan, dtype=float)

    for key, value in d_data.items():
        if key == "config":
            continue
        if not isinstance(value, dict):
            raise TypeError(f"d.json[{key}] must be object.")

        require_keys(value, ["depth", "num_image", "without_warmup"], f"d.json[{key}]")
        depth = value["depth"]
        num_image = value["num_image"]
        if depth not in depth_to_idx:
            raise ValueError(f"d.json[{key}].depth={depth} not in expected depths {EXPECTED_DEPTHS}")
        if num_image not in num_img_to_idx:
            raise ValueError(
                f"d.json[{key}].num_image={num_image} not in expected num_images {EXPECTED_NUM_IMAGES}"
            )

        wo = value["without_warmup"]
        if not isinstance(wo, dict):
            raise TypeError(f"d.json[{key}].without_warmup must be object.")
        require_keys(wo, ["ece_list"], f"d.json[{key}].without_warmup")
        ece_arr = as_1d_float_array(wo["ece_list"], f"d.json[{key}].without_warmup.ece_list")

        row = num_img_to_idx[num_image]
        col = depth_to_idx[depth]
        if not np.isnan(matrix[row, col]):
            raise ValueError(f"Duplicate depth/num_image cell in d.json for depth={depth}, num_image={num_image}")
        matrix[row, col] = float(np.mean(ece_arr))

    if np.isnan(matrix).any():
        raise ValueError("d.json is missing at least one depth/num_image combination for heatmap.")

    return matrix, EXPECTED_DEPTHS, EXPECTED_NUM_IMAGES


def load_data() -> dict[str, Any]:
    c_data = load_json(INPUT_C)
    d_inset_data = load_json(INPUT_D_INSET)
    d_data = load_json(INPUT_D)

    c_conf_bin, c_acc = extract_without_reliability(c_data, "c.json")
    inset_1_conf_bin, inset_1_acc = extract_inset_reliability(d_inset_data, INSET_1_RUN)
    inset_2_conf_bin, inset_2_acc = extract_inset_reliability(d_inset_data, INSET_2_RUN)
    inset_3_conf_bin, inset_3_acc = extract_inset_reliability(d_inset_data, INSET_3_RUN)
    inset_4_conf_bin, inset_4_acc = extract_inset_reliability(d_inset_data, INSET_4_RUN)

    heatmap_matrix, depths, num_images = build_heatmap_matrix(d_data)

    return {
        "c_conf_bin": c_conf_bin,
        "c_acc": c_acc,
        "inset_1_conf_bin": inset_1_conf_bin,
        "inset_1_acc": inset_1_acc,
        "inset_2_conf_bin": inset_2_conf_bin,
        "inset_2_acc": inset_2_acc,
        "inset_3_conf_bin": inset_3_conf_bin,
        "inset_3_acc": inset_3_acc,
        "inset_4_conf_bin": inset_4_conf_bin,
        "inset_4_acc": inset_4_acc,
        "heatmap_matrix": heatmap_matrix,
        "depths": depths,
        "num_images": num_images,
    }


def plot_reliability_common(
    conf_bin: np.ndarray,
    acc: np.ndarray,
    save_path: Path,
    width_mm: int,
    height_mm: int,
    include_ideal: bool,
    xlabel: str,
    ylabel: str,
    xticks: list[float],
    yticks: list[float],
) -> None:
    plt.figure(figsize=(to_mm(width_mm), to_mm(height_mm)))

    num_bin = conf_bin.size
    bin_width = 1.0 / num_bin

    if include_ideal:
        plt.bar(conf_bin, conf_bin, width=bin_width, color=IDEAL_COLOR, zorder=1)

    plt.bar(conf_bin, acc, width=bin_width, color=WITHOUT_COLOR, zorder=2)
    plt.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=0.5, zorder=3)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(save_path)
    plt.close()


def plot_c(conf_bin: np.ndarray, acc: np.ndarray, save_path: Path) -> None:
    plot_reliability_common(
        conf_bin=conf_bin,
        acc=acc,
        save_path=save_path,
        width_mm=30,
        height_mm=30,
        include_ideal=True,
        xlabel="Confidence",
        ylabel="Accuracy",
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
    )


def plot_d_inset_1(conf_bin: np.ndarray, acc: np.ndarray, save_path: Path) -> None:
    plot_reliability_common(
        conf_bin=conf_bin,
        acc=acc,
        save_path=save_path,
        width_mm=9,
        height_mm=9,
        include_ideal=False,
        xlabel="Conf.",
        ylabel="Acc.",
        xticks=[0, 1],
        yticks=[0, 1],
    )


def plot_d_inset_2(conf_bin: np.ndarray, acc: np.ndarray, save_path: Path) -> None:
    plot_reliability_common(
        conf_bin=conf_bin,
        acc=acc,
        save_path=save_path,
        width_mm=9,
        height_mm=9,
        include_ideal=False,
        xlabel="Conf.",
        ylabel="Acc.",
        xticks=[0, 1],
        yticks=[0, 1],
    )


def plot_d_inset_3(conf_bin: np.ndarray, acc: np.ndarray, save_path: Path) -> None:
    plot_reliability_common(
        conf_bin=conf_bin,
        acc=acc,
        save_path=save_path,
        width_mm=9,
        height_mm=9,
        include_ideal=False,
        xlabel="Conf.",
        ylabel="Acc.",
        xticks=[0, 1],
        yticks=[0, 1],
    )


def plot_d_inset_4(conf_bin: np.ndarray, acc: np.ndarray, save_path: Path) -> None:
    plot_reliability_common(
        conf_bin=conf_bin,
        acc=acc,
        save_path=save_path,
        width_mm=9,
        height_mm=9,
        include_ideal=False,
        xlabel="Conf.",
        ylabel="Acc.",
        xticks=[0, 1],
        yticks=[0, 1],
    )


def plot_d(matrix: np.ndarray, depths: list[int], num_images: list[int], save_path: Path) -> None:
    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.imshow(matrix, origin="lower", cmap="coolwarm", vmin=0, vmax=0.25, aspect="auto")
    plt.clim(0, 0.25)
    plt.xlabel("Network depth")
    plt.ylabel("Size of training data")
    ax = plt.gca()
    ax.set_xticks(np.arange(len(depths)), depths)
    ax.set_yticks(np.arange(len(num_images)), [f"{num / 1000:g}" for num in num_images])
    ax.text(-0.05, 1.01, "$×10³$", transform=ax.transAxes, ha="left", va="bottom")
    plt.savefig(save_path)
    plt.close()


def build_stats_payload() -> dict[str, Any]:
    return {
        "config": {
            "input_files": {
                "c": str(INPUT_C),
                "d_inset": str(INPUT_D_INSET),
                "d": str(INPUT_D),
            },
            "output_files": {
                "c": str(OUT_C),
                "d_inset_1": str(OUT_D_INSET_1),
                "d_inset_2": str(OUT_D_INSET_2),
                "d_inset_3": str(OUT_D_INSET_3),
                "d_inset_4": str(OUT_D_INSET_4),
                "d": str(OUT_D),
                "stats": str(OUT_STATS),
            },
            "inset_mapping": {
                "d_inset_1.svg": INSET_1_RUN,
                "d_inset_2.svg": INSET_2_RUN,
                "d_inset_3.svg": INSET_3_RUN,
                "d_inset_4.svg": INSET_4_RUN,
            },
            "heatmap": {
                "aggregation": "mean(without_warmup.ece_list)",
                "cmap": "coolwarm",
                "clim": [0, 0.25],
            },
        }
    }


def write_stats(payload: dict[str, Any], save_path: Path) -> None:
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    data = load_data()

    plot_c(data["c_conf_bin"], data["c_acc"], OUT_C)
    plot_d_inset_1(data["inset_1_conf_bin"], data["inset_1_acc"], OUT_D_INSET_1)
    plot_d_inset_2(data["inset_2_conf_bin"], data["inset_2_acc"], OUT_D_INSET_2)
    plot_d_inset_3(data["inset_3_conf_bin"], data["inset_3_acc"], OUT_D_INSET_3)
    plot_d_inset_4(data["inset_4_conf_bin"], data["inset_4_acc"], OUT_D_INSET_4)
    plot_d(data["heatmap_matrix"], data["depths"], data["num_images"], OUT_D)

    write_stats(build_stats_payload(), OUT_STATS)

    print(f"Saved: {OUT_C}")
    print(f"Saved: {OUT_D_INSET_1}")
    print(f"Saved: {OUT_D_INSET_2}")
    print(f"Saved: {OUT_D_INSET_3}")
    print(f"Saved: {OUT_D_INSET_4}")
    print(f"Saved: {OUT_D}")
    print(f"Saved: {OUT_STATS}")


if __name__ == "__main__":
    main()
