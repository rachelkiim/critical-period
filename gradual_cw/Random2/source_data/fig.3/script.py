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
# python3 source_data/fig.3/script.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom.figure import GraphConfig as c, to_mm
from src.evaluation.stats import mannwhitneyu, wilcoxon_signed_rank

INPUT_BC = Path(__file__).resolve().parent / "b-c.json"
INPUT_D = Path(__file__).resolve().parent / "d.json"
INPUT_EG = Path(__file__).resolve().parent / "e-g.json"
INPUT_HJ = Path(__file__).resolve().parent / "h-j.json"

OUT_B_LEFT = [Path(__file__).resolve().parent / f"b_left_{i}.svg" for i in range(1, 7)]
OUT_B_RIGHT = [Path(__file__).resolve().parent / f"b_right_{i}.svg" for i in range(1, 7)]
OUT_C = [Path(__file__).resolve().parent / f"c_{i}.svg" for i in range(1, 7)]

OUT_D = Path(__file__).resolve().parent / "d.svg"
OUT_E = Path(__file__).resolve().parent / "e.svg"
OUT_F_LEFT = Path(__file__).resolve().parent / "f_left.svg"
OUT_F_RIGHT = Path(__file__).resolve().parent / "f_right.svg"
OUT_G = Path(__file__).resolve().parent / "g.svg"
OUT_H = Path(__file__).resolve().parent / "h.svg"
OUT_I_LEFT = Path(__file__).resolve().parent / "i_left.svg"
OUT_I_RIGHT = Path(__file__).resolve().parent / "i_right.svg"
OUT_J = Path(__file__).resolve().parent / "j.svg"

OUT_STATS = Path(__file__).resolve().parent / "stats.json"

B_DOT_SIZE = 5.0


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


def as_1d_int_array(values: Any, context: str, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=int).ravel()
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{context} must not be empty.")
    return arr


def as_2d_float_array(values: Any, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{context} must be a non-empty 2D numeric array.")
    return arr


def pick_test_xy(bc_data: dict[str, Any], net_key: str) -> tuple[np.ndarray, np.ndarray]:
    if "per_net_test_xy" in bc_data:
        per = bc_data["per_net_test_xy"]
        if isinstance(per, dict) and net_key in per:
            xy = per[net_key]
        else:
            xy = bc_data.get("test_xy")
    else:
        xy = bc_data.get("test_xy")

    if not isinstance(xy, dict):
        raise TypeError(f"test_xy for {net_key} must be object.")
    require_keys(xy, ["x", "y"], f"test_xy[{net_key}]")
    x = as_1d_float_array(xy["x"], f"test_xy[{net_key}].x")
    y = as_1d_float_array(xy["y"], f"test_xy[{net_key}].y")
    if x.size != y.size:
        raise ValueError(f"test_xy[{net_key}]: x/y length mismatch.")
    return x, y


def load_data() -> dict[str, Any]:
    bc = load_json(INPUT_BC)
    d = load_json(INPUT_D)
    eg = load_json(INPUT_EG)
    hj = load_json(INPUT_HJ)

    require_keys(bc, ["pretrained_", "untrained_", "config"], "b-c.json")
    if not isinstance(bc["pretrained_"], dict) or not isinstance(bc["untrained_"], dict):
        raise TypeError("b-c.json pretrained_/untrained_ must be objects.")
    if not isinstance(bc["config"], dict):
        raise TypeError("b-c.json.config must be object.")
    require_keys(bc["config"], ["selected_nets"], "b-c.json.config")
    selected_nets = bc["config"]["selected_nets"]
    if not isinstance(selected_nets, list) or len(selected_nets) != 6:
        raise ValueError("b-c.json.config.selected_nets must be a list of length 6.")

    net_payloads: list[dict[str, Any]] = []
    for net_key in selected_nets:
        if not isinstance(net_key, str):
            raise TypeError("selected_nets entries must be strings.")
        if net_key not in bc["untrained_"] or net_key not in bc["pretrained_"]:
            raise KeyError(f"Missing {net_key} in b-c.json untrained_/pretrained_.")

        u_entry = bc["untrained_"][net_key]
        p_entry = bc["pretrained_"][net_key]
        if not isinstance(u_entry, dict) or not isinstance(p_entry, dict):
            raise TypeError(f"b-c.json[{net_key}] entries must be objects.")

        require_keys(u_entry, ["pred", "confidence"], f"b-c.json.untrained_.{net_key}")
        require_keys(p_entry, ["pred", "confidence"], f"b-c.json.pretrained_.{net_key}")

        x, y = pick_test_xy(bc, net_key)
        u_pred = as_1d_int_array(u_entry["pred"], f"b-c.json.untrained_.{net_key}.pred")
        p_pred = as_1d_int_array(p_entry["pred"], f"b-c.json.pretrained_.{net_key}.pred")
        u_conf = as_1d_float_array(u_entry["confidence"], f"b-c.json.untrained_.{net_key}.confidence")
        p_conf = as_1d_float_array(p_entry["confidence"], f"b-c.json.pretrained_.{net_key}.confidence")

        n = x.size
        if not (u_pred.size == p_pred.size == u_conf.size == p_conf.size == n):
            raise ValueError(f"{net_key}: test_xy/pred/confidence length mismatch.")

        net_payloads.append(
            {
                "net_key": net_key,
                "x": x,
                "y": y,
                "u_pred": u_pred,
                "p_pred": p_pred,
                "u_conf": u_conf,
                "p_conf": p_conf,
            }
        )

    require_keys(d, ["untrained", "pretrained"], "d.json")
    if not isinstance(d["untrained"], dict) or not isinstance(d["pretrained"], dict):
        raise TypeError("d.json untrained/pretrained must be objects.")
    require_keys(d["untrained"], ["pred_std_list"], "d.json.untrained")
    require_keys(d["pretrained"], ["pred_std_list"], "d.json.pretrained")
    d_untrained = as_1d_float_array(d["untrained"]["pred_std_list"], "d.json.untrained.pred_std_list")
    d_pretrained = as_1d_float_array(d["pretrained"]["pred_std_list"], "d.json.pretrained.pred_std_list")
    if d_untrained.size != d_pretrained.size:
        raise ValueError("d.json: untrained/pretrained pred_std_list length mismatch.")

    require_keys(eg, ["untrained"], "e-g.json")
    if not isinstance(eg["untrained"], dict):
        raise TypeError("e-g.json.untrained must be object.")
    require_keys(eg["untrained"], ["logit", "prob", "conf"], "e-g.json.untrained")
    eg_logit = as_2d_float_array(eg["untrained"]["logit"], "e-g.json.untrained.logit")
    eg_prob = as_2d_float_array(eg["untrained"]["prob"], "e-g.json.untrained.prob")
    eg_conf = as_1d_float_array(eg["untrained"]["conf"], "e-g.json.untrained.conf")
    if eg_logit.shape != eg_prob.shape:
        raise ValueError("e-g.json: logit/prob shape mismatch.")
    if eg_logit.shape[0] != eg_conf.size:
        raise ValueError("e-g.json: sample count mismatch among logit/prob/conf.")

    require_keys(hj, ["warmup"], "h-j.json")
    if not isinstance(hj["warmup"], dict):
        raise TypeError("h-j.json.warmup must be object.")
    require_keys(hj["warmup"], ["logit", "prob", "conf"], "h-j.json.warmup")
    hj_logit = as_2d_float_array(hj["warmup"]["logit"], "h-j.json.warmup.logit")
    hj_prob = as_2d_float_array(hj["warmup"]["prob"], "h-j.json.warmup.prob")
    hj_conf = as_1d_float_array(hj["warmup"]["conf"], "h-j.json.warmup.conf")
    if hj_logit.shape != hj_prob.shape:
        raise ValueError("h-j.json: logit/prob shape mismatch.")
    if hj_logit.shape[0] != hj_conf.size:
        raise ValueError("h-j.json: sample count mismatch among logit/prob/conf.")

    return {
        "net_payloads": net_payloads,
        "d_untrained": d_untrained,
        "d_pretrained": d_pretrained,
        "eg_logit": eg_logit,
        "eg_prob": eg_prob,
        "eg_conf": eg_conf,
        "hj_logit": hj_logit,
        "hj_prob": hj_prob,
        "hj_conf": hj_conf,
    }


def plot_input_space(
    x: np.ndarray,
    y: np.ndarray,
    conf: np.ndarray,
    save_path: Path,
    save_axes_png: bool = False,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(to_mm(30), to_mm(30)))
    scatter = ax.scatter(x, y, c=conf, s=B_DOT_SIZE, cmap="coolwarm")
    scatter.set_clim(0.5, 1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    plt.savefig(save_path)

    png_path: Path | None = None
    if save_axes_png:
        fig.canvas.draw()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax.axis("off")
        png_path = save_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches=bbox, dpi=300)

    plt.close(fig)
    return png_path


def plot_confidence_boxplot(untrained_conf: np.ndarray, pretrained_conf: np.ndarray, save_path: Path) -> dict[str, Any]:
    plt.figure(figsize=(to_mm(10), to_mm(30)))
    bp = plt.boxplot([untrained_conf, pretrained_conf], positions=[0, 1], widths=0.5, patch_artist=True)
    bp["boxes"][0].set_facecolor(c.color.DARKGRAY)
    bp["boxes"][1].set_facecolor(c.color.SKY)
    plt.axhline(0.5, color="k", linestyle="--", linewidth=0.5)
    plt.xticks([0, 1], ["U", "R"])
    plt.ylim(0.4, 1.1)
    plt.yticks([0.4, 0.6, 0.8, 1.0])
    plt.ylabel("Confidence")
    plt.savefig(save_path)
    plt.close()

    test_result = mannwhitneyu(untrained_conf, pretrained_conf, alternative="two-sided", method="asymptotic")
    return {
        "test_result": test_result,
        "summary": {
            "untrained": {
                "mean": float(np.mean(untrained_conf)),
                "median": float(np.median(untrained_conf)),
                "std": float(np.std(untrained_conf)),
            },
            "pretrained": {
                "mean": float(np.mean(pretrained_conf)),
                "median": float(np.median(pretrained_conf)),
                "std": float(np.std(pretrained_conf)),
            },
        },
    }


def plot_d_class_bias(untrained_std: np.ndarray, pretrained_std: np.ndarray, save_path: Path) -> dict[str, Any]:
    plt.figure(figsize=(to_mm(10), to_mm(30)))

    rng = np.random.default_rng(0)
    for pos, values in zip([0, 1], [untrained_std, pretrained_std], strict=True):
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

    bp = plt.boxplot([untrained_std, pretrained_std], positions=[0, 1], widths=0.5, patch_artist=True, zorder=2)
    bp["boxes"][0].set_facecolor(c.color.DARKGRAY)
    bp["boxes"][1].set_facecolor(c.color.SKY)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_alpha(0.7)
    plt.xticks([0, 1], ["U", "R"])
    plt.ylabel("Std. of class ratio")
    plt.ylim(0, 0.6)
    plt.yticks([0.0, 0.2, 0.4, 0.6])
    plt.savefig(save_path)
    plt.close()

    test_result = wilcoxon_signed_rank(
        untrained_std,
        pretrained_std,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="approx",
    )
    return {
        "test_result": test_result,
        "summary": {
            "untrained": {
                "mean": float(np.mean(untrained_std)),
                "median": float(np.median(untrained_std)),
                "std": float(np.std(untrained_std)),
            },
            "pretrained": {
                "mean": float(np.mean(pretrained_std)),
                "median": float(np.median(pretrained_std)),
                "std": float(np.std(pretrained_std)),
            },
        },
    }


def plot_logit_hist(logit: np.ndarray, color: str, save_path: Path) -> None:
    plt.figure(figsize=(to_mm(30), to_mm(10)))
    plt.hist(logit.flatten(), bins=60, density=True, alpha=0.5, color=color)
    plt.xlim(-10, 10)
    plt.xticks([-10, -5, 0, 5, 10])
    plt.ylim(0, 0.4)
    plt.yticks([0, 0.4])
    plt.xlabel("Logit")
    plt.ylabel("Density")
    plt.savefig(save_path)
    plt.close()


def plot_prob_vs_logit(logit: np.ndarray, prob: np.ndarray, color: str, save_path: Path, num_points: int = 2000) -> None:
    n = min(num_points, logit.size, prob.size)
    logit_flat = logit.flatten()[:n]
    prob_flat = prob.flatten()[:n]
    x_line = np.linspace(-10, 10, 100)
    y_sigmoid = 1.0 / (1.0 + np.exp(-x_line))

    plt.figure(figsize=(to_mm(30), to_mm(30)))
    plt.scatter(logit_flat, prob_flat, s=1, alpha=0.5, c=color)
    plt.plot(x_line, y_sigmoid, color="k", linewidth=0.5, linestyle="--")
    plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.xticks([-10, -5, 0, 5, 10])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("Logit")
    plt.ylabel("Probability")
    plt.savefig(save_path)
    plt.close()


def plot_prob_hist(
    prob: np.ndarray,
    color: str,
    save_path: Path,
    xlim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
) -> None:
    plt.figure(figsize=(to_mm(10), to_mm(30)))
    plt.hist(prob.flatten(), bins=60, range=(0, 1), density=True, alpha=0.5, color=color, orientation="horizontal")
    plt.axhline(0.5, color="k", linestyle="--", linewidth=0.5)
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if xlim is not None:
        plt.xlim(*xlim)
    if xticks is not None:
        plt.xticks(xticks)
    plt.xlabel("Density")
    plt.ylabel("Probability")
    plt.savefig(save_path)
    plt.close()


def plot_conf_hist(
    conf: np.ndarray,
    color: str,
    save_path: Path,
    xlim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
) -> None:
    plt.figure(figsize=(to_mm(15), to_mm(30)))
    plt.hist(conf, bins=30, range=(0.5, 1), density=True, alpha=0.5, color=color, orientation="horizontal")
    plt.ylim(0.5, 1)
    plt.yticks([0.5, 0.75, 1])
    if xlim is not None:
        plt.xlim(*xlim)
    if xticks is not None:
        plt.xticks(xticks)
    plt.xlabel("Density")
    plt.ylabel("Confidence")
    plt.savefig(save_path)
    plt.close()


def write_stats(stats_payload: dict[str, Any]) -> None:
    with OUT_STATS.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)
        f.write("\n")


def main() -> None:
    data = load_data()

    c_tests: dict[str, Any] = {}
    c_summaries: dict[str, Any] = {}
    b_png_paths: list[Path] = []

    for i, payload in enumerate(data["net_payloads"], start=1):
        left_png = plot_input_space(payload["x"], payload["y"], payload["u_conf"], OUT_B_LEFT[i - 1], save_axes_png=True)
        right_png = plot_input_space(payload["x"], payload["y"], payload["p_conf"], OUT_B_RIGHT[i - 1], save_axes_png=True)
        if left_png is not None:
            b_png_paths.append(left_png)
        if right_png is not None:
            b_png_paths.append(right_png)

        c_info = plot_confidence_boxplot(payload["u_conf"], payload["p_conf"], OUT_C[i - 1])
        c_tests[f"c_{i}"] = c_info["test_result"]
        c_summaries[f"c_{i}"] = {
            "net_key": payload["net_key"],
            **c_info["summary"],
        }

    d_info = plot_d_class_bias(data["d_untrained"], data["d_pretrained"], OUT_D)

    plot_logit_hist(data["eg_logit"], c.color.DARKGRAY, OUT_E)
    plot_prob_vs_logit(data["eg_logit"], data["eg_prob"], c.color.DARKGRAY, OUT_F_LEFT)
    plot_prob_hist(data["eg_prob"], c.color.DARKGRAY, OUT_F_RIGHT, xlim=(0, 3), xticks=[0, 3])
    plot_conf_hist(data["eg_conf"], c.color.DARKGRAY, OUT_G, xlim=(0, 5), xticks=[0, 2.5, 5])

    plot_logit_hist(data["hj_logit"], c.color.SKY, OUT_H)
    plot_prob_vs_logit(data["hj_logit"], data["hj_prob"], c.color.SKY, OUT_I_LEFT)
    plot_prob_hist(data["hj_prob"], c.color.SKY, OUT_I_RIGHT, xlim=(0, 10), xticks=[0, 10])
    plot_conf_hist(data["hj_conf"], c.color.SKY, OUT_J, xlim=(0, 10), xticks=[0, 10, 20])

    stats_payload = {
        "c_mannwhitneyu_by_panel": c_tests,
        "d_wilcoxon_signed_rank": d_info["test_result"],
        "summary": {
            "c": c_summaries,
            "d": d_info["summary"],
        },
        "config": {
            "input_files": {
                "b_c": str(INPUT_BC),
                "d": str(INPUT_D),
                "e_g": str(INPUT_EG),
                "h_j": str(INPUT_HJ),
            },
            "output_files": {
                **{f"b_left_{i}": str(p) for i, p in enumerate(OUT_B_LEFT, start=1)},
                **{f"b_right_{i}": str(p) for i, p in enumerate(OUT_B_RIGHT, start=1)},
                **{f"c_{i}": str(p) for i, p in enumerate(OUT_C, start=1)},
                "d": str(OUT_D),
                "e": str(OUT_E),
                "f_left": str(OUT_F_LEFT),
                "f_right": str(OUT_F_RIGHT),
                "g": str(OUT_G),
                "h": str(OUT_H),
                "i_left": str(OUT_I_LEFT),
                "i_right": str(OUT_I_RIGHT),
                "j": str(OUT_J),
                "stats": str(OUT_STATS),
            },
            "tests": {
                "c": {
                    "test": "mannwhitneyu",
                    "alternative": "two-sided",
                    "method": "asymptotic",
                    "num_tests": 6,
                },
                "d": {
                    "test": "wilcoxon_signed_rank",
                    "alternative": "two-sided",
                    "zero_method": "wilcox",
                    "correction": False,
                    "method": "approx",
                },
            },
            "selected_net_order": [p["net_key"] for p in data["net_payloads"]],
            "colors": {
                "untrained": c.color.DARKGRAY,
                "random_trained": c.color.SKY,
                "input_space_cmap": "coolwarm",
            },
        },
    }
    write_stats(stats_payload)

    for p in OUT_B_LEFT + OUT_B_RIGHT + OUT_C:
        print(f"Saved: {p}")
    for p in b_png_paths:
        print(f"Saved: {p}")
    print(f"Saved: {OUT_D}")
    print(f"Saved: {OUT_E}")
    print(f"Saved: {OUT_F_LEFT}")
    print(f"Saved: {OUT_F_RIGHT}")
    print(f"Saved: {OUT_G}")
    print(f"Saved: {OUT_H}")
    print(f"Saved: {OUT_I_LEFT}")
    print(f"Saved: {OUT_I_RIGHT}")
    print(f"Saved: {OUT_J}")
    print(f"Saved: {OUT_STATS}")


if __name__ == "__main__":
    main()
