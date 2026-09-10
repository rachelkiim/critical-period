from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ORDER = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "densenet121",
    "densenet169",
    "densenet201",
    "vit",
]

ECE_METHOD_ORDER = ["msp", "temp_scaling", "vec_scaling", "isotonic_reg"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _as_1d_float_array(values: Any, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{context} must not be empty.")
    return arr


def _format_mean_std(values: Any, scale: float = 1.0, percent: bool = False) -> str:
    arr = _as_1d_float_array(values, "values")
    mean = float(np.mean(arr)) * scale
    std = float(np.std(arr)) * scale
    text = f"({mean:.3f} ± {std:.3f})"
    if percent:
        return f"{text}%"
    return text


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    accuracy_path = base_dir / "accuracy.json"
    ece_path = base_dir / "ece.json"
    out_path = base_dir / "stats.json"

    accuracy_data = load_json(accuracy_path)
    ece_data = load_json(ece_path)

    accuracy_stats: dict[str, Any] = {}
    for model in MODEL_ORDER:
        if model not in accuracy_data:
            raise KeyError(f"Missing model in accuracy.json: {model}")
        entry = accuracy_data[model]
        if not isinstance(entry, dict):
            raise TypeError(f"accuracy.json[{model}] must be object.")
        accuracy_stats[model] = {
            "without_warmup": _format_mean_std(
                entry["without_warmup"], scale=1.0, percent=False
            ),
            "with_warmup": _format_mean_std(
                entry["with_warmup"], scale=1.0, percent=False
            ),
        }

    ece_stats: dict[str, Any] = {}
    for model in MODEL_ORDER:
        if model not in ece_data:
            raise KeyError(f"Missing model in ece.json: {model}")
        model_entry = ece_data[model]
        if not isinstance(model_entry, dict):
            raise TypeError(f"ece.json[{model}] must be object.")
        ece_stats[model] = {}
        for method in ECE_METHOD_ORDER:
            if method not in model_entry:
                raise KeyError(f"Missing method in ece.json[{model}]: {method}")
            method_entry = model_entry[method]
            if not isinstance(method_entry, dict):
                raise TypeError(f"ece.json[{model}][{method}] must be object.")
            ece_stats[model][method] = {
                "without_warmup": _format_mean_std(
                    method_entry["without_warmup"], scale=100.0, percent=True
                ),
                "with_warmup": _format_mean_std(
                    method_entry["with_warmup"], scale=100.0, percent=True
                ),
            }

    output = {
        "accuracy": accuracy_stats,
        "ece_percent": ece_stats,
        "config": {
            "format": "(mean ± std), fixed to 3 decimal places",
            "accuracy_scale": "x1",
            "ece_scale": "x100 and suffixed with %",
            "inputs": {
                "accuracy": str(accuracy_path),
                "ece": str(ece_path),
            },
        },
    }

    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
