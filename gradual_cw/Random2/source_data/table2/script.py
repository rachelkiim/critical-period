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

METHOD_ORDER = ["msp", "temp_scaling", "odin", "energy"]


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


def _format_mean_std(values: Any) -> str:
    arr = _as_1d_float_array(values, "values")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return f"({mean:.3f} ± {std:.3f})"


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    ood_path = base_dir / "ood_detection.json"
    out_path = base_dir / "stats.json"

    ood_data = load_json(ood_path)

    ood_stats: dict[str, Any] = {}
    for model in MODEL_ORDER:
        if model not in ood_data:
            raise KeyError(f"Missing model in ood_detection.json: {model}")
        model_entry = ood_data[model]
        if not isinstance(model_entry, dict):
            raise TypeError(f"ood_detection.json[{model}] must be object.")
        ood_stats[model] = {}
        for method in METHOD_ORDER:
            if method not in model_entry:
                raise KeyError(f"Missing method in ood_detection.json[{model}]: {method}")
            method_entry = model_entry[method]
            if not isinstance(method_entry, dict):
                raise TypeError(f"ood_detection.json[{model}][{method}] must be object.")
            ood_stats[model][method] = {
                "without_warmup": _format_mean_std(method_entry["without_warmup"]),
                "with_warmup": _format_mean_std(method_entry["with_warmup"]),
            }

    output = {
        "ood_detection": ood_stats,
        "config": {
            "format": "(mean ± std), fixed to 3 decimal places",
            "ood_scale": "x1",
            "input": str(ood_path),
        },
    }

    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
