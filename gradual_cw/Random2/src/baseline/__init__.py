"""Baseline methods for calibration and OOD detection comparison."""

from src.baseline.calibration_methods import (
    collect_logits_and_labels,
    compute_ece_from_probs,
    find_best_temperature,
    temperature_scaling_ece,
    VectorScaling,
    train_vector_scaling,
    vector_scaling_ece,
    train_isotonic_regression,
    isotonic_regression_ece,
)

from src.baseline.ood_methods import (
    msp_score,
    temperature_scaled_score,
    odin_score,
    energy_score,
    compute_scores_for_loader,
    compute_auroc,
    find_best_params,
)

__all__ = [
    # Calibration
    "collect_logits_and_labels",
    "compute_ece_from_probs",
    "find_best_temperature",
    "temperature_scaling_ece",
    "VectorScaling",
    "train_vector_scaling",
    "vector_scaling_ece",
    "train_isotonic_regression",
    "isotonic_regression_ece",
    # OOD detection
    "msp_score",
    "temperature_scaled_score",
    "odin_score",
    "energy_score",
    "compute_scores_for_loader",
    "compute_auroc",
    "find_best_params",
]
