"""Visualization module for BP Calibration Analysis."""

from src.visualization.training_plots import (
    plot_random_noise_samples,
    plot_warmup_curves,
    plot_training_loss,
    plot_training_acc,
    plot_validation_loss,
    plot_validation_acc,
)

from src.visualization.calibration_plots import (
    plot_ece_curves,
    plot_conf_acc_line,
    plot_conf_acc_scatter,
    plot_loss_acc_trajectory,
    plot_reliability_diagram,
    plot_ece_bar,
    plot_gap_bar,
    plot_aggregate_boxplots,
    plot_confidence_histogram,
)

from src.visualization.ood_plots import (
    plot_ood_histogram,
    plot_ood_boxplot,
    plot_id_ood_histogram,
    plot_cdf,
    plot_roc_curve,
    plot_auroc_bar,
)

from src.visualization.overconfidence_plots import (
    plot_confidence_distribution,
    plot_probability_distribution,
    plot_logit_distribution,
    plot_logit_boxplot,
    plot_logit_vs_probability,
    plot_mean_confidence_vs_outputs,
    plot_mean_confidence_vs_depth,
    plot_logit_variance_vs_depth,
)

from src.visualization.input_space_plots import (
    plot_confidence_map,
    plot_confidence_boxplot,
    plot_prediction_std_boxplot,
    plot_warmup_loss,
)

__all__ = [
    # Training plots
    "plot_random_noise_samples",
    "plot_warmup_curves",
    "plot_training_loss",
    "plot_training_acc",
    "plot_validation_loss",
    "plot_validation_acc",
    # Calibration plots
    "plot_ece_curves",
    "plot_conf_acc_line",
    "plot_conf_acc_scatter",
    "plot_loss_acc_trajectory",
    "plot_reliability_diagram",
    "plot_ece_bar",
    "plot_gap_bar",
    "plot_aggregate_boxplots",
    "plot_confidence_histogram",
    # OOD plots
    "plot_ood_histogram",
    "plot_ood_boxplot",
    "plot_id_ood_histogram",
    "plot_cdf",
    "plot_roc_curve",
    "plot_auroc_bar",
    # Overconfidence plots
    "plot_confidence_distribution",
    "plot_probability_distribution",
    "plot_logit_distribution",
    "plot_logit_boxplot",
    "plot_logit_vs_probability",
    "plot_mean_confidence_vs_outputs",
    "plot_mean_confidence_vs_depth",
    "plot_logit_variance_vs_depth",
    # Input space plots
    "plot_confidence_map",
    "plot_confidence_boxplot",
    "plot_prediction_std_boxplot",
    "plot_warmup_loss",
]
