"""Combined experiment logger for wandb and JSON outputs."""

from pathlib import Path
from typing import Optional

from src.logger.json_logger import JSONLogger
from src.logger.wandb_logger import WandbLogger


class ExperimentLogger:
    """Combined logger for both wandb and JSON."""

    def __init__(
        self,
        config: dict,
        exp_path: str,
        wandb_project: str = "bp-calibration",
        wandb_entity: Optional[str] = None,
        no_wandb: bool = False,
    ):
        """Initialize experiment logger.

        Args:
            config: Experiment configuration
            exp_path: Path to experiment directory
            wandb_project: Wandb project name
            wandb_entity: Wandb entity/team name
            no_wandb: If True, disable wandb logging
        """
        self.json_logger = JSONLogger(exp_path)
        self.wandb_logger = WandbLogger(
            project=wandb_project,
            config=config,
            entity=wandb_entity,
            disabled=no_wandb,
            run_name=config.get("exp_id"),
        )
        self.exp_path = Path(exp_path)

    def define_metrics_for_training(self, num_nets: int):
        """Define wandb metrics with custom x-axes for training curves.

        This allows each training run (different networks, warmup/no-warmup)
        to have its own epoch counter without step conflicts.

        Args:
            num_nets: Number of networks being trained
        """
        for net_id in range(num_nets):
            # Define metrics for warmup training
            self.wandb_logger.define_metric(f"warmup/net{net_id}/*", step_metric=f"warmup/net{net_id}/epoch")
            # Define metrics for training with warmup
            self.wandb_logger.define_metric(f"train_w/net{net_id}/*", step_metric=f"train_w/net{net_id}/epoch")
            # Define metrics for training without warmup
            self.wandb_logger.define_metric(f"train_wo/net{net_id}/*", step_metric=f"train_wo/net{net_id}/epoch")

    def log_metrics(
        self,
        metrics: dict,
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics
            step: Optional epoch number (will be included as 'epoch' field)
            prefix: Prefix for metric names (e.g., "train_w/net0/", "train_wo/net0/")
        """
        if prefix:
            prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            # Include epoch as a metric field for custom x-axis
            if step is not None:
                prefixed_metrics[f"{prefix}epoch"] = step
            self.wandb_logger.log(prefixed_metrics)
        else:
            self.wandb_logger.log(metrics)

    def log_final_metrics(self, metrics: dict):
        """Log final comparison metrics to wandb 'final' group."""
        final_metrics = {f"final/{k}": v for k, v in metrics.items()}
        self.wandb_logger.log(final_metrics)
        self.wandb_logger.log_summary(final_metrics)

    def log_figure(self, fig_name: str, fig_path: str):
        """Log a figure to wandb.

        Args:
            fig_name: Name/key for the figure
            fig_path: Path to the figure file
        """
        self.wandb_logger.log_image(f"figures/{fig_name}", fig_path)

    def add_network_result(self, *args, **kwargs):
        """Add network training result to JSON logger."""
        self.json_logger.training_results.add_network_result(*args, **kwargs)

    def set_calibration_results(self, *args, **kwargs):
        """Set calibration results in JSON logger."""
        self.json_logger.calibration_results.set_results(*args, **kwargs)

    def set_ood_results(self, *args, **kwargs):
        """Set OOD detection results in JSON logger."""
        self.json_logger.ood_results.set_results(*args, **kwargs)

    def save_all(self):
        """Save all JSON results and finish wandb."""
        self.json_logger.save_all()
        self.wandb_logger.finish()

    def get_figures_path(self) -> Path:
        """Get path to figures directory."""
        return self.exp_path / "figures"

    def get_checkpoints_path(self) -> Path:
        """Get path to checkpoints directory."""
        return self.exp_path / "checkpoints"
