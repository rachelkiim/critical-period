"""Weights & Biases logging wrapper."""

from typing import Optional


class WandbLogger:
    """Wrapper for wandb logging with optional disable support."""

    def __init__(
        self,
        project: str,
        config: dict,
        entity: Optional[str] = None,
        disabled: bool = False,
        run_name: Optional[str] = None,
    ):
        """Initialize wandb logger.

        Args:
            project: Wandb project name
            config: Experiment configuration dict
            entity: Wandb entity/team name
            disabled: If True, logging is disabled (--no_wandb mode)
            run_name: Optional run name (defaults to exp_id)
        """
        self.disabled = disabled
        self.run = None

        if not disabled:
            try:
                import wandb

                self.wandb = wandb
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    config=config,
                    name=run_name or config.get("exp_id", None),
                )
            except ImportError:
                print("Warning: wandb not installed. Logging disabled.")
                self.disabled = True

    def log(self, data: dict, step: Optional[int] = None, commit: bool = True):
        """Log metrics to wandb.

        Args:
            data: Dictionary of metrics to log
            step: Optional step number (deprecated, use epoch in data instead)
            commit: Whether to commit the log immediately
        """
        if self.disabled:
            return

        # Don't use step parameter to avoid conflicts with multiple training runs
        self.wandb.log(data, commit=commit)

    def define_metric(self, metric_name: str, step_metric: str):
        """Define a metric with a custom step metric.

        Args:
            metric_name: Glob pattern for metrics (e.g., "train_w/*")
            step_metric: The metric to use as x-axis (e.g., "train_w/epoch")
        """
        if self.disabled:
            return

        self.wandb.define_metric(metric_name, step_metric=step_metric)

    def log_image(self, key: str, image_path: str, caption: Optional[str] = None):
        """Log an image to wandb.

        Args:
            key: Metric key for the image
            image_path: Path to the image file
            caption: Optional caption for the image
        """
        if self.disabled:
            return

        # SVG files need to be logged as HTML since PIL can't open them
        if image_path.lower().endswith(".svg"):
            with open(image_path, "r") as f:
                svg_content = f.read()
            self.wandb.log({key: self.wandb.Html(svg_content)})
        else:
            self.wandb.log({key: self.wandb.Image(image_path, caption=caption)})

    def log_summary(self, data: dict):
        """Log summary metrics (shown in wandb table).

        Args:
            data: Dictionary of summary metrics
        """
        if self.disabled:
            return

        for key, value in data.items():
            self.run.summary[key] = value

    def finish(self):
        """Finish the wandb run."""
        if not self.disabled and self.run:
            self.run.finish()
