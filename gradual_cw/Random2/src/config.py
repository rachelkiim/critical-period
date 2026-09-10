"""Configuration module for experiment setup and argument parsing."""

import argparse
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration dataclass for experiment hyperparameters."""

    # Model configuration
    batch_size: int = 128
    depth: int = 2
    width: int = 256
    mode: str = "BP"  # "BP" (Backpropagation) or "FA" (Feedback Alignment)

    # Random noise pretraining
    lr_noise: float = 1e-4
    weight_decay_noise: float = 0
    num_noise: int = 32000
    epochs_noise: int = 10

    # Main training
    lr: float = 2e-5
    weight_decay: float = 1e-3
    num_image: int = 1000
    epochs: int = 30

    # Experiment configuration
    num_nets: int = 10
    seed: int = 42

    # Logging
    no_wandb: bool = False
    no_figures: bool = False
    wandb_project: str = "Random2"
    wandb_entity: Optional[str] = None

    # Paths
    data_path: str = "./data"
    exp_base_path: str = "./exp"

    # Dataset
    dataset: str = "cifar10"
    ood_dataset: str = "svhn"
    num_classes: int = 10

    # Derived fields (set after initialization)
    exp_id: str = field(default="", init=False)
    exp_path: str = field(default="", init=False)

    def __post_init__(self):
        """Generate exp_id and paths after initialization."""
        self.exp_id = self.generate_exp_id()
        self.exp_path = str(Path(self.exp_base_path) / self.exp_id)

    def generate_exp_id(self) -> str:
        """Generate experiment ID from timestamp and key hyperparameters.

        Format: {timestamp}_bs{batch}_d{depth}_w{width}_ep{epochs}_nn{nets}
        """
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        return f"{timestamp}_bs{self.batch_size}_d{self.depth}_w{self.width}_ep{self.epochs}_nn{self.num_nets}"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Optional[str] = None):
        """Save configuration to JSON file."""
        if path is None:
            path = Path(self.exp_path) / "config.json"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Remove derived fields before creating instance
        config_dict.pop('exp_id', None)
        config_dict.pop('exp_path', None)

        return cls(**config_dict)


def parse_args() -> ExperimentConfig:
    """Parse command line arguments and return ExperimentConfig."""
    parser = argparse.ArgumentParser(
        description="Calibration Analysis - Random Noise Warm-up Training"
    )

    # Model configuration
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Network depth (default: 2)")
    parser.add_argument("--width", type=int, default=256,
                        help="Network width / hidden units (default: 256)")
    parser.add_argument("--mode", type=str, default="BP", choices=["BP", "FA"],
                        help="Training mode: BP or FA (default: FA)")

    # Random noise pretraining
    parser.add_argument("--lr_noise", type=float, default=2e-5,
                        help="Learning rate for noise pretraining (default: 2e-5)")
    parser.add_argument("--weight_decay_noise", type=float, default=1e-3,
                        help="Weight decay for noise pretraining (default: 1e-3)")
    parser.add_argument("--num_noise", type=int, default=32000,
                        help="Number of noise samples per epoch (default: 32000)")
    parser.add_argument("--epochs_noise", type=int, default=10,
                        help="Number of noise pretraining epochs (default: 10)")

    # Main training
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for main training (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for main training (default: 0)")
    parser.add_argument("--num_image", type=int, default=1000,
                        help="Number of training images (default: 1000)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")

    # Experiment configuration
    parser.add_argument("--num_nets", type=int, default=10,
                        help="Number of independent networks to train (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Logging
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--wandb_project", type=str, default="Random2",
                        help="Wandb project name (default: Random2)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity/team name")

    # Paths
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to datasets (default: ./data)")
    parser.add_argument("--exp_base_path", type=str, default="./exp",
                        help="Base path for experiment outputs (default: ./exp)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Direct output directory (skip auto exp_id generation)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"],
                        help="Training dataset (default: cifar10)")
    parser.add_argument("--ood_dataset", type=str, default="svhn",
                        choices=["svhn", "cifar100", "mnist"],
                        help="OOD detection dataset (default: svhn)")

    args = parser.parse_args()

    # Determine num_classes based on dataset
    num_classes = 10 if args.dataset == "cifar10" else 100

    # Create config from parsed arguments
    config = ExperimentConfig(
        batch_size=args.batch_size,
        depth=args.depth,
        width=args.width,
        mode=args.mode,
        lr_noise=args.lr_noise,
        weight_decay_noise=args.weight_decay_noise,
        num_noise=args.num_noise,
        epochs_noise=args.epochs_noise,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_image=args.num_image,
        epochs=args.epochs,
        num_nets=args.num_nets,
        seed=args.seed,
        no_wandb=args.no_wandb,
        no_figures=args.no_figures,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        data_path=args.data_path,
        exp_base_path=args.exp_base_path,
        dataset=args.dataset,
        ood_dataset=args.ood_dataset,
        num_classes=num_classes,
    )

    # Override exp_path if --output_dir is provided
    if args.output_dir:
        config.exp_path = args.output_dir
        config.exp_id = os.path.basename(args.output_dir)

    return config


def setup_experiment_dirs(config: ExperimentConfig) -> dict:
    """Create experiment directory structure.

    Returns:
        dict: Paths to created directories
    """
    exp_path = Path(config.exp_path)

    dirs = {
        "root": exp_path,
        "checkpoints": exp_path / "checkpoints",
        "figures": exp_path / "figures",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return {k: str(v) for k, v in dirs.items()}
