"""JSON file logger for experiment outputs."""

import json
from pathlib import Path

from src.logger.encoders import NumpyEncoder
from src.logger.results import CalibrationResults, OODResults, TrainingResults


class JSONLogger:
    """Logger for saving results to JSON files."""

    def __init__(self, exp_path: str):
        """Initialize JSON logger.

        Args:
            exp_path: Path to experiment directory
        """
        self.exp_path = Path(exp_path)
        self.exp_path.mkdir(parents=True, exist_ok=True)

        self.training_results = TrainingResults()
        self.calibration_results = CalibrationResults()
        self.ood_results = OODResults()

    def save_training_results(self, filename: str = "training_results.json"):
        """Save training results to JSON file."""
        path = self.exp_path / filename
        with open(path, "w") as f:
            json.dump(self.training_results.to_dict(), f, indent=2, cls=NumpyEncoder)

    def save_calibration_results(self, filename: str = "calibration.json"):
        """Save calibration results to JSON file."""
        path = self.exp_path / filename
        with open(path, "w") as f:
            json.dump(self.calibration_results.to_dict(), f, indent=2, cls=NumpyEncoder)

    def save_ood_results(self, filename: str = "ood_detection.json"):
        """Save OOD detection results to JSON file."""
        path = self.exp_path / filename
        with open(path, "w") as f:
            json.dump(self.ood_results.to_dict(), f, indent=2, cls=NumpyEncoder)

    def save_all(self):
        """Save all results to JSON files."""
        self.save_training_results()
        self.save_calibration_results()
        self.save_ood_results()

    @staticmethod
    def load_json(path: str) -> dict:
        """Load JSON file."""
        with open(path, "r") as f:
            return json.load(f)
