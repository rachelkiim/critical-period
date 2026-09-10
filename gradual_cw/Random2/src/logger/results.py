"""Result dataclasses for experiment outputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingResults:
    """Data structure for storing training results."""

    with_warmup: Dict[str, List] = field(default_factory=lambda: {"networks": []})
    without_warmup: Dict[str, List] = field(default_factory=lambda: {"networks": []})

    def add_network_result(
        self,
        net_id: int,
        with_warmup: bool,
        warmup_metrics: Optional[Dict] = None,
        selected_epochs: Optional[Dict] = None,
        train_loss: List[float] = None,
        train_acc: List[float] = None,
        test_loss: List[float] = None,
        test_acc: List[float] = None,
        train_ece: List[float] = None,
        train_conf: List[float] = None,
        test_ece: List[float] = None,
        test_conf: List[float] = None,
    ):
        """Add results for a single network."""
        result = {
            "net_id": net_id,
            "train_loss": train_loss or [],
            "train_acc": train_acc or [],
            "test_loss": test_loss or [],
            "test_acc": test_acc or [],
            "train_ece": train_ece or [],
            "train_conf": train_conf or [],
            "test_ece": test_ece or [],
            "test_conf": test_conf or [],
        }

        if warmup_metrics:
            result["warmup"] = warmup_metrics
        if selected_epochs:
            result["selected_epochs"] = selected_epochs

        target = self.with_warmup if with_warmup else self.without_warmup
        target["networks"].append(result)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "with_warmup": self.with_warmup,
            "without_warmup": self.without_warmup,
        }


@dataclass
class CalibrationResults:
    """Data structure for storing calibration results."""

    with_warmup: Dict[str, Any] = field(default_factory=dict)
    without_warmup: Dict[str, Any] = field(default_factory=dict)

    def set_results(
        self,
        with_warmup: bool,
        reliability_diagram: Dict[str, List],
        ece_final: float,
        mce_final: Optional[float] = None,
        acc_conf_gap: Optional[float] = None,
        ece_list: Optional[List[float]] = None,
        gap_list: Optional[List[float]] = None,
        reliability_diagram_list: Optional[List[Dict[str, List]]] = None,
    ):
        """Set calibration results for a condition."""
        result = {
            "reliability_diagram": reliability_diagram,
            "ece_final": ece_final,
        }

        if mce_final is not None:
            result["mce_final"] = mce_final
        if acc_conf_gap is not None:
            result["acc_conf_gap"] = acc_conf_gap
        if ece_list is not None:
            result["ece_list"] = ece_list
        if gap_list is not None:
            result["gap_list"] = gap_list
        if reliability_diagram_list is not None:
            result["reliability_diagram_list"] = reliability_diagram_list

        if with_warmup:
            self.with_warmup = result
        else:
            self.without_warmup = result

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "with_warmup": self.with_warmup,
            "without_warmup": self.without_warmup,
        }


@dataclass
class OODResults:
    """Data structure for storing OOD detection results."""

    with_warmup: Dict[str, Any] = field(default_factory=dict)
    without_warmup: Dict[str, Any] = field(default_factory=dict)

    def set_results(
        self,
        with_warmup: bool,
        conf_id: List[float],
        conf_ood: List[float],
        roc: Dict[str, List[float]],
        auroc: float,
        auroc_list: Optional[List[float]] = None,
        roc_list: Optional[List[Dict[str, List[float]]]] = None,
    ):
        """Set OOD detection results for a condition."""
        result = {
            "conf_id": conf_id,
            "conf_ood": conf_ood,
            "roc": roc,
            "auroc": auroc,
        }

        if auroc_list is not None:
            result["auroc_list"] = auroc_list
        if roc_list is not None:
            result["roc_list"] = roc_list

        if with_warmup:
            self.with_warmup = result
        else:
            self.without_warmup = result

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "with_warmup": self.with_warmup,
            "without_warmup": self.without_warmup,
        }
