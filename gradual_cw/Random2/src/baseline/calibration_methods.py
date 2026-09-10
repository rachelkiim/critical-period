"""Post-hoc calibration baseline methods.

Implements temperature scaling, vector scaling, and isotonic regression
for post-hoc calibration of neural network confidence scores.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from src.evaluation.calibration import reliability_diagram, ece


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def collect_logits_and_labels(model, loader, device=None):
    """Run a single forward pass over the loader and collect all logits and labels.

    Args:
        model: Neural network model.
        loader: DataLoader for the dataset.
        device: Device to run on.

    Returns:
        Tuple of (logits, labels) as tensors on the given device.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for data, labels in loader:
            logits = model(data.to(device))
            all_logits.append(logits)
            all_labels.append(labels.to(device))

    return torch.cat(all_logits), torch.cat(all_labels)


def compute_ece_from_probs(probs, labels, num_bin=10):
    """Compute ECE from probability matrix and labels.

    Args:
        probs: Softmax probabilities (N, C) as numpy array or tensor.
        labels: Ground-truth labels (N,) as numpy array or tensor.
        num_bin: Number of bins for reliability diagram.

    Returns:
        float: The Expected Calibration Error.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    acc, conf, num_sample = reliability_diagram(correct, confidences, num_bin=num_bin)
    return ece(acc, conf, num_sample)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def find_best_temperature(model, loader, T_candidates, device=None):
    """Find the temperature T that minimizes ECE on the given dataset.

    Args:
        model: Neural network model.
        loader: DataLoader (validation set).
        T_candidates: List of temperature values to try.
        device: Device to run on.

    Returns:
        float: The best temperature T.
    """
    logits, labels = collect_logits_and_labels(model, loader, device)

    best_t = None
    best_ece = float('inf')
    for T in T_candidates:
        probs = F.softmax(logits / T, dim=1)
        current_ece = compute_ece_from_probs(probs, labels)
        if current_ece < best_ece:
            best_ece = current_ece
            best_t = T

    return best_t


def temperature_scaling_ece(model, loader, T, device=None):
    """Compute ECE after temperature scaling.

    Args:
        model: Neural network model.
        loader: DataLoader (test set).
        T: Temperature scaling factor.
        device: Device to run on.

    Returns:
        float: ECE value.
    """
    logits, labels = collect_logits_and_labels(model, loader, device)
    probs = F.softmax(logits / T, dim=1)
    return compute_ece_from_probs(probs, labels)


# ---------------------------------------------------------------------------
# Vector scaling
# ---------------------------------------------------------------------------

class VectorScaling(nn.Module):
    """Vector Scaling: W * logits + b (element-wise)."""

    def __init__(self, num_classes):
        super().__init__()
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        return logits * self.W + self.b


def train_vector_scaling(model, loader, num_classes=10, max_iter=50, device=None):
    """Train a VectorScaling model on the given dataset using LBFGS.

    Args:
        model: Neural network model.
        loader: DataLoader (validation set).
        num_classes: Number of output classes.
        max_iter: Maximum LBFGS iterations.
        device: Device to run on.

    Returns:
        VectorScaling: The trained scaling model.
    """
    if device is None:
        device = next(model.parameters()).device

    logits, labels = collect_logits_and_labels(model, loader, device)

    vs_model = VectorScaling(num_classes).to(device)
    optimizer = torch.optim.LBFGS(vs_model.parameters(), lr=0.0005, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = vs_model(logits)
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return vs_model


def vector_scaling_ece(model, vs_model, loader, device=None):
    """Compute ECE using a trained VectorScaling model.

    Args:
        model: Neural network model.
        vs_model: Trained VectorScaling model.
        loader: DataLoader (test set).
        device: Device to run on.

    Returns:
        float: ECE value.
    """
    if device is None:
        device = next(model.parameters()).device

    logits, labels = collect_logits_and_labels(model, loader, device)
    vs_model.eval()
    with torch.no_grad():
        scaled_logits = vs_model(logits)
    probs = F.softmax(scaled_logits, dim=1)
    return compute_ece_from_probs(probs, labels)


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------

def train_isotonic_regression(model, loader, device=None):
    """Train an isotonic regression calibrator.

    Args:
        model: Neural network model.
        loader: DataLoader (validation set).
        device: Device to run on.

    Returns:
        IsotonicRegression: The trained calibrator.
    """
    logits, labels = collect_logits_and_labels(model, loader, device)
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    correct = predictions.eq(labels)

    confidences = confidences.cpu().numpy()
    correct = correct.cpu().numpy().astype(int)

    ir_model = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir_model.fit(confidences, correct)
    return ir_model


def isotonic_regression_ece(model, ir_model, loader, num_bin=15, device=None):
    """Compute ECE using a trained isotonic regression calibrator.

    Args:
        model: Neural network model.
        ir_model: Trained IsotonicRegression calibrator.
        loader: DataLoader (test set).
        num_bin: Number of bins for reliability diagram.
        device: Device to run on.

    Returns:
        float: ECE value.
    """
    logits, labels = collect_logits_and_labels(model, loader, device)
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    correct = predictions.eq(labels)

    confidences = confidences.cpu().numpy()
    correct = correct.cpu().numpy()

    calibrated_confidences = ir_model.transform(confidences)

    acc, conf, num_sample = reliability_diagram(correct, calibrated_confidences, num_bin=num_bin)
    return ece(acc, conf, num_sample)
