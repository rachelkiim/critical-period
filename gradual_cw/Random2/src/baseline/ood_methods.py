"""OOD detection baseline methods.

Implements MSP, temperature scaling, ODIN, and energy score
for out-of-distribution detection.
"""

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.ood_detection import roc_curve, auc
from src.data.dataloader import CIFAR10_MEAN, CIFAR10_STD


# ---------------------------------------------------------------------------
# Score functions (per-batch)
# ---------------------------------------------------------------------------

@torch.no_grad()
def msp_score(model, x):
    """Maximum softmax probability (baseline).

    Args:
        model: Neural network model (already on device).
        x: Input batch tensor (already on device).

    Returns:
        np.ndarray: MSP scores for the batch.
    """
    model.eval()
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    scores, _ = torch.max(probs, dim=1)
    return scores.cpu().numpy()


@torch.no_grad()
def temperature_scaled_score(model, x, T=1.0):
    """Temperature-scaled maximum softmax probability.

    Args:
        model: Neural network model.
        x: Input batch tensor.
        T: Temperature scaling factor.

    Returns:
        np.ndarray: Temperature-scaled MSP scores.
    """
    model.eval()
    logits = model(x)
    probs = F.softmax(logits / T, dim=1)
    scores, _ = torch.max(probs, dim=1)
    return scores.cpu().numpy()


def odin_score(model, x, T=1000.0, epsilon=0.001,
               mean=None, std=None, use_energy=False):
    """ODIN score with input perturbation.

    Applies gradient-based input perturbation, then computes either
    temperature-scaled MSP or energy score.

    Args:
        model: Neural network model.
        x: Input batch tensor.
        T: Temperature scaling factor.
        epsilon: Perturbation magnitude.
        mean: Normalization mean (defaults to CIFAR10_MEAN).
        std: Normalization std (defaults to CIFAR10_STD).
        use_energy: If True, return energy score instead of MSP.

    Returns:
        np.ndarray: ODIN (or energy) scores for the batch.
    """
    if mean is None:
        mean = CIFAR10_MEAN
    if std is None:
        std = CIFAR10_STD

    model.eval()

    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)[None, :, None, None]

    x = x.clone().detach().requires_grad_(True)

    # Temporarily freeze model parameters
    frozen = []
    for p in model.parameters():
        if p.requires_grad:
            frozen.append(p)
            p.requires_grad_(False)

    try:
        logits = model(x)
        scaled = logits / T
        _, yhat = torch.max(scaled, dim=1)
        loss = F.cross_entropy(scaled, yhat)
        loss.backward()

        # Compute perturbation in pixel space
        grad_sign_pix = (x.grad.detach() / std_t).sign()
        x_pix = x.detach() * std_t + mean_t  # denormalize
        x_pix_pert = torch.clamp(x_pix - epsilon * grad_sign_pix, 0.0, 1.0)

        with torch.no_grad():
            x_pert = (x_pix_pert - mean_t) / std_t  # renormalize
            logits_pert = model(x_pert)

            if use_energy:
                scores = energy_score(logits_pert)
            else:
                probs = F.softmax(logits_pert / T, dim=1)
                scores, _ = probs.max(dim=1)
                scores = scores.cpu().numpy()
        return scores
    finally:
        for p in frozen:
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)


def energy_score(logits):
    """Energy score: log(sum(exp(logits))).

    Args:
        logits: Logits tensor or numpy array (N, C).

    Returns:
        np.ndarray: Energy scores (N,).
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    return np.log(np.sum(np.exp(logits), axis=1))


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def compute_scores_for_loader(model, loader, score_fn, device=None):
    """Compute scores for an entire DataLoader using a per-batch score function.

    Args:
        model: Neural network model.
        loader: DataLoader for the dataset.
        score_fn: Callable(model, batch_tensor) -> np.ndarray of scores.
        device: Device to run on.

    Returns:
        np.ndarray: Concatenated scores for all samples.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_scores = []
    for data, *_ in loader:
        data = data.to(device)
        scores = score_fn(model, data)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        all_scores.extend(scores)

    return np.array(all_scores)


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC from in-distribution and OOD scores.

    Args:
        id_scores: Scores for in-distribution data.
        ood_scores: Scores for out-of-distribution data.

    Returns:
        float: AUROC value.
    """
    fpr, tpr = roc_curve(id_scores, ood_scores)
    return auc(fpr, tpr)


def find_best_params(model, id_loader, ood_loader, param_grid,
                     score_fn_factory, device=None):
    """Generic hyperparameter search maximizing AUROC.

    Args:
        model: Neural network model.
        id_loader: DataLoader for in-distribution validation set.
        ood_loader: DataLoader for OOD validation set.
        param_grid: List of parameter dicts to try, e.g.
            [{"T": 1.5}, {"T": 2.0}] or
            [{"T": 1.5, "epsilon": 0.01}, {"T": 2.0, "epsilon": 0.01}]
        score_fn_factory: Callable(params_dict) -> score_fn,
            where score_fn is Callable(model, batch) -> scores.
        device: Device to run on.

    Returns:
        Tuple of (best_params_dict, best_auroc).
    """
    if device is None:
        device = next(model.parameters()).device

    best_params = None
    best_auroc = -1.0

    for params in param_grid:
        score_fn = score_fn_factory(params)
        id_scores = compute_scores_for_loader(model, id_loader, score_fn, device)
        ood_scores = compute_scores_for_loader(model, ood_loader, score_fn, device)

        if len(id_scores) > 0 and len(ood_scores) > 0:
            auroc = compute_auroc(id_scores, ood_scores)
            if auroc > best_auroc:
                best_auroc = auroc
                best_params = params

    return best_params, best_auroc
