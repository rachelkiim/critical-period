"""Model inference helpers for confidence and prediction extraction."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def inference(model, loader, device=None):
    """
    Evaluate the model on the given data loader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing the data to test on.
        device (torch.device): Device to use for testing.

    Returns:
        tuple: A tuple containing:
            - pred (np.ndarray): Boolean array indicating if each prediction is correct.
            - conf (np.ndarray): Confidence scores for each prediction.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    pred = []  # List to store prediction results
    conf = []  # List to store confidence scores

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Perform predictions using the model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pred.append((predicted == labels).cpu().numpy())
            conf.append(F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy())

    pred = np.concatenate(pred)
    conf = np.concatenate(conf)

    return pred, conf


def get_predictions_and_confidence(
    model,
    loader,
    device=None,
    return_probs=False,
    return_logits=False,
):
    """
    Get predictions, confidence scores, predicted labels, and true labels.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader providing the data.
        device: Device to use for evaluation.
        return_probs: If True, also return full softmax probabilities (N, C).
        return_logits: If True, also return raw logits (N, C).

    Returns:
        Base tuple: (pred_correct, conf, pred_labels, true_labels)
        + probabilities array if return_probs=True
        + logits array if return_logits=True
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    pred_correct = []
    conf = []
    pred_labels = []
    true_labels = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_conf, predicted = torch.max(probs, dim=1)

            pred_correct.append((predicted == labels).cpu().numpy())
            conf.append(max_conf.cpu().numpy())
            pred_labels.append(predicted.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

            if return_probs:
                all_probs.append(probs.cpu().numpy())
            if return_logits:
                all_logits.append(outputs.cpu().numpy())

    result = (
        np.concatenate(pred_correct),
        np.concatenate(conf),
        np.concatenate(pred_labels),
        np.concatenate(true_labels),
    )
    if return_probs:
        result += (np.concatenate(all_probs),)
    if return_logits:
        result += (np.concatenate(all_logits),)
    return result


def get_confidence_only(model, loader, device=None) -> np.ndarray:
    """
    Get only confidence scores (for OOD detection where labels don't matter).

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader providing the data.
        device: Device to use for evaluation.

    Returns:
        np.ndarray: Confidence scores for each sample.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    conf = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_conf = torch.max(probs, dim=1)[0]
            conf.append(max_conf.cpu().numpy())

    return np.concatenate(conf)
