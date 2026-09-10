"""OOD detection utilities including ROC and AUC computation."""

import numpy as np


def cdf(data, bins=20, range=(0, 1)):
    """
    Computes the cumulative distribution function (CDF) of the given data.

    Args:
        data (array-like): Input data for which the CDF is computed.
        bins (int, optional): Number of bins for the histogram. Default is 20.
        range (tuple, optional): The lower and upper range of the bins.

    Returns:
        tuple: A tuple containing the CDF values and the bin edges.
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=False)
    cdf_vals = np.cumsum(hist) / np.sum(hist)
    return cdf_vals, bin_edges


def roc_curve(a, b):
    """
    Computes the ROC curve points for two score distributions.

    Uses O(n log n) vectorized computation. Does not require equal-length inputs.

    Args:
        a (array-like): Scores for the positive class (e.g., in-distribution).
        b (array-like): Scores for the negative class (e.g., out-of-distribution).

    Returns:
        tuple: A tuple of false positive rates (FPR) and true positive rates (TPR).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    scores = np.concatenate([a, b])
    labels = np.concatenate([np.ones_like(a, dtype=int), np.zeros_like(b, dtype=int)])

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)

    P = np.sum(labels)
    N = len(labels) - P
    distinct = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    idx = np.where(distinct)[0]

    fpr = np.r_[0.0, fps[idx] / (N if N > 0 else 1)]
    tpr = np.r_[0.0, tps[idx] / (P if P > 0 else 1)]

    return fpr, tpr


def auc(fpr, tpr):
    """
    Calculates the area under the ROC curve using the trapezoidal rule.

    Args:
        fpr (array-like): False positive rates.
        tpr (array-like): True positive rates.

    Returns:
        float: The calculated area under the curve (AUC).
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    assert fpr.shape == tpr.shape, "Length of input arrays must be the same."

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    return float(np.trapezoid(tpr, fpr))
