"""Statistical test helpers for evaluation."""

import numpy as np
from scipy import stats


def _format_p_scientific(p: float, precision: int = 4) -> str:
    p = float(p)
    if p <= 0.0:
        p = float(np.nextafter(0.0, 1.0))
    return f"{p:.{precision}e}"


def mannwhitneyu(x, y, alternative="two-sided", method="asymptotic"):
    """Run Mann-Whitney U test and return summary statistics.

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    alternative : {"two-sided", "less", "greater"}
        Alternative hypothesis for the test.
    method : {"asymptotic", "exact", "auto"}
        P-value computation method used by SciPy.

    Returns
    -------
    dict
        Dictionary with keys: U, p, r, n1, n2.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n1 = len(x)
    n2 = len(y)
    n_total = n1 + n2

    res = stats.mannwhitneyu(x, y, alternative=alternative, method=method)
    u_stat = float(res.statistic)
    p_value = float(res.pvalue)

    mu_u = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n_total + 1) / 12)
    z_score = (u_stat - mu_u) / sigma_u
    effect_r = float(z_score / np.sqrt(n_total))

    return {
        "U": u_stat,
        "p": _format_p_scientific(p_value),
        "r": effect_r,
        "n1": n1,
        "n2": n2,
    }


def wilcoxon_signed_rank(
    x,
    y=None,
    alternative="two-sided",
    zero_method="wilcox",
    correction=False,
    method="approx",
):
    """Run Wilcoxon signed-rank test and return summary statistics.

    Parameters
    ----------
    x, y : array-like
        Paired samples. If ``y`` is None, ``x`` is treated as paired differences.
    alternative : {"two-sided", "less", "greater"}
        Alternative hypothesis for the test.
    zero_method : {"wilcox", "pratt", "zsplit"}
        Strategy for handling zero-differences.
    correction : bool
        Whether to apply continuity correction in normal approximation.
    method : {"auto", "exact", "approx"}
        P-value computation method used by SciPy.

    Returns
    -------
    dict
        Dictionary with keys: W, p, r, n.
    """
    x = np.asarray(x)
    if y is None:
        diff = x
    else:
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape for paired Wilcoxon test.")
        diff = x - y

    diff = np.ravel(diff)
    diff_nonzero = diff[diff != 0]
    n_nonzero = int(len(diff_nonzero))
    if n_nonzero == 0:
        raise ValueError("All paired differences are zero; Wilcoxon test is undefined.")

    res = stats.wilcoxon(
        diff,
        alternative=alternative,
        zero_method=zero_method,
        correction=correction,
        method=method,
    )
    w_stat = float(res.statistic)
    p_value = float(res.pvalue)

    z_score = getattr(res, "zstatistic", None)

    if z_score is not None:
        z_score = float(z_score)
        effect_r = float(z_score / np.sqrt(n_nonzero))
    else:
        effect_r = None

    return {
        "W": w_stat,
        "p": _format_p_scientific(p_value),
        "r": effect_r,
        "n": n_nonzero,
    }
