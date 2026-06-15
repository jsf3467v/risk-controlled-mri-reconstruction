"""Uncertainty gate. Empirical-Bernstein risk control of residual thresholds and the accept decision."""
import numpy as np


def risk_controlled_threshold(scores, errors, epsilon, alpha, candidates=16):
    """Largest residual threshold whose accepted-set risk stays under epsilon.

    The file uses an empirical-Bernstein upper bound (Maurer-Pontil) over a fixed grid of acceptance
    fractions, Bonferroni-corrected across the grid. So the data-dependent threshold keeps a real
    high-probability guarantee. The README explains the bound, the grid, and the candidates lever.
    """
    order = np.argsort(scores)
    scores = np.asarray(scores)[order]
    errors = np.clip(np.asarray(errors)[order], 0.0, 1.0)
    n = len(errors)
    k = np.arange(1, n + 1)
    mean = np.cumsum(errors) / k
    variance = np.maximum(np.cumsum(errors ** 2) / k - mean ** 2, 0.0) * k / np.maximum(k - 1, 1)
    grid = np.unique(np.linspace(1, n, min(candidates, n)).round().astype(int)) - 1
    log_term = np.log(2.0 * len(grid) / alpha)
    bound = (mean[grid] + np.sqrt(2 * variance[grid] * log_term / k[grid])
             + 7 * log_term / (3 * np.maximum(k[grid] - 1, 1)))
    valid = bound <= epsilon
    if not valid.any():
        return float("-inf")
    return float(scores[grid[np.flatnonzero(valid)[-1]]])


def mondrian_thresholds(scores, errors, groups, epsilon, alpha, minimum, candidates=16):
    """One risk-controlled threshold per group; groups with too few slices default to escalate."""
    scores, errors, groups = np.asarray(scores), np.asarray(errors), np.asarray(groups)
    thresholds = {}
    for name in np.unique(groups):
        pick = groups == name
        if pick.sum() < minimum:
            thresholds[str(name)] = float("-inf")
        else:
            thresholds[str(name)] = risk_controlled_threshold(scores[pick], errors[pick], epsilon, alpha, candidates)
    return thresholds


def accepted_groups(scores, groups, thresholds):
    """Boolean accept mask using each slice's group-specific threshold."""
    scores = np.asarray(scores)
    groups = np.asarray(groups).astype(str)
    limit = np.full(scores.shape, float("-inf"))
    for name, value in thresholds.items():
        limit[groups == str(name)] = value
    return scores <= limit