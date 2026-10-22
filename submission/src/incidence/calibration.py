"""Probability-scale metrics and category-aware OOF interval calibration."""

import numpy as np
import pandas as pd

from incidence.models import logit, probability


def interval_radius(actual, predicted_logit, groups, coverage: float = 0.8, scale=1.0) -> float:
    """Use group-max OOF errors so duplicate rows do not count as independent evidence."""
    scores = pd.DataFrame({
        "group": np.asarray(groups),
        "score": np.abs(logit(actual) - np.asarray(predicted_logit)) / np.asarray(scale),
    }).groupby("group")["score"].max().to_numpy()
    if not len(scores) or not 0 < coverage < 1:
        raise ValueError("Calibration needs nonempty groups and coverage in (0, 1)")
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * coverage)))
    return float(np.sort(scores)[rank - 1])


def prediction_intervals(predicted_logit, radius: float, scale=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded, logit-symmetric predictive intervals around a point estimate."""
    return probability(np.asarray(predicted_logit) - radius * np.asarray(scale)), probability(np.asarray(predicted_logit) + radius * np.asarray(scale))


def regression_metrics(actual, point, lower, upper) -> dict[str, float]:
    """Measure errors, coverage and Winkler interval score on the probability scale."""
    actual, point, lower, upper = [np.asarray(values, dtype=float) for values in (actual, point, lower, upper)]
    if not (np.isfinite([actual, point, lower, upper]).all()
            and np.all((0 <= lower) & (lower <= point) & (point <= upper) & (upper <= 1))):
        raise ValueError("Predictions must be finite and ordered within [0, 1]")
    interval_score = upper - lower + 10 * np.maximum(lower - actual, 0) + 10 * np.maximum(actual - upper, 0)
    return {
        "mae": float(np.mean(np.abs(actual - point))),
        "rmse": float(np.sqrt(np.mean((actual - point) ** 2))),
        "coverage_80": float(np.mean((lower <= actual) & (actual <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "interval_score": float(np.mean(interval_score)),
    }