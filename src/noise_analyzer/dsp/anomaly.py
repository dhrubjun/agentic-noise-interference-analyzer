from dataclasses import dataclass

import numpy as np

from noise_analyzer.dsp.baseline import BaselineResult


@dataclass(frozen=True)
class DeviationScoreResult:
    """Robust deviation scores for a scalar feature track."""

    scores: np.ndarray
    baseline_location: float
    baseline_scale: float
    method: str


def calculate_robust_deviation_scores(
    feature_values: np.ndarray,
    baseline: BaselineResult,
) -> DeviationScoreResult:
    """Calculate robust standardized deviation scores."""

    values = np.asarray(
        feature_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "feature_values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "feature_values must not be empty."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "feature_values must contain only finite values."
        )

    if baseline.status != "valid":
        raise ValueError(
            "baseline must have valid non-zero scale."
        )

    scores = (
        values - baseline.location
    ) / baseline.robust_scale

    return DeviationScoreResult(
        scores=scores,
        baseline_location=baseline.location,
        baseline_scale=baseline.robust_scale,
        method="median_mad_standardized",
    )