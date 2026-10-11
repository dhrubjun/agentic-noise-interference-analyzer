from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    """Robust baseline statistics for a scalar feature track."""

    location: float
    mad: float
    robust_scale: float
    method: str
    status: str
    number_of_values: int


def estimate_robust_baseline(
    feature_values: np.ndarray,
) -> BaselineResult:
    """Estimate baseline location and scale using median and MAD."""

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

    location = float(
        np.median(values)
    )

    absolute_deviations = np.abs(
        values - location
    )

    mad = float(
        np.median(absolute_deviations)
    )

    robust_scale = float(
        1.4826 * mad
    )

    if robust_scale == 0.0:
        status = "degenerate"
    else:
        status = "valid"

    return BaselineResult(
        location=location,
        mad=mad,
        robust_scale=robust_scale,
        method="median_mad",
        status=status,
        number_of_values=int(values.size),
    )