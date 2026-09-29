from dataclasses import dataclass

import numpy as np

from noise_analyzer.io.validation import (
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord


@dataclass(frozen=True)
class TimeDomainStatistics:
    """Basic time-domain measurements for a signal."""

    mean: float
    median: float
    rms: float
    standard_deviation: float
    variance: float
    minimum: float
    maximum: float
    peak_to_peak: float


def calculate_time_domain_statistics(
    record: SignalRecord,
) -> TimeDomainStatistics:
    """Calculate basic time-domain statistics for a signal."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    samples = record.samples

    return TimeDomainStatistics(
        mean=float(np.mean(samples)),
        median=float(np.median(samples)),
        rms=float(np.sqrt(np.mean(samples**2))),
        standard_deviation=float(np.std(samples)),
        variance=float(np.var(samples)),
        minimum=float(np.min(samples)),
        maximum=float(np.max(samples)),
        peak_to_peak=float(np.ptp(samples)),
    )