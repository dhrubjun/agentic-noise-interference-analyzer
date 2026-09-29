from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from noise_analyzer.io.validation import (
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord


@dataclass(frozen=True)
class PSDResult:
    """Welch power spectral density result."""

    frequencies_hz: np.ndarray
    psd: np.ndarray
    sample_rate_hz: float
    window: str
    nperseg: int
    noverlap: int
    frequency_resolution_hz: float
    integrated_power: float


def calculate_welch_psd(
    record: SignalRecord,
    nperseg: int,
    noverlap: int | None = None,
    window: str = "hann",
) -> PSDResult:
    """Calculate one-sided Welch power spectral density."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    number_of_samples = record.number_of_samples

    if number_of_samples < 2:
        raise ValueError(
            "At least two samples are required for PSD analysis."
        )

    if nperseg < 2:
        raise ValueError(
            "nperseg must be at least 2."
        )

    if nperseg > number_of_samples:
        raise ValueError(
            "nperseg must not exceed the number of signal samples."
        )

    if noverlap is None:
        noverlap = nperseg // 2

    if noverlap < 0:
        raise ValueError(
            "noverlap must be non-negative."
        )

    if noverlap >= nperseg:
        raise ValueError(
            "noverlap must be smaller than nperseg."
        )

    frequencies_hz, psd = welch(
        record.samples,
        fs=record.sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        return_onesided=True,
    )

    frequency_resolution_hz = (
        record.sample_rate_hz / nperseg
    )

    integrated_power = float(
        np.trapezoid(
            psd,
            frequencies_hz,
        )
    )

    return PSDResult(
        frequencies_hz=frequencies_hz,
        psd=psd,
        sample_rate_hz=record.sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        frequency_resolution_hz=frequency_resolution_hz,
        integrated_power=integrated_power,
    )