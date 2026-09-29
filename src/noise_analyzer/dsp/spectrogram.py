from dataclasses import dataclass

import numpy as np
from scipy.signal import spectrogram

from noise_analyzer.io.validation import (
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord


@dataclass(frozen=True)
class SpectrogramResult:
    """Time-frequency spectrogram result."""

    frequencies_hz: np.ndarray
    times_seconds: np.ndarray
    power_spectral_density: np.ndarray
    sample_rate_hz: float
    window: str
    nperseg: int
    noverlap: int
    frequency_resolution_hz: float
    time_step_seconds: float


def calculate_spectrogram(
    record: SignalRecord,
    nperseg: int,
    noverlap: int | None = None,
    window: str = "hann",
) -> SpectrogramResult:
    """Calculate a one-sided PSD spectrogram."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    number_of_samples = record.number_of_samples

    if number_of_samples < 2:
        raise ValueError(
            "At least two samples are required for spectrogram analysis."
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

    frequencies_hz, times_seconds, psd = spectrogram(
        record.samples,
        fs=record.sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )

    frequency_resolution_hz = (
        record.sample_rate_hz / nperseg
    )

    hop_size_samples = nperseg - noverlap

    time_step_seconds = (
        hop_size_samples / record.sample_rate_hz
    )

    return SpectrogramResult(
        frequencies_hz=frequencies_hz,
        times_seconds=times_seconds,
        power_spectral_density=psd,
        sample_rate_hz=record.sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        frequency_resolution_hz=frequency_resolution_hz,
        time_step_seconds=time_step_seconds,
    )