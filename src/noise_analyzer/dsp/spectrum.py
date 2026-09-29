from dataclasses import dataclass

import numpy as np

from noise_analyzer.io.validation import (
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord


@dataclass(frozen=True)
class SpectrumResult:
    """One-sided amplitude spectrum for a real-valued signal."""

    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    frequency_resolution_hz: float
    number_of_samples: int
    sample_rate_hz: float
    window: str | None
    coherent_gain: float


def calculate_amplitude_spectrum(
    record: SignalRecord,
    window: str | None = None,
) -> SpectrumResult:
    """Calculate a one-sided amplitude spectrum."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    number_of_samples = record.number_of_samples

    if number_of_samples < 2:
        raise ValueError(
            "At least two samples are required for spectral analysis."
        )

    sample_rate_hz = record.sample_rate_hz
    samples = record.samples

    if window is None:
        window_values = np.ones(number_of_samples, dtype=float)
        coherent_gain = 1.0

    elif window == "hann":
        window_values = np.hanning(number_of_samples)
        coherent_gain = float(np.mean(window_values))

    else:
        raise ValueError(
            "Unsupported window. Use None or 'hann'."
        )

    windowed_samples = samples * window_values

    fft_values = np.fft.rfft(windowed_samples)

    frequencies_hz = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sample_rate_hz,
    )

    amplitudes = (
        np.abs(fft_values)
        / number_of_samples
        / coherent_gain
    )

    if number_of_samples % 2 == 0:
        amplitudes[1:-1] *= 2.0
    else:
        amplitudes[1:] *= 2.0

    frequency_resolution_hz = (
        sample_rate_hz / number_of_samples
    )

    return SpectrumResult(
        frequencies_hz=frequencies_hz,
        amplitudes=amplitudes,
        frequency_resolution_hz=frequency_resolution_hz,
        number_of_samples=number_of_samples,
        sample_rate_hz=sample_rate_hz,
        window=window,
        coherent_gain=coherent_gain,
    )