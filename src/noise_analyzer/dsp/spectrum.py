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


def calculate_amplitude_spectrum(
    record: SignalRecord,
) -> SpectrumResult:
    """Calculate the one-sided amplitude spectrum of a real-valued signal."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    samples = record.samples
    number_of_samples = record.number_of_samples
    sample_rate_hz = record.sample_rate_hz

    if number_of_samples < 2:
        raise ValueError(
            "At least two samples are required for spectral analysis."
        )

    fft_values = np.fft.rfft(samples)

    frequencies_hz = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sample_rate_hz,
    )

    amplitudes = np.abs(fft_values) / number_of_samples

    if number_of_samples % 2 == 0:
        # Do not double DC or the Nyquist bin.
        amplitudes[1:-1] *= 2.0
    else:
        # Odd-length signals do not contain a Nyquist bin.
        amplitudes[1:] *= 2.0

    frequency_resolution_hz = sample_rate_hz / number_of_samples

    return SpectrumResult(
        frequencies_hz=frequencies_hz,
        amplitudes=amplitudes,
        frequency_resolution_hz=frequency_resolution_hz,
        number_of_samples=number_of_samples,
        sample_rate_hz=sample_rate_hz,
    )