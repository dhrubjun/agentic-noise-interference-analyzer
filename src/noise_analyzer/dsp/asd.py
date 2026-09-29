from dataclasses import dataclass

import numpy as np

from noise_analyzer.dsp.psd import (
    PSDResult,
    calculate_welch_psd,
)
from noise_analyzer.models.signal import SignalRecord


@dataclass(frozen=True)
class ASDResult:
    """Amplitude spectral density result."""

    frequencies_hz: np.ndarray
    asd: np.ndarray
    sample_rate_hz: float
    window: str
    nperseg: int
    noverlap: int
    frequency_resolution_hz: float


def calculate_asd(
    record: SignalRecord,
    nperseg: int,
    noverlap: int | None = None,
    window: str = "hann",
) -> ASDResult:
    """Calculate amplitude spectral density from Welch PSD."""

    psd_result: PSDResult = calculate_welch_psd(
        record=record,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )

    asd = np.sqrt(psd_result.psd)

    return ASDResult(
        frequencies_hz=psd_result.frequencies_hz,
        asd=asd,
        sample_rate_hz=psd_result.sample_rate_hz,
        window=psd_result.window,
        nperseg=psd_result.nperseg,
        noverlap=psd_result.noverlap,
        frequency_resolution_hz=psd_result.frequency_resolution_hz,
    )