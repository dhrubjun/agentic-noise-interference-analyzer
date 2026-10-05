from dataclasses import dataclass

import numpy as np

from noise_analyzer.dsp.psd import PSDResult


@dataclass(frozen=True)
class NoiseFloorResult:
    """Estimated broadband noise floor from a PSD."""

    noise_floor_db: float
    method: str
    lower_frequency_hz: float
    upper_frequency_hz: float
    number_of_bins: int


def estimate_noise_floor(
    psd_result: PSDResult,
    lower_frequency_hz: float | None = None,
    upper_frequency_hz: float | None = None,
) -> NoiseFloorResult:
    """Estimate the global noise floor using the median PSD level in dB."""

    frequencies = psd_result.frequencies_hz
    psd_values = psd_result.psd

    if lower_frequency_hz is None:
        lower_frequency_hz = float(frequencies[0])

    if upper_frequency_hz is None:
        upper_frequency_hz = float(frequencies[-1])

    if lower_frequency_hz < 0:
        raise ValueError(
            "lower_frequency_hz must be non-negative."
        )

    if upper_frequency_hz <= lower_frequency_hz:
        raise ValueError(
            "upper_frequency_hz must be greater than lower_frequency_hz."
        )

    nyquist_frequency_hz = psd_result.sample_rate_hz / 2.0

    if upper_frequency_hz > nyquist_frequency_hz:
        raise ValueError(
            "upper_frequency_hz must not exceed the Nyquist frequency."
        )

    frequency_mask = (
        (frequencies >= lower_frequency_hz)
        & (frequencies <= upper_frequency_hz)
    )

    selected_psd = psd_values[frequency_mask]

    positive_psd = selected_psd[selected_psd > 0]

    if positive_psd.size == 0:
        raise ValueError(
            "Selected frequency range contains no positive PSD values."
        )

    psd_db = 10.0 * np.log10(positive_psd)

    noise_floor_db = float(
        np.median(psd_db)
    )

    return NoiseFloorResult(
        noise_floor_db=noise_floor_db,
        method="median_psd_db",
        lower_frequency_hz=float(lower_frequency_hz),
        upper_frequency_hz=float(upper_frequency_hz),
        number_of_bins=int(positive_psd.size),
    )