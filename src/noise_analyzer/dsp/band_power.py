from dataclasses import dataclass

import numpy as np

from noise_analyzer.dsp.psd import PSDResult


@dataclass(frozen=True)
class BandPowerResult:
    """Power contained within one frequency band."""

    lower_frequency_hz: float
    upper_frequency_hz: float
    band_power: float
    fraction_of_total_power: float


def calculate_band_power(
    psd_result: PSDResult,
    lower_frequency_hz: float,
    upper_frequency_hz: float,
) -> BandPowerResult:
    """Calculate integrated PSD power within a selected frequency band."""

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

    frequencies = psd_result.frequencies_hz
    psd = psd_result.psd

    interior_mask = (
        (frequencies > lower_frequency_hz)
        & (frequencies < upper_frequency_hz)
    )

    interior_frequencies = frequencies[interior_mask]
    interior_psd = psd[interior_mask]

    lower_psd = np.interp(
        lower_frequency_hz,
        frequencies,
        psd,
    )

    upper_psd = np.interp(
        upper_frequency_hz,
        frequencies,
        psd,
    )

    band_frequencies = np.concatenate(
        (
            [lower_frequency_hz],
            interior_frequencies,
            [upper_frequency_hz],
        )
    )

    band_psd = np.concatenate(
        (
            [lower_psd],
            interior_psd,
            [upper_psd],
        )
    )

    band_power = float(
        np.trapezoid(
            band_psd,
            band_frequencies,
        )
    )

    if psd_result.integrated_power > 0:
        fraction_of_total_power = (
            band_power / psd_result.integrated_power
        )
    else:
        fraction_of_total_power = 0.0

    return BandPowerResult(
        lower_frequency_hz=lower_frequency_hz,
        upper_frequency_hz=upper_frequency_hz,
        band_power=band_power,
        fraction_of_total_power=float(
            fraction_of_total_power
        ),
    )