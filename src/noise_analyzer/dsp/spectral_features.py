from dataclasses import dataclass

import numpy as np

from noise_analyzer.dsp.psd import PSDResult


@dataclass(frozen=True)
class SpectralFlatnessResult:
    """Spectral flatness calculated from PSD values."""

    spectral_flatness: float
    lower_frequency_hz: float
    upper_frequency_hz: float
    number_of_bins: int

@dataclass(frozen=True)
class SpectralCentroidResult:
    spectral_centroid_hz: float
    lower_frequency_hz: float
    upper_frequency_hz: float
    number_of_bins: int


@dataclass(frozen=True)
class SpectralSpreadResult:
    spectral_spread_hz: float
    lower_frequency_hz: float
    upper_frequency_hz: float
    number_of_bins: int

@dataclass(frozen=True)
class PSDPercentilesResult:
    percentile_10_db: float
    percentile_25_db: float
    percentile_50_db: float
    percentile_75_db: float
    percentile_90_db: float
    lower_frequency_hz: float
    upper_frequency_hz: float
    number_of_bins: int


def calculate_spectral_flatness(
    psd_result: PSDResult,
    lower_frequency_hz: float | None = None,
    upper_frequency_hz: float | None = None,
) -> SpectralFlatnessResult:
    """Calculate spectral flatness from PSD values."""

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

    geometric_mean = float(
        np.exp(
            np.mean(
                np.log(positive_psd)
            )
        )
    )

    arithmetic_mean = float(
        np.mean(positive_psd)
    )

    spectral_flatness = (
        geometric_mean / arithmetic_mean
    )

    return SpectralFlatnessResult(
        spectral_flatness=float(spectral_flatness),
        lower_frequency_hz=float(lower_frequency_hz),
        upper_frequency_hz=float(upper_frequency_hz),
        number_of_bins=int(positive_psd.size),
    )

def calculate_spectral_centroid(
    psd_result: PSDResult,
    lower_frequency_hz: float | None = None,
    upper_frequency_hz: float | None = None,
) -> SpectralCentroidResult:
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

    if upper_frequency_hz > psd_result.sample_rate_hz / 2.0:
        raise ValueError(
            "upper_frequency_hz must not exceed the Nyquist frequency."
        )

    mask = (
        (frequencies >= lower_frequency_hz)
        & (frequencies <= upper_frequency_hz)
    )

    selected_frequencies = frequencies[mask]
    selected_psd = psd_values[mask]

    total_power = float(np.sum(selected_psd))

    if total_power <= 0:
        raise ValueError(
            "Selected frequency range contains no positive spectral power."
        )

    centroid_hz = float(
        np.sum(
            selected_frequencies * selected_psd
        )
        / total_power
    )

    return SpectralCentroidResult(
        spectral_centroid_hz=centroid_hz,
        lower_frequency_hz=float(lower_frequency_hz),
        upper_frequency_hz=float(upper_frequency_hz),
        number_of_bins=int(selected_psd.size),
    )

def calculate_spectral_spread(
    psd_result: PSDResult,
    lower_frequency_hz: float | None = None,
    upper_frequency_hz: float | None = None,
) -> SpectralSpreadResult:
    centroid = calculate_spectral_centroid(
        psd_result,
        lower_frequency_hz=lower_frequency_hz,
        upper_frequency_hz=upper_frequency_hz,
    )

    frequencies = psd_result.frequencies_hz
    psd_values = psd_result.psd

    mask = (
        (frequencies >= centroid.lower_frequency_hz)
        & (frequencies <= centroid.upper_frequency_hz)
    )

    selected_frequencies = frequencies[mask]
    selected_psd = psd_values[mask]

    total_power = float(np.sum(selected_psd))

    variance_hz_squared = float(
        np.sum(
            (
                selected_frequencies
                - centroid.spectral_centroid_hz
            ) ** 2
            * selected_psd
        )
        / total_power
    )

    spread_hz = float(
        np.sqrt(variance_hz_squared)
    )

    return SpectralSpreadResult(
        spectral_spread_hz=spread_hz,
        lower_frequency_hz=centroid.lower_frequency_hz,
        upper_frequency_hz=centroid.upper_frequency_hz,
        number_of_bins=int(selected_psd.size),
    )

def calculate_psd_percentiles(
    psd_result: PSDResult,
    lower_frequency_hz: float | None = None,
    upper_frequency_hz: float | None = None,
) -> PSDPercentilesResult:
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

    if upper_frequency_hz > psd_result.sample_rate_hz / 2.0:
        raise ValueError(
            "upper_frequency_hz must not exceed the Nyquist frequency."
        )

    mask = (
        (frequencies >= lower_frequency_hz)
        & (frequencies <= upper_frequency_hz)
    )

    selected_psd = psd_values[mask]

    positive_psd = selected_psd[selected_psd > 0]

    if positive_psd.size == 0:
        raise ValueError(
            "Selected frequency range contains no positive PSD values."
        )

    psd_db = 10.0 * np.log10(positive_psd)

    return PSDPercentilesResult(
        percentile_10_db=float(np.percentile(psd_db, 10)),
        percentile_25_db=float(np.percentile(psd_db, 25)),
        percentile_50_db=float(np.percentile(psd_db, 50)),
        percentile_75_db=float(np.percentile(psd_db, 75)),
        percentile_90_db=float(np.percentile(psd_db, 90)),
        lower_frequency_hz=float(lower_frequency_hz),
        upper_frequency_hz=float(upper_frequency_hz),
        number_of_bins=int(positive_psd.size),
    )