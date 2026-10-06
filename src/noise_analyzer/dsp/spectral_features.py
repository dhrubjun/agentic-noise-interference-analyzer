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

@dataclass(frozen=True)
class OccupiedBandwidthResult:
    occupied_bandwidth_hz: float
    lower_edge_hz: float
    upper_edge_hz: float
    power_fraction: float


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

def calculate_occupied_bandwidth(
    psd_result: PSDResult,
    power_fraction: float = 0.90,
) -> OccupiedBandwidthResult:
    """
    Calculate the frequency interval containing a chosen
    fraction of the total integrated spectral power.

    For power_fraction=0.90, 5% of the power is excluded
    from each side of the spectrum.
    """

    if not 0.0 < power_fraction < 1.0:
        raise ValueError(
            "power_fraction must be between 0 and 1."
        )

    frequencies = np.asarray(
        psd_result.frequencies_hz,
        dtype=float,
    )

    psd_values = np.asarray(
        psd_result.psd,
        dtype=float,
    )

    if frequencies.size < 2:
        raise ValueError(
            "At least two PSD frequency bins are required."
        )

    if np.any(psd_values < 0):
        raise ValueError(
            "PSD values must be non-negative."
        )

    # Power contained between each neighbouring
    # pair of frequency bins.
    frequency_steps = np.diff(frequencies)

    segment_powers = (
        0.5
        * (
            psd_values[:-1]
            + psd_values[1:]
        )
        * frequency_steps
    )

    total_power = float(
        np.sum(segment_powers)
    )

    if total_power <= 0:
        raise ValueError(
            "PSD contains no positive integrated power."
        )

    cumulative_power = np.concatenate(
        (
            [0.0],
            np.cumsum(segment_powers),
        )
    )

    excluded_fraction = (
        1.0 - power_fraction
    )

    lower_target = (
        excluded_fraction
        / 2.0
        * total_power
    )

    upper_target = (
        1.0
        - excluded_fraction / 2.0
    ) * total_power

    lower_edge_hz = float(
        np.interp(
            lower_target,
            cumulative_power,
            frequencies,
        )
    )

    upper_edge_hz = float(
        np.interp(
            upper_target,
            cumulative_power,
            frequencies,
        )
    )

    return OccupiedBandwidthResult(
        occupied_bandwidth_hz=(
            upper_edge_hz
            - lower_edge_hz
        ),
        lower_edge_hz=lower_edge_hz,
        upper_edge_hz=upper_edge_hz,
        power_fraction=float(power_fraction),
    )