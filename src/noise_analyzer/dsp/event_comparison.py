from dataclasses import dataclass

from noise_analyzer.dsp.band_power import calculate_band_power
from noise_analyzer.dsp.psd import calculate_welch_psd
from noise_analyzer.dsp.spectral_features import (
    calculate_spectral_centroid,
    calculate_spectral_flatness,
    calculate_spectral_spread,
)
from noise_analyzer.models.signal import SignalRecord


@dataclass(frozen=True)
class SpectralComparisonResult:
    """Spectral comparison between baseline and event signal segments."""

    baseline_total_power: float
    event_total_power: float
    total_power_ratio: float

    baseline_band_power: float
    event_band_power: float
    band_power_ratio: float

    baseline_centroid_hz: float
    event_centroid_hz: float
    centroid_shift_hz: float

    baseline_spread_hz: float
    event_spread_hz: float
    spread_change_hz: float

    baseline_flatness: float
    event_flatness: float
    flatness_change: float

    lower_frequency_hz: float
    upper_frequency_hz: float


def compare_event_to_baseline_spectrum(
    baseline_record: SignalRecord,
    event_record: SignalRecord,
    lower_frequency_hz: float,
    upper_frequency_hz: float,
    nperseg: int,
    window: str = "hann",
) -> SpectralComparisonResult:
    """Compare validated spectral measurements for baseline and event segments."""

    if baseline_record.sample_rate_hz != event_record.sample_rate_hz:
        raise ValueError(
            "baseline_record and event_record must have the same sample rate."
        )

    baseline_psd = calculate_welch_psd(
        baseline_record,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window=window,
    )

    event_psd = calculate_welch_psd(
        event_record,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window=window,
    )

    baseline_band = calculate_band_power(
        baseline_psd,
        lower_frequency_hz=lower_frequency_hz,
        upper_frequency_hz=upper_frequency_hz,
    )

    event_band = calculate_band_power(
        event_psd,
        lower_frequency_hz=lower_frequency_hz,
        upper_frequency_hz=upper_frequency_hz,
    )

    baseline_centroid = calculate_spectral_centroid(
        baseline_psd
    )

    event_centroid = calculate_spectral_centroid(
        event_psd
    )

    baseline_spread = calculate_spectral_spread(
        baseline_psd
    )

    event_spread = calculate_spectral_spread(
        event_psd
    )

    baseline_flatness = calculate_spectral_flatness(
        baseline_psd
    )

    event_flatness = calculate_spectral_flatness(
        event_psd
    )

    if baseline_psd.integrated_power > 0:
        total_power_ratio = (
            event_psd.integrated_power
            / baseline_psd.integrated_power
        )
    else:
        total_power_ratio = float("inf")

    if baseline_band.band_power > 0:
        band_power_ratio = (
            event_band.band_power
            / baseline_band.band_power
        )
    else:
        band_power_ratio = float("inf")

    return SpectralComparisonResult(
        baseline_total_power=float(
            baseline_psd.integrated_power
        ),
        event_total_power=float(
            event_psd.integrated_power
        ),
        total_power_ratio=float(total_power_ratio),

        baseline_band_power=float(
            baseline_band.band_power
        ),
        event_band_power=float(
            event_band.band_power
        ),
        band_power_ratio=float(band_power_ratio),

        baseline_centroid_hz=float(
            baseline_centroid.spectral_centroid_hz
        ),
        event_centroid_hz=float(
            event_centroid.spectral_centroid_hz
        ),
        centroid_shift_hz=float(
            event_centroid.spectral_centroid_hz
            - baseline_centroid.spectral_centroid_hz
        ),

        baseline_spread_hz=float(
            baseline_spread.spectral_spread_hz
        ),
        event_spread_hz=float(
            event_spread.spectral_spread_hz
        ),
        spread_change_hz=float(
            event_spread.spectral_spread_hz
            - baseline_spread.spectral_spread_hz
        ),

        baseline_flatness=float(
            baseline_flatness.spectral_flatness
        ),
        event_flatness=float(
            event_flatness.spectral_flatness
        ),
        flatness_change=float(
            event_flatness.spectral_flatness
            - baseline_flatness.spectral_flatness
        ),

        lower_frequency_hz=float(lower_frequency_hz),
        upper_frequency_hz=float(upper_frequency_hz),
    )