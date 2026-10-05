from dataclasses import dataclass

import numpy as np

from noise_analyzer.dsp.psd import PSDResult

from noise_analyzer.dsp.peaks import PeakDetectionResult


@dataclass(frozen=True)
class NarrowbandLineResult:
    """Characterization of one narrowband spectral line."""

    frequency_hz: float
    peak_level_db: float
    local_noise_floor_db: float
    line_to_floor_db: float
    neighbourhood_lower_hz: float
    neighbourhood_upper_hz: float
    excluded_peak_width_hz: float
    number_of_background_bins: int

@dataclass(frozen=True)
class NarrowbandCharacterizationResult:
    """Characterization results for detected narrowband spectral lines."""

    lines: tuple[NarrowbandLineResult, ...]
    neighbourhood_width_hz: float
    excluded_peak_width_hz: float


def characterize_narrowband_line(
    psd_result: PSDResult,
    line_frequency_hz: float,
    neighbourhood_width_hz: float = 100.0,
    excluded_peak_width_hz: float = 5.0,
) -> NarrowbandLineResult:
    """Characterize one spectral line relative to its local background."""

    if line_frequency_hz < 0:
        raise ValueError(
            "line_frequency_hz must be non-negative."
        )

    nyquist_frequency_hz = psd_result.sample_rate_hz / 2.0

    if line_frequency_hz > nyquist_frequency_hz:
        raise ValueError(
            "line_frequency_hz must not exceed the Nyquist frequency."
        )

    if neighbourhood_width_hz <= 0:
        raise ValueError(
            "neighbourhood_width_hz must be greater than zero."
        )

    if excluded_peak_width_hz < 0:
        raise ValueError(
            "excluded_peak_width_hz must be non-negative."
        )

    if excluded_peak_width_hz >= neighbourhood_width_hz:
        raise ValueError(
            "excluded_peak_width_hz must be smaller than neighbourhood_width_hz."
        )

    frequencies = psd_result.frequencies_hz
    psd_values = psd_result.psd

    peak_index = int(
        np.argmin(
            np.abs(frequencies - line_frequency_hz)
        )
    )

    actual_peak_frequency_hz = float(
        frequencies[peak_index]
    )

    peak_psd = float(
        psd_values[peak_index]
    )

    if peak_psd <= 0:
        raise ValueError(
            "Selected spectral line has non-positive PSD."
        )

    peak_level_db = float(
        10.0 * np.log10(peak_psd)
    )

    neighbourhood_lower_hz = max(
        0.0,
        actual_peak_frequency_hz - neighbourhood_width_hz,
    )

    neighbourhood_upper_hz = min(
        nyquist_frequency_hz,
        actual_peak_frequency_hz + neighbourhood_width_hz,
    )

    neighbourhood_mask = (
        (frequencies >= neighbourhood_lower_hz)
        & (frequencies <= neighbourhood_upper_hz)
    )

    exclusion_mask = (
        np.abs(
            frequencies - actual_peak_frequency_hz
        )
        <= excluded_peak_width_hz
    )

    background_mask = (
        neighbourhood_mask
        & ~exclusion_mask
    )

    background_psd = psd_values[background_mask]

    positive_background_psd = (
        background_psd[background_psd > 0]
    )

    if positive_background_psd.size == 0:
        raise ValueError(
            "No positive background PSD bins remain after peak exclusion."
        )

    background_db = (
        10.0 * np.log10(
            positive_background_psd
        )
    )

    local_noise_floor_db = float(
        np.median(background_db)
    )

    line_to_floor_db = (
        peak_level_db - local_noise_floor_db
    )

    return NarrowbandLineResult(
        frequency_hz=actual_peak_frequency_hz,
        peak_level_db=peak_level_db,
        local_noise_floor_db=local_noise_floor_db,
        line_to_floor_db=float(line_to_floor_db),
        neighbourhood_lower_hz=float(
            neighbourhood_lower_hz
        ),
        neighbourhood_upper_hz=float(
            neighbourhood_upper_hz
        ),
        excluded_peak_width_hz=float(
            excluded_peak_width_hz
        ),
        number_of_background_bins=int(
            positive_background_psd.size
        ),
    )

def characterize_detected_lines(
    psd_result: PSDResult,
    peaks: PeakDetectionResult,
    neighbourhood_width_hz: float = 100.0,
    excluded_peak_width_hz: float = 5.0,
) -> NarrowbandCharacterizationResult:
    """Characterize all detected spectral peaks against local PSD background."""

    lines = tuple(
        characterize_narrowband_line(
            psd_result,
            line_frequency_hz=peak.frequency_hz,
            neighbourhood_width_hz=neighbourhood_width_hz,
            excluded_peak_width_hz=excluded_peak_width_hz,
        )
        for peak in peaks.peaks
    )

    return NarrowbandCharacterizationResult(
        lines=lines,
        neighbourhood_width_hz=neighbourhood_width_hz,
        excluded_peak_width_hz=excluded_peak_width_hz,
    )