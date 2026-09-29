from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from noise_analyzer.dsp.spectrum import SpectrumResult


@dataclass(frozen=True)
class SpectralPeak:
    """One detected spectral peak."""

    frequency_hz: float
    amplitude: float
    prominence: float


@dataclass(frozen=True)
class PeakDetectionResult:
    """Collection of detected spectral peaks."""

    peaks: tuple[SpectralPeak, ...]
    min_prominence: float
    min_distance_hz: float | None


def detect_spectral_peaks(
    spectrum: SpectrumResult,
    min_prominence: float,
    min_distance_hz: float | None = None,
) -> PeakDetectionResult:
    """Detect significant local peaks in an amplitude spectrum."""

    if min_prominence < 0:
        raise ValueError(
            "min_prominence must be non-negative."
        )

    if min_distance_hz is not None and min_distance_hz < 0:
        raise ValueError(
            "min_distance_hz must be non-negative."
        )

    find_peaks_kwargs = {
        "prominence": min_prominence,
    }

    if min_distance_hz is not None:
        distance_bins = max(
            1,
            int(
                np.ceil(
                    min_distance_hz
                    / spectrum.frequency_resolution_hz
                )
            ),
        )

        find_peaks_kwargs["distance"] = distance_bins

    peak_indices, properties = find_peaks(
        spectrum.amplitudes,
        **find_peaks_kwargs,
    )

    prominences = properties["prominences"]

    detected_peaks = tuple(
        SpectralPeak(
            frequency_hz=float(
                spectrum.frequencies_hz[index]
            ),
            amplitude=float(
                spectrum.amplitudes[index]
            ),
            prominence=float(prominence),
        )
        for index, prominence in zip(
            peak_indices,
            prominences,
        )
    )

    return PeakDetectionResult(
        peaks=detected_peaks,
        min_prominence=min_prominence,
        min_distance_hz=min_distance_hz,
    )