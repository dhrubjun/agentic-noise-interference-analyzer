import numpy as np
import pytest

from noise_analyzer.dsp.harmonics import (
    detect_harmonic_family,
)
from noise_analyzer.dsp.peaks import (
    detect_spectral_peaks,
)
from noise_analyzer.dsp.spectrum import (
    calculate_amplitude_spectrum,
)
from noise_analyzer.models.signal import SignalRecord

from noise_analyzer.dsp.harmonics import (
    detect_harmonic_family,
    find_best_harmonic_family,
)


def test_detects_known_harmonic_family():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    samples = (
        1.0 * np.sin(
            2.0 * np.pi * 50.0 * time
        )
        + 0.8 * np.sin(
            2.0 * np.pi * 100.0 * time
        )
        + 0.6 * np.sin(
            2.0 * np.pi * 150.0 * time
        )
        + 0.5 * np.sin(
            2.0 * np.pi * 200.0 * time
        )
        + 0.4 * np.sin(
            2.0 * np.pi * 250.0 * time
        )
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window="hann",
    )

    peaks = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
        min_distance_hz=20.0,
    )

    result = detect_harmonic_family(
        peaks,
        candidate_fundamental_hz=50.0,
        tolerance_hz=1.0,
        max_harmonic_order=5,
    )

    assert len(result.matches) == 5

    assert [
        match.harmonic_order
        for match in result.matches
    ] == [1, 2, 3, 4, 5]

def test_harmonic_detection_allows_frequency_tolerance():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    samples = (
        np.sin(
            2.0 * np.pi * 50.2 * time
        )
        + np.sin(
            2.0 * np.pi * 99.8 * time
        )
        + np.sin(
            2.0 * np.pi * 150.3 * time
        )
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window="hann",
    )

    peaks = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
        min_distance_hz=20.0,
    )

    result = detect_harmonic_family(
        peaks,
        candidate_fundamental_hz=50.0,
        tolerance_hz=0.5,
        max_harmonic_order=3,
    )

    assert len(result.matches) == 3

def test_nonharmonic_peaks_do_not_form_full_family():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    samples = (
        np.sin(
            2.0 * np.pi * 47.0 * time
        )
        + np.sin(
            2.0 * np.pi * 113.0 * time
        )
        + np.sin(
            2.0 * np.pi * 181.0 * time
        )
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window="hann",
    )

    peaks = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
        min_distance_hz=20.0,
    )

    result = detect_harmonic_family(
        peaks,
        candidate_fundamental_hz=50.0,
        tolerance_hz=1.0,
        max_harmonic_order=5,
    )

    assert len(result.matches) < 3

def test_harmonic_detection_rejects_invalid_fundamental():
    from noise_analyzer.dsp.peaks import (
        PeakDetectionResult,
    )

    empty_peaks = PeakDetectionResult(
        peaks=(),
        min_prominence=0.1,
        min_distance_hz=None,
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        detect_harmonic_family(
            empty_peaks,
            candidate_fundamental_hz=0.0,
            tolerance_hz=1.0,
        )

def test_automatic_search_detects_missing_fundamental():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    samples = (
        np.sin(
            2.0 * np.pi * 100.0 * time
        )
        + np.sin(
            2.0 * np.pi * 150.0 * time
        )
        + np.sin(
            2.0 * np.pi * 200.0 * time
        )
        + np.sin(
            2.0 * np.pi * 250.0 * time
        )
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window="hann",
    )

    peaks = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
        min_distance_hz=20.0,
    )

    result = find_best_harmonic_family(
        peaks,
        tolerance_hz=1.0,
        max_harmonic_order=6,
        minimum_matches=3,
    )

    assert result.best_family is not None

    assert np.isclose(
        result.best_family.candidate_fundamental_hz,
        50.0,
        atol=1.0,
    )

    assert len(
        result.best_family.matches
    ) >= 4

def test_automatic_search_rejects_weak_nonharmonic_structure():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    samples = (
        np.sin(
            2.0 * np.pi * 470.0 * time
        )
        + np.sin(
            2.0 * np.pi * 913.0 * time
        )
        + np.sin(
            2.0 * np.pi * 1720.0 * time
        )
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window="hann",
    )

    peaks = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
        min_distance_hz=20.0,
    )

    result = find_best_harmonic_family(
        peaks,
        tolerance_hz=1.0,
        max_harmonic_order=10,
        minimum_matches=3,
    )

    assert result.best_family is None