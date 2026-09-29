import numpy as np
import pytest

from noise_analyzer.dsp.peaks import detect_spectral_peaks
from noise_analyzer.dsp.spectrum import calculate_amplitude_spectrum
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import (
    generate_sine,
    generate_two_tone,
)

def test_peak_detector_finds_two_known_tones():
    _, samples = generate_two_tone(
        frequency_1_hz=1000.0,
        amplitude_1=1.0,
        frequency_2_hz=1800.0,
        amplitude_2=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window=None,
    )

    result = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
    )

    detected_frequencies = np.array(
        [peak.frequency_hz for peak in result.peaks]
    )

    assert len(result.peaks) == 2

    assert np.any(
        np.isclose(
            detected_frequencies,
            1000.0,
            atol=spectrum.frequency_resolution_hz,
        )
    )

    assert np.any(
        np.isclose(
            detected_frequencies,
            1800.0,
            atol=spectrum.frequency_resolution_hz,
        )
    )

def test_peak_detector_reports_correct_two_tone_amplitudes():
    _, samples = generate_two_tone(
        frequency_1_hz=1000.0,
        amplitude_1=1.0,
        frequency_2_hz=1800.0,
        amplitude_2=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window=None,
    )

    result = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
    )

    peak_1000 = min(
        result.peaks,
        key=lambda peak: abs(
            peak.frequency_hz - 1000.0
        ),
    )

    peak_1800 = min(
        result.peaks,
        key=lambda peak: abs(
            peak.frequency_hz - 1800.0
        ),
    )

    assert np.isclose(
        peak_1000.amplitude,
        1.0,
        atol=1e-12,
    )

    assert np.isclose(
        peak_1800.amplitude,
        0.5,
        atol=1e-12,
    )

def test_peak_detector_finds_single_reference_tone():
    _, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=2.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    spectrum = calculate_amplitude_spectrum(record)

    result = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
    )

    assert len(result.peaks) == 1

    assert np.isclose(
        result.peaks[0].frequency_hz,
        1000.0,
        atol=spectrum.frequency_resolution_hz,
    )

def test_peak_detector_rejects_negative_prominence():
    _, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=1.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    spectrum = calculate_amplitude_spectrum(record)

    with pytest.raises(
        ValueError,
        match="min_prominence must be non-negative",
    ):
        detect_spectral_peaks(
            spectrum,
            min_prominence=-0.1,
        )