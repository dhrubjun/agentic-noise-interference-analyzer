import numpy as np
import pytest

from noise_analyzer.dsp.narrowband import (
    characterize_narrowband_line,
)
from noise_analyzer.dsp.psd import (
    calculate_welch_psd,
)
from noise_analyzer.models.signal import SignalRecord

from noise_analyzer.dsp.peaks import detect_spectral_peaks
from noise_analyzer.dsp.spectrum import calculate_amplitude_spectrum
from noise_analyzer.dsp.narrowband import characterize_detected_lines

def calculate_test_psd(
    samples: np.ndarray,
    sample_rate_hz: float = 10000.0,
):
    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    return calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
        window="hann",
    )


def test_line_to_floor_increases_with_tone_amplitude():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_samples,
    )

    results = []

    for amplitude in (0.5, 1.0, 2.0):
        tone = amplitude * np.sin(
            2.0 * np.pi * 1000.0 * time
        )

        psd = calculate_test_psd(
            noise + tone,
            sample_rate_hz,
        )

        result = characterize_narrowband_line(
            psd,
            line_frequency_hz=1000.0,
            neighbourhood_width_hz=100.0,
            excluded_peak_width_hz=5.0,
        )

        results.append(
            result.line_to_floor_db
        )

    assert (
        results[0]
        < results[1]
        < results[2]
    )

def test_line_to_floor_decreases_with_noise_level():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    base_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_samples,
    )

    tone = np.sin(
        2.0 * np.pi * 1000.0 * time
    )

    results = []

    for noise_scale in (0.5, 1.0, 2.0):
        samples = (
            noise_scale * base_noise
            + tone
        )

        psd = calculate_test_psd(
            samples,
            sample_rate_hz,
        )

        result = characterize_narrowband_line(
            psd,
            line_frequency_hz=1000.0,
            neighbourhood_width_hz=100.0,
            excluded_peak_width_hz=5.0,
        )

        results.append(
            result.line_to_floor_db
        )

    assert (
        results[0]
        > results[1]
        > results[2]
    )

def test_narrowband_line_reports_correct_frequency():
    sample_rate_hz = 10000.0

    time = (
        np.arange(100000)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    samples = (
        rng.normal(
            size=time.size
        )
        + 3.0
        * np.sin(
            2.0 * np.pi * 1200.0 * time
        )
    )

    psd = calculate_test_psd(
        samples,
        sample_rate_hz,
    )

    result = characterize_narrowband_line(
        psd,
        line_frequency_hz=1200.0,
    )

    assert np.isclose(
        result.frequency_hz,
        1200.0,
        atol=psd.frequency_resolution_hz,
    )

    assert result.line_to_floor_db > 0
    assert result.number_of_background_bins > 0

def test_narrowband_line_rejects_invalid_exclusion_width():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    psd = calculate_test_psd(samples)

    with pytest.raises(
        ValueError,
        match="smaller than neighbourhood_width_hz",
    ):
        characterize_narrowband_line(
            psd,
            line_frequency_hz=1000.0,
            neighbourhood_width_hz=50.0,
            excluded_peak_width_hz=50.0,
        )

def test_characterizes_detected_lines():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = np.arange(
        number_of_samples
    ) / sample_rate_hz

    rng = np.random.default_rng(42)

    noise = rng.normal(
        scale=0.5,
        size=number_of_samples,
    )

    samples = (
        noise
        + 2.0 * np.sin(
            2.0 * np.pi * 1000.0 * time
        )
        + 1.5 * np.sin(
            2.0 * np.pi * 2000.0 * time
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
        min_prominence=0.5,
        min_distance_hz=100.0,
    )

    psd = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
        window="hann",
    )

    result = characterize_detected_lines(
        psd,
        peaks,
        neighbourhood_width_hz=100.0,
        excluded_peak_width_hz=5.0,
    )

    assert len(result.lines) >= 2

    frequencies = [
        line.frequency_hz
        for line in result.lines
    ]

    assert any(
        np.isclose(
            frequency,
            1000.0,
            atol=psd.frequency_resolution_hz,
        )
        for frequency in frequencies
    )

    assert any(
        np.isclose(
            frequency,
            2000.0,
            atol=psd.frequency_resolution_hz,
        )
        for frequency in frequencies
    )