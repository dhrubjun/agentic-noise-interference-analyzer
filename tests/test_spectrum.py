import numpy as np
import pytest

from noise_analyzer.dsp.spectrum import calculate_amplitude_spectrum
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import (
    generate_sine,
    generate_two_tone,
)


def test_amplitude_spectrum_for_reference_sine():
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

    result = calculate_amplitude_spectrum(record)

    dominant_index = np.argmax(result.amplitudes)
    dominant_frequency = result.frequencies_hz[dominant_index]
    dominant_amplitude = result.amplitudes[dominant_index]

    assert np.isclose(
        result.frequency_resolution_hz,
        0.2,
    )

    assert np.isclose(
        dominant_frequency,
        1000.0,
        atol=result.frequency_resolution_hz,
    )

    assert np.isclose(
        dominant_amplitude,
        2.0,
        atol=1e-12,
    )

def test_amplitude_spectrum_for_two_tone_signal():
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

    result = calculate_amplitude_spectrum(record)

    index_1000 = np.argmin(
        np.abs(result.frequencies_hz - 1000.0)
    )

    index_1800 = np.argmin(
        np.abs(result.frequencies_hz - 1800.0)
    )

    assert np.isclose(
        result.amplitudes[index_1000],
        1.0,
        atol=1e-12,
    )

    assert np.isclose(
        result.amplitudes[index_1800],
        0.5,
        atol=1e-12,
    )

def test_amplitude_spectrum_frequency_axis():
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

    result = calculate_amplitude_spectrum(record)

    assert np.isclose(result.frequencies_hz[0], 0.0)

    assert np.isclose(
        result.frequencies_hz[-1],
        5000.0,
    )

    assert np.isclose(
        result.frequency_resolution_hz,
        1.0,
    )

def test_amplitude_spectrum_rejects_single_sample():
    record = SignalRecord(
        samples=np.array([1.0]),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="At least two samples",
    ):
        calculate_amplitude_spectrum(record)