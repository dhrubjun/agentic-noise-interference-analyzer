import numpy as np
import pytest

from noise_analyzer.dsp.psd import calculate_welch_psd
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import (
    generate_noisy_sine,
    generate_sine,
)

def test_welch_psd_detects_reference_sine():
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

    result = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    peak_index = np.argmax(result.psd)

    peak_frequency = result.frequencies_hz[peak_index]

    assert np.isclose(
        peak_frequency,
        1000.0,
        atol=result.frequency_resolution_hz,
    )

    assert np.isclose(
        result.frequency_resolution_hz,
        1.0,
    )

def test_welch_psd_integrated_power_matches_sine_power():
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

    result = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    expected_power = 2.0

    assert np.isclose(
        result.integrated_power,
        expected_power,
        rtol=0.01,
    )

def test_welch_psd_reports_configuration():
    _, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=2.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = calculate_welch_psd(
        record,
        nperseg=2000,
        noverlap=1000,
    )

    assert result.window == "hann"
    assert result.nperseg == 2000
    assert result.noverlap == 1000

    assert np.isclose(
        result.frequency_resolution_hz,
        5.0,
    )

def test_welch_psd_rejects_nperseg_larger_than_signal():
    record = SignalRecord(
        samples=np.array([1.0, 2.0, 3.0]),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="nperseg must not exceed",
    ):
        calculate_welch_psd(
            record,
            nperseg=10,
        )

def test_welch_psd_rejects_invalid_overlap():
    record = SignalRecord(
        samples=np.arange(100, dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="noverlap must be smaller",
    ):
        calculate_welch_psd(
            record,
            nperseg=50,
            noverlap=50,
        )