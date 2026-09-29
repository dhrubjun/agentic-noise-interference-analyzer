import numpy as np
import pytest

from noise_analyzer.dsp.spectrogram import calculate_spectrogram
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import (
    generate_intermittent_interference,
)

def test_spectrogram_reports_expected_resolution():
    _, samples = generate_intermittent_interference(
        base_frequency_hz=1000.0,
        base_amplitude=1.0,
        interference_frequency_hz=1800.0,
        interference_amplitude=0.5,
        interference_start_seconds=4.0,
        interference_end_seconds=6.0,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = calculate_spectrogram(
        record,
        nperseg=1000,
        noverlap=500,
    )

    assert np.isclose(
        result.frequency_resolution_hz,
        10.0,
    )

    assert np.isclose(
        result.time_step_seconds,
        0.05,
    )

    assert result.window == "hann"
    assert result.nperseg == 1000
    assert result.noverlap == 500

def test_spectrogram_detects_base_tone_throughout():
    _, samples = generate_intermittent_interference(
        base_frequency_hz=1000.0,
        base_amplitude=1.0,
        interference_frequency_hz=1800.0,
        interference_amplitude=0.5,
        interference_start_seconds=4.0,
        interference_end_seconds=6.0,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = calculate_spectrogram(
        record,
        nperseg=1000,
        noverlap=500,
    )

    index_1000 = np.argmin(
        np.abs(result.frequencies_hz - 1000.0)
    )

    power_1000 = result.power_spectral_density[index_1000, :]

    assert np.all(power_1000 > 0)

def test_spectrogram_localizes_intermittent_interference():
    _, samples = generate_intermittent_interference(
        base_frequency_hz=1000.0,
        base_amplitude=1.0,
        interference_frequency_hz=1800.0,
        interference_amplitude=0.5,
        interference_start_seconds=4.0,
        interference_end_seconds=6.0,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = calculate_spectrogram(
        record,
        nperseg=1000,
        noverlap=500,
    )

    index_1800 = np.argmin(
        np.abs(result.frequencies_hz - 1800.0)
    )

    power_1800 = result.power_spectral_density[index_1800, :]

    before_mask = result.times_seconds < 3.5
    during_mask = (
        (result.times_seconds >= 4.2)
        & (result.times_seconds <= 5.8)
    )
    after_mask = result.times_seconds > 6.5

    before_power = np.mean(power_1800[before_mask])
    during_power = np.mean(power_1800[during_mask])
    after_power = np.mean(power_1800[after_mask])

    assert during_power > before_power
    assert during_power > after_power

def test_spectrogram_rejects_invalid_overlap():
    record = SignalRecord(
        samples=np.arange(1000, dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="noverlap must be smaller",
    ):
        calculate_spectrogram(
            record,
            nperseg=100,
            noverlap=100,
        )