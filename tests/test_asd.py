import numpy as np

from noise_analyzer.dsp.asd import calculate_asd
from noise_analyzer.dsp.psd import calculate_welch_psd
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import generate_sine

def test_asd_squared_matches_psd():
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

    psd_result = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    asd_result = calculate_asd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    assert np.allclose(
        asd_result.asd**2,
        psd_result.psd,
    )

def test_asd_frequency_axis_matches_psd():
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

    psd_result = calculate_welch_psd(
        record,
        nperseg=2000,
        noverlap=1000,
    )

    asd_result = calculate_asd(
        record,
        nperseg=2000,
        noverlap=1000,
    )

    assert np.array_equal(
        asd_result.frequencies_hz,
        psd_result.frequencies_hz,
    )

    assert np.isclose(
        asd_result.frequency_resolution_hz,
        psd_result.frequency_resolution_hz,
    )

def test_asd_detects_reference_sine_frequency():
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

    result = calculate_asd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    peak_index = np.argmax(result.asd)

    peak_frequency = result.frequencies_hz[peak_index]

    assert np.isclose(
        peak_frequency,
        1000.0,
        atol=result.frequency_resolution_hz,
    )

