import numpy as np
import pytest

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import generate_sine


def test_signal_record_with_reference_sine():
    time, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=2.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
        channel_name="reference_sine",
        units="V",
    )

    assert record.number_of_samples == 50000
    assert np.isclose(record.sample_rate_hz, 10000.0)
    assert np.isclose(record.duration_seconds, 5.0)
    assert np.isclose(record.nyquist_frequency_hz, 5000.0)
    assert np.isclose(record.sampling_interval_seconds, 0.0001)

    assert record.channel_name == "reference_sine"
    assert record.units == "V"

    assert np.array_equal(record.samples, samples)

def test_signal_record_rejects_zero_sample_rate():
    samples = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="sample_rate_hz must be greater than zero",
    ):
        SignalRecord(
            samples=samples,
            sample_rate_hz=0.0,
        )

def test_signal_record_rejects_nonfinite_sample_rate():
    samples = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="sample_rate_hz must be finite",
    ):
        SignalRecord(
            samples=samples,
            sample_rate_hz=np.inf,
        )