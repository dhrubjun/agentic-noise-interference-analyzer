import numpy as np
import pytest

from noise_analyzer.dsp.time_domain import calculate_time_domain_statistics
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import (
    generate_constant,
    generate_dc_offset_sine,
    generate_sine,
)


def test_time_domain_statistics_for_reference_sine():
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

    result = calculate_time_domain_statistics(record)

    expected_rms = 2.0 / np.sqrt(2.0)

    assert np.isclose(result.mean, 0.0, atol=1e-12)
    assert np.isclose(result.median, 0.0, atol=1e-12)
    assert np.isclose(result.rms, expected_rms, atol=1e-12)

    assert np.isclose(
        result.standard_deviation,
        expected_rms,
        atol=1e-12,
    )

    assert np.isclose(
        result.variance,
        2.0,
        atol=1e-12,
    )

def test_time_domain_statistics_for_constant_signal():
    _, samples = generate_constant(
        value=3.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = calculate_time_domain_statistics(record)

    assert np.isclose(result.mean, 3.0)
    assert np.isclose(result.median, 3.0)
    assert np.isclose(result.rms, 3.0)

    assert np.isclose(result.standard_deviation, 0.0)
    assert np.isclose(result.variance, 0.0)

    assert np.isclose(result.minimum, 3.0)
    assert np.isclose(result.maximum, 3.0)
    assert np.isclose(result.peak_to_peak, 0.0)

def test_time_domain_statistics_for_dc_offset_sine():
    _, samples = generate_dc_offset_sine(
        dc_offset=2.0,
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = calculate_time_domain_statistics(record)

    assert np.isclose(result.mean, 2.0, atol=1e-12)

def test_time_domain_statistics_rejects_empty_signal():
    record = SignalRecord(
        samples=np.array([], dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="Signal contains no samples",
    ):
        calculate_time_domain_statistics(record)

def test_time_domain_statistics_rejects_nan():
    record = SignalRecord(
        samples=np.array([1.0, np.nan, 3.0]),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="non-finite sample values",
    ):
        calculate_time_domain_statistics(record)