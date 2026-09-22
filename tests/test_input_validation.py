import numpy as np
import pytest

from noise_analyzer.io.validation import (
    is_constant_signal,
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord


def test_validate_finite_samples_accepts_valid_signal():
    record = SignalRecord(
        samples=np.array([1.0, 2.0, 3.0]),
        sample_rate_hz=1000.0,
    )

    validate_finite_samples(record)

def test_validate_finite_samples_rejects_nan():
    record = SignalRecord(
        samples=np.array([1.0, np.nan, 3.0]),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="non-finite sample values",
    ):
        validate_finite_samples(record)

def test_validate_finite_samples_rejects_inf():
    record = SignalRecord(
        samples=np.array([1.0, np.inf, 3.0]),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="non-finite sample values",
    ):
        validate_finite_samples(record)

def test_validate_nonempty_signal_rejects_empty_signal():
    record = SignalRecord(
        samples=np.array([], dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="Signal contains no samples",
    ):
        validate_nonempty_signal(record)

def test_validate_nonempty_signal_accepts_signal_with_samples():
    record = SignalRecord(
        samples=np.array([1.0]),
        sample_rate_hz=1000.0,
    )

    validate_nonempty_signal(record)

def test_is_constant_signal_detects_constant_signal():
    record = SignalRecord(
        samples=np.array([3.0, 3.0, 3.0, 3.0]),
        sample_rate_hz=1000.0,
    )

    assert is_constant_signal(record) is True

def test_is_constant_signal_rejects_nonconstant_signal():
    record = SignalRecord(
        samples=np.array([1.0, 2.0, 1.0, 2.0]),
        sample_rate_hz=1000.0,
    )

    assert is_constant_signal(record) is False

def test_is_constant_signal_returns_false_for_empty_signal():
    record = SignalRecord(
        samples=np.array([], dtype=float),
        sample_rate_hz=1000.0,
    )

    assert is_constant_signal(record) is False