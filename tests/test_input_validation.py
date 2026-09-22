import numpy as np
import pytest

from noise_analyzer.io.validation import validate_finite_samples
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