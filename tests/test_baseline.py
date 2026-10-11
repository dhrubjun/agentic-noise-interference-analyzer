import numpy as np
import pytest

from noise_analyzer.dsp.baseline import (
    estimate_robust_baseline,
)


def test_robust_baseline_known_values():
    values = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            100.0,
        ]
    )

    result = estimate_robust_baseline(values)

    assert np.isclose(
        result.location,
        3.0,
    )

    assert np.isclose(
        result.mad,
        1.0,
    )

    assert np.isclose(
        result.robust_scale,
        1.4826,
    )

    assert result.method == "median_mad"
    assert result.status == "valid"
    assert result.number_of_values == 5

def test_robust_baseline_marks_zero_mad_as_degenerate():
    values = np.array(
        [
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )

    result = estimate_robust_baseline(values)

    assert np.isclose(result.location, 2.0)
    assert np.isclose(result.mad, 0.0)
    assert np.isclose(result.robust_scale, 0.0)
    assert result.status == "degenerate"

def test_robust_baseline_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="feature_values must not be empty",
    ):
        estimate_robust_baseline(
            np.array([], dtype=float)
        )


def test_robust_baseline_rejects_nonfinite_input():
    with pytest.raises(
        ValueError,
        match="only finite values",
    ):
        estimate_robust_baseline(
            np.array([1.0, np.nan, 2.0])
        )


def test_robust_baseline_rejects_multidimensional_input():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        estimate_robust_baseline(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )

def test_robust_baseline_resists_large_event_values():
    values = np.array(
        [
            0.70,
            0.71,
            0.69,
            0.72,
            0.70,
            0.71,
            0.70,
            3.00,
            4.00,
        ]
    )

    result = estimate_robust_baseline(values)

    assert np.isclose(
        result.location,
        0.71,
    )

    assert result.location < np.mean(values)

    assert result.status == "valid"

from noise_analyzer.dsp.short_time import calculate_short_time_rms
from noise_analyzer.models.signal import SignalRecord


def test_robust_baseline_from_short_time_rms():
    sample_rate_hz = 1000.0
    frequency_hz = 100.0
    duration_seconds = 4.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    samples = np.sin(
        2.0 * np.pi * frequency_hz * time
    )

    event_mask = (
        (time >= 2.0)
        & (time < 2.5)
    )

    samples[event_mask] *= 4.0

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    rms_result = calculate_short_time_rms(
        record,
        frame_length_samples=100,
        hop_length_samples=50,
    )

    baseline = estimate_robust_baseline(
        rms_result.rms
    )

    expected_normal_rms = 1.0 / np.sqrt(2.0)

    assert np.isclose(
        baseline.location,
        expected_normal_rms,
        atol=1e-12,
    )

    assert baseline.status == "valid"