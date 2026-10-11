import numpy as np
import pytest

from noise_analyzer.dsp.anomaly import (
    calculate_robust_deviation_scores,
)
from noise_analyzer.dsp.baseline import BaselineResult


def test_robust_deviation_scores_known_values():
    baseline = BaselineResult(
        location=10.0,
        mad=2.0,
        robust_scale=4.0,
        method="median_mad",
        status="valid",
        number_of_values=5,
    )

    values = np.array(
        [
            10.0,
            14.0,
            18.0,
            6.0,
        ]
    )

    result = calculate_robust_deviation_scores(
        values,
        baseline,
    )

    expected_scores = np.array(
        [
            0.0,
            1.0,
            2.0,
            -1.0,
        ]
    )

    assert np.allclose(
        result.scores,
        expected_scores,
    )

    assert result.baseline_location == 10.0
    assert result.baseline_scale == 4.0

def test_deviation_scores_reject_degenerate_baseline():
    baseline = BaselineResult(
        location=2.0,
        mad=0.0,
        robust_scale=0.0,
        method="median_mad",
        status="degenerate",
        number_of_values=4,
    )

    values = np.array(
        [2.0, 2.0, 3.0]
    )

    with pytest.raises(
        ValueError,
        match="baseline must have valid non-zero scale",
    ):
        calculate_robust_deviation_scores(
            values,
            baseline,
        )


def test_deviation_scores_reject_empty_input():
    baseline = BaselineResult(
        location=0.0,
        mad=1.0,
        robust_scale=1.4826,
        method="median_mad",
        status="valid",
        number_of_values=5,
    )

    with pytest.raises(
        ValueError,
        match="feature_values must not be empty",
    ):
        calculate_robust_deviation_scores(
            np.array([], dtype=float),
            baseline,
        )


def test_deviation_scores_reject_nonfinite_input():
    baseline = BaselineResult(
        location=0.0,
        mad=1.0,
        robust_scale=1.4826,
        method="median_mad",
        status="valid",
        number_of_values=5,
    )

    with pytest.raises(
        ValueError,
        match="only finite values",
    ):
        calculate_robust_deviation_scores(
            np.array(
                [1.0, np.nan, 2.0]
            ),
            baseline,
        )


def test_deviation_scores_reject_multidimensional_input():
    baseline = BaselineResult(
        location=0.0,
        mad=1.0,
        robust_scale=1.4826,
        method="median_mad",
        status="valid",
        number_of_values=5,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        calculate_robust_deviation_scores(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            baseline,
        )

from noise_analyzer.dsp.baseline import estimate_robust_baseline
from noise_analyzer.dsp.short_time import calculate_short_time_rms
from noise_analyzer.models.signal import SignalRecord


def test_deviation_scores_increase_during_amplitude_event():
    sample_rate_hz = 1000.0
    frequency_hz = 100.0
    duration_seconds = 4.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    samples = (
        np.sin(
            2.0 * np.pi * frequency_hz * time
        )
        + 0.02 * np.sin(
            2.0 * np.pi * 17.0 * time
        )
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

    score_result = calculate_robust_deviation_scores(
        rms_result.rms,
        baseline,
    )

    before_event = rms_result.times_seconds < 1.8

    during_event = (
        (rms_result.times_seconds >= 2.1)
        & (rms_result.times_seconds <= 2.4)
    )

    assert np.max(
        np.abs(score_result.scores[before_event])
    ) < 1.5
    
    assert np.min(
        score_result.scores[during_event]
    ) > 5.0