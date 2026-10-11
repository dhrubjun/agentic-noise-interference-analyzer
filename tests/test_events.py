import numpy as np

from noise_analyzer.dsp.events import (
    EventInterval,
    detect_events_with_hysteresis,
    postprocess_events,
)


def test_detects_single_event_with_hysteresis():
    times = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )

    scores = np.array(
        [0.5, 1.0, 6.5, 5.0, 3.5, 2.0, 0.5]
    )

    events = detect_events_with_hysteresis(
        times_seconds=times,
        scores=scores,
        start_threshold=6.0,
        end_threshold=3.0,
    )

    assert len(events) == 1

    event = events[0]

    assert event.start_index == 2
    assert event.end_index == 4

    assert np.isclose(
        event.start_time_s,
        2.0,
    )

    assert np.isclose(
        event.end_time_s,
        4.0,
    )

    assert event.peak_index == 2

    assert np.isclose(
        event.peak_score,
        6.5,
    )

import pytest


def test_detects_two_separate_events():
    times = np.arange(9, dtype=float)

    scores = np.array(
        [
            0.0,
            6.5,
            4.0,
            2.0,
            0.5,
            7.0,
            5.0,
            2.5,
            0.0,
        ]
    )

    events = detect_events_with_hysteresis(
        times_seconds=times,
        scores=scores,
        start_threshold=6.0,
        end_threshold=3.0,
    )

    assert len(events) == 2

    assert events[0].start_index == 1
    assert events[0].end_index == 2

    assert events[1].start_index == 5
    assert events[1].end_index == 6


def test_event_can_continue_to_end_of_record():
    times = np.arange(5, dtype=float)

    scores = np.array(
        [
            0.0,
            1.0,
            6.5,
            5.0,
            4.0,
        ]
    )

    events = detect_events_with_hysteresis(
        times_seconds=times,
        scores=scores,
        start_threshold=6.0,
        end_threshold=3.0,
    )

    assert len(events) == 1

    assert events[0].start_index == 2
    assert events[0].end_index == 4


def test_returns_empty_list_when_no_event_exists():
    times = np.arange(5, dtype=float)

    scores = np.array(
        [
            0.2,
            0.5,
            1.0,
            2.0,
            1.5,
        ]
    )

    events = detect_events_with_hysteresis(
        times_seconds=times,
        scores=scores,
        start_threshold=6.0,
        end_threshold=3.0,
    )

    assert events == []


def test_rejects_invalid_threshold_order():
    times = np.arange(5, dtype=float)
    scores = np.arange(5, dtype=float)

    with pytest.raises(
        ValueError,
        match="start_threshold must be greater than end_threshold",
    ):
        detect_events_with_hysteresis(
            times_seconds=times,
            scores=scores,
            start_threshold=3.0,
            end_threshold=6.0,
        )


def test_rejects_mismatched_array_lengths():
    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        detect_events_with_hysteresis(
            times_seconds=np.array([0.0, 1.0]),
            scores=np.array([1.0]),
            start_threshold=6.0,
            end_threshold=3.0,
        )

from noise_analyzer.dsp.anomaly import (
    calculate_robust_deviation_scores,
)
from noise_analyzer.dsp.baseline import (
    estimate_robust_baseline,
)
from noise_analyzer.dsp.short_time import (
    calculate_short_time_rms,
)
from noise_analyzer.models.signal import SignalRecord


def test_detects_known_amplitude_event_end_to_end():
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

    events = detect_events_with_hysteresis(
        times_seconds=rms_result.times_seconds,
        scores=score_result.scores,
        start_threshold=6.0,
        end_threshold=3.0,
    )

    assert len(events) == 1

    event = events[0]

    assert event.start_time_s <= 2.10
    assert event.end_time_s >= 2.40

    assert event.peak_score > 6.0

def test_postprocess_rejects_event_shorter_than_minimum_duration():
    events = [
        EventInterval(
            start_index=10,
            end_index=10,
            start_time_s=1.0,
            end_time_s=1.0,
            duration_s=0.0,
            peak_index=10,
            peak_time_s=1.0,
            peak_score=8.0,
        )
    ]

    result = postprocess_events(
        events,
        frame_length_seconds=0.1,
        minimum_event_duration_s=0.2,
        merge_gap_s=0.0,
    )

    assert result == []

def test_postprocess_keeps_event_long_enough():
    events = [
        EventInterval(
            start_index=10,
            end_index=12,
            start_time_s=1.0,
            end_time_s=1.2,
            duration_s=0.2,
            peak_index=11,
            peak_time_s=1.1,
            peak_score=8.0,
        )
    ]

    result = postprocess_events(
        events,
        frame_length_seconds=0.1,
        minimum_event_duration_s=0.25,
        merge_gap_s=0.0,
    )

    assert len(result) == 1

def test_postprocess_merges_nearby_events():
    events = [
        EventInterval(
            start_index=10,
            end_index=12,
            start_time_s=1.0,
            end_time_s=1.2,
            duration_s=0.2,
            peak_index=11,
            peak_time_s=1.1,
            peak_score=7.0,
        ),
        EventInterval(
            start_index=14,
            end_index=16,
            start_time_s=1.4,
            end_time_s=1.6,
            duration_s=0.2,
            peak_index=15,
            peak_time_s=1.5,
            peak_score=9.0,
        ),
    ]

    result = postprocess_events(
        events,
        frame_length_seconds=0.1,
        minimum_event_duration_s=0.0,
        merge_gap_s=0.1,
    )

    assert len(result) == 1

    event = result[0]

    assert event.start_index == 10
    assert event.end_index == 16

    assert event.peak_index == 15
    assert np.isclose(event.peak_score, 9.0)

def test_postprocess_keeps_separated_events_distinct():
    events = [
        EventInterval(
            start_index=10,
            end_index=12,
            start_time_s=1.0,
            end_time_s=1.2,
            duration_s=0.2,
            peak_index=11,
            peak_time_s=1.1,
            peak_score=7.0,
        ),
        EventInterval(
            start_index=16,
            end_index=18,
            start_time_s=1.6,
            end_time_s=1.8,
            duration_s=0.2,
            peak_index=17,
            peak_time_s=1.7,
            peak_score=9.0,
        ),
    ]

    result = postprocess_events(
        events,
        frame_length_seconds=0.1,
        minimum_event_duration_s=0.0,
        merge_gap_s=0.1,
    )

    assert len(result) == 2

def test_postprocess_rejects_nonpositive_frame_length():
    with pytest.raises(
        ValueError,
        match="frame_length_seconds must be greater than zero",
    ):
        postprocess_events(
            [],
            frame_length_seconds=0.0,
        )


def test_postprocess_rejects_negative_minimum_duration():
    with pytest.raises(
        ValueError,
        match="minimum_event_duration_s must be non-negative",
    ):
        postprocess_events(
            [],
            frame_length_seconds=0.1,
            minimum_event_duration_s=-0.1,
        )


def test_postprocess_rejects_negative_merge_gap():
    with pytest.raises(
        ValueError,
        match="merge_gap_s must be non-negative",
    ):
        postprocess_events(
            [],
            frame_length_seconds=0.1,
            merge_gap_s=-0.1,
        )

def test_end_to_end_event_postprocessing_keeps_known_event():
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

    raw_events = detect_events_with_hysteresis(
        times_seconds=rms_result.times_seconds,
        scores=score_result.scores,
        start_threshold=6.0,
        end_threshold=3.0,
    )

    final_events = postprocess_events(
        raw_events,
        frame_length_seconds=rms_result.frame_length_seconds,
        minimum_event_duration_s=0.2,
        merge_gap_s=0.1,
    )

    assert len(final_events) == 1

    event = final_events[0]

    assert event.start_time_s <= 2.10
    assert event.end_time_s >= 2.40
    assert event.peak_score > 6.0