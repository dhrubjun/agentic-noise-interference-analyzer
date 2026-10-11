import numpy as np

from noise_analyzer.dsp.event_comparison import (
    compare_event_to_baseline_spectrum,
)
from noise_analyzer.models.signal import SignalRecord


def test_event_spectral_comparison_detects_four_times_power():
    sample_rate_hz = 1000.0
    duration_seconds = 4.0
    frequency_hz = 100.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    baseline_samples = np.sin(
        2.0 * np.pi * frequency_hz * time
    )

    event_samples = 2.0 * np.sin(
        2.0 * np.pi * frequency_hz * time
    )

    baseline_record = SignalRecord(
        samples=baseline_samples,
        sample_rate_hz=sample_rate_hz,
    )

    event_record = SignalRecord(
        samples=event_samples,
        sample_rate_hz=sample_rate_hz,
    )

    result = compare_event_to_baseline_spectrum(
        baseline_record=baseline_record,
        event_record=event_record,
        lower_frequency_hz=80.0,
        upper_frequency_hz=120.0,
        nperseg=500,
    )

    assert np.isclose(
        result.total_power_ratio,
        4.0,
        rtol=1e-6,
    )

    assert np.isclose(
        result.band_power_ratio,
        4.0,
        rtol=1e-6,
    )

def test_event_spectral_comparison_detects_centroid_shift():
    sample_rate_hz = 1000.0
    duration_seconds = 4.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    baseline_samples = np.sin(
        2.0 * np.pi * 100.0 * time
    )

    event_samples = np.sin(
        2.0 * np.pi * 300.0 * time
    )

    baseline_record = SignalRecord(
        samples=baseline_samples,
        sample_rate_hz=sample_rate_hz,
    )

    event_record = SignalRecord(
        samples=event_samples,
        sample_rate_hz=sample_rate_hz,
    )

    result = compare_event_to_baseline_spectrum(
        baseline_record=baseline_record,
        event_record=event_record,
        lower_frequency_hz=50.0,
        upper_frequency_hz=350.0,
        nperseg=500,
    )

    assert np.isclose(
        result.baseline_centroid_hz,
        100.0,
        atol=2.0,
    )

    assert np.isclose(
        result.event_centroid_hz,
        300.0,
        atol=2.0,
    )

    assert result.centroid_shift_hz > 190.0

def test_event_spectral_comparison_detects_flatness_drop():
    sample_rate_hz = 1000.0
    duration_seconds = 4.0

    rng = np.random.default_rng(12345)

    baseline_samples = rng.normal(
        loc=0.0,
        scale=1.0,
        size=int(sample_rate_hz * duration_seconds),
    )

    time = np.arange(
        baseline_samples.size
    ) / sample_rate_hz

    event_samples = (
        0.2 * rng.normal(
            loc=0.0,
            scale=1.0,
            size=baseline_samples.size,
        )
        + 3.0 * np.sin(
            2.0 * np.pi * 100.0 * time
        )
    )

    baseline_record = SignalRecord(
        samples=baseline_samples,
        sample_rate_hz=sample_rate_hz,
    )

    event_record = SignalRecord(
        samples=event_samples,
        sample_rate_hz=sample_rate_hz,
    )

    result = compare_event_to_baseline_spectrum(
        baseline_record=baseline_record,
        event_record=event_record,
        lower_frequency_hz=50.0,
        upper_frequency_hz=150.0,
        nperseg=500,
    )

    assert (
        result.event_flatness
        < result.baseline_flatness
    )

    assert result.flatness_change < 0.0

import pytest


def test_event_spectral_comparison_rejects_sample_rate_mismatch():
    baseline_record = SignalRecord(
        samples=np.ones(1000),
        sample_rate_hz=1000.0,
    )

    event_record = SignalRecord(
        samples=np.ones(1000),
        sample_rate_hz=2000.0,
    )

    with pytest.raises(
        ValueError,
        match="must have the same sample rate",
    ):
        compare_event_to_baseline_spectrum(
            baseline_record=baseline_record,
            event_record=event_record,
            lower_frequency_hz=80.0,
            upper_frequency_hz=120.0,
            nperseg=500,
        )

from noise_analyzer.dsp.anomaly import (
    calculate_robust_deviation_scores,
)
from noise_analyzer.dsp.baseline import (
    estimate_robust_baseline,
)
from noise_analyzer.dsp.events import (
    detect_events_with_hysteresis,
    postprocess_events,
)
from noise_analyzer.dsp.short_time import (
    calculate_short_time_band_power,
)


def test_detected_event_has_higher_in_band_power_than_baseline():
    sample_rate_hz = 1000.0
    duration_seconds = 6.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    samples = 0.2 * np.sin(
        2.0 * np.pi * 300.0 * time
    )

    event_mask = (
        (time >= 3.0)
        & (time < 4.0)
    )

    samples[event_mask] += np.sin(
        2.0 * np.pi * 100.0 * time[event_mask]
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    band_track = calculate_short_time_band_power(
        record,
        lower_frequency_hz=80.0,
        upper_frequency_hz=120.0,
        frame_length_samples=200,
        hop_length_samples=100,
    )

    baseline = estimate_robust_baseline(
        band_track.band_power
    )

    scores = calculate_robust_deviation_scores(
        band_track.band_power,
        baseline,
    )

    raw_events = detect_events_with_hysteresis(
        times_seconds=band_track.times_seconds,
        scores=scores.scores,
        start_threshold=6.0,
        end_threshold=3.0,
    )

    events = postprocess_events(
        raw_events,
        frame_length_seconds=band_track.frame_length_seconds,
        minimum_event_duration_s=0.4,
        merge_gap_s=0.1,
    )

    assert len(events) == 1

    event = events[0]

    event_start_sample = int(
        max(
            0.0,
            event.start_time_s
            - band_track.frame_length_seconds / 2.0,
        )
        * sample_rate_hz
    )

    event_end_sample = int(
        min(
            duration_seconds,
            event.end_time_s
            + band_track.frame_length_seconds / 2.0,
        )
        * sample_rate_hz
    )

    baseline_record = SignalRecord(
        samples=samples[:2000],
        sample_rate_hz=sample_rate_hz,
    )

    event_record = SignalRecord(
        samples=samples[
            event_start_sample:event_end_sample
        ],
        sample_rate_hz=sample_rate_hz,
    )

    result = compare_event_to_baseline_spectrum(
        baseline_record=baseline_record,
        event_record=event_record,
        lower_frequency_hz=80.0,
        upper_frequency_hz=120.0,
        nperseg=200,
    )

    assert result.event_band_power > result.baseline_band_power
    assert result.band_power_ratio > 10.0