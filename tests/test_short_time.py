import numpy as np

from noise_analyzer.dsp.short_time import (
    calculate_short_time_rms,
    frame_signal,
)
from noise_analyzer.models.signal import SignalRecord


def test_frame_signal_extracts_expected_frames():
    samples = np.arange(8, dtype=float)

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=4.0,
    )

    result = frame_signal(
        record,
        frame_length_samples=4,
        hop_length_samples=2,
    )

    expected_frames = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
        ]
    )

    expected_times = np.array(
        [0.5, 1.0, 1.5]
    )

    assert np.array_equal(
        result.frames,
        expected_frames,
    )

    assert np.allclose(
        result.times_seconds,
        expected_times,
    )

    assert result.frame_length_samples == 4
    assert result.hop_length_samples == 2

    assert np.isclose(
        result.frame_length_seconds,
        1.0,
    )

    assert np.isclose(
        result.hop_length_seconds,
        0.5,
    )

import pytest


def test_frame_signal_rejects_frame_longer_than_signal():
    record = SignalRecord(
        samples=np.arange(5, dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="frame_length_samples must not exceed",
    ):
        frame_signal(
            record,
            frame_length_samples=10,
            hop_length_samples=2,
        )


def test_frame_signal_rejects_zero_frame_length():
    record = SignalRecord(
        samples=np.arange(10, dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="frame_length_samples must be at least 1",
    ):
        frame_signal(
            record,
            frame_length_samples=0,
            hop_length_samples=2,
        )


def test_frame_signal_rejects_zero_hop_length():
    record = SignalRecord(
        samples=np.arange(10, dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="hop_length_samples must be at least 1",
    ):
        frame_signal(
            record,
            frame_length_samples=4,
            hop_length_samples=0,
        )


def test_frame_signal_uses_only_complete_frames():
    record = SignalRecord(
        samples=np.arange(9, dtype=float),
        sample_rate_hz=1.0,
    )

    result = frame_signal(
        record,
        frame_length_samples=4,
        hop_length_samples=3,
    )

    expected_frames = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )

    assert np.array_equal(
        result.frames,
        expected_frames,
    )

def test_short_time_rms_for_constant_signal():
    samples = np.full(
        1000,
        3.0,
        dtype=float,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=1000.0,
    )

    result = calculate_short_time_rms(
        record,
        frame_length_samples=100,
        hop_length_samples=50,
    )

    assert np.allclose(
        result.rms,
        3.0,
    )

def test_short_time_rms_for_sine_wave():
    sample_rate_hz = 1000.0
    frequency_hz = 100.0
    amplitude = 2.0
    duration_seconds = 2.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    samples = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    result = calculate_short_time_rms(
        record,
        frame_length_samples=100,
        hop_length_samples=50,
    )

    expected_rms = amplitude / np.sqrt(2.0)

    assert np.allclose(
        result.rms,
        expected_rms,
        atol=1e-12,
    )

def test_short_time_rms_tracks_amplitude_change():
    sample_rate_hz = 1000.0
    frequency_hz = 100.0
    duration_seconds = 2.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    samples = np.empty_like(time)

    first_half = time < 1.0
    second_half = time >= 1.0

    samples[first_half] = 1.0 * np.sin(
        2.0 * np.pi * frequency_hz * time[first_half]
    )

    samples[second_half] = 2.0 * np.sin(
        2.0 * np.pi * frequency_hz * time[second_half]
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    result = calculate_short_time_rms(
        record,
        frame_length_samples=100,
        hop_length_samples=50,
    )

    before_change = result.times_seconds <= 0.95
    after_change = result.times_seconds >= 1.05

    expected_before = 1.0 / np.sqrt(2.0)
    expected_after = 2.0 / np.sqrt(2.0)

    assert np.allclose(
        result.rms[before_change],
        expected_before,
        atol=1e-12,
    )

    assert np.allclose(
        result.rms[after_change],
        expected_after,
        atol=1e-12,
    )