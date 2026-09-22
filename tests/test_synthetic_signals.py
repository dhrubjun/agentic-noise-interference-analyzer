import numpy as np
import pytest

from noise_analyzer.synthetic.signals import (
    generate_constant,
    generate_dc_offset_sine,
    generate_intermittent_interference,
    generate_noisy_sine,
    generate_sine,
    generate_two_tone,
)


def test_generate_reference_sine():
    time, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=2.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    # Expected number of samples
    assert len(samples) == 50000
    assert len(time) == 50000

    # Expected sampling interval
    assert np.isclose(time[0], 0.0)
    assert np.isclose(time[1] - time[0], 1.0 / 10000.0)

    # Expected mean
    assert np.isclose(np.mean(samples), 0.0, atol=1e-12)

    # Expected RMS for a sinusoid: A / sqrt(2)
    expected_rms = 2.0 / np.sqrt(2.0)
    measured_rms = np.sqrt(np.mean(samples**2))

    assert np.isclose(
        measured_rms,
        expected_rms,
        atol=1e-12,
    )

def test_generate_constant_signal():
    time, samples = generate_constant(
        value=3.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    assert len(samples) == 50000
    assert len(time) == 50000

    assert np.isclose(time[0], 0.0)
    assert np.isclose(time[1] - time[0], 1.0 / 10000.0)

    assert np.all(samples == 3.0)

def test_generate_two_tone_signal():
    time, samples = generate_two_tone(
        frequency_1_hz=1000.0,
        amplitude_1=1.0,
        frequency_2_hz=1800.0,
        amplitude_2=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    assert len(samples) == 50000
    assert len(time) == 50000

    expected_samples = (
        1.0 * np.sin(2.0 * np.pi * 1000.0 * time)
        + 0.5 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    assert np.allclose(samples, expected_samples)

def test_generate_noisy_sine_is_reproducible_with_same_seed():
    time_1, samples_1 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
        seed=42,
    )

    time_2, samples_2 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
        seed=42,
    )

    assert len(samples_1) == 100000
    assert len(time_1) == 100000

    assert np.allclose(time_1, time_2)
    assert np.allclose(samples_1, samples_2)

def test_generate_noisy_sine_changes_with_different_seed():
    _, samples_1 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
        seed=42,
    )

    _, samples_2 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
        seed=43,
    )

    assert not np.allclose(samples_1, samples_2)

def test_generate_noisy_sine_noise_level_is_reasonable():
    time, noisy_samples = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
        seed=42,
    )

    clean_samples = 1.0 * np.sin(2.0 * np.pi * 1000.0 * time)

    noise = noisy_samples - clean_samples

    measured_noise_std = np.std(noise)

    assert np.isclose(measured_noise_std, 0.1, atol=0.01)

def test_increased_noise_has_larger_variation():
    time_low, samples_low = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
        seed=42,
    )

    time_high, samples_high = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
        seed=42,
    )

    clean_signal = np.sin(2.0 * np.pi * 1000.0 * time_low)

    noise_low = samples_low - clean_signal
    noise_high = samples_high - clean_signal

    assert np.allclose(time_low, time_high)

    assert np.std(noise_high) > np.std(noise_low)

    assert np.isclose(
        np.std(noise_low),
        0.1,
        atol=0.01,
    )

    assert np.isclose(
        np.std(noise_high),
        0.5,
        atol=0.02,
    )

def test_generate_dc_offset_sine():
    time, samples = generate_dc_offset_sine(
        dc_offset=2.0,
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    assert len(samples) == 50000
    assert len(time) == 50000

    expected_samples = (
        2.0
        + np.sin(2.0 * np.pi * 1000.0 * time)
    )

    assert np.allclose(samples, expected_samples)

    # For this coherent reference case, the sine averages to zero.
    assert np.isclose(np.mean(samples), 2.0, atol=1e-12)

def test_generate_intermittent_interference():
    time, samples = generate_intermittent_interference(
        base_frequency_hz=1000.0,
        base_amplitude=1.0,
        interference_frequency_hz=1800.0,
        interference_amplitude=0.5,
        interference_start_seconds=4.0,
        interference_end_seconds=6.0,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
    )

    assert len(samples) == 100000
    assert len(time) == 100000

    base_signal = np.sin(
        2.0 * np.pi * 1000.0 * time
    )

    interference = 0.5 * np.sin(
        2.0 * np.pi * 1800.0 * time
    )

    before_mask = time < 4.0
    during_mask = (time >= 4.0) & (time < 6.0)
    after_mask = time >= 6.0

    assert np.allclose(
        samples[before_mask],
        base_signal[before_mask],
    )

    assert np.allclose(
        samples[during_mask],
        base_signal[during_mask] + interference[during_mask],
    )

    assert np.allclose(
        samples[after_mask],
        base_signal[after_mask],
    )

def test_generate_non_bin_centered_sine():
    frequency_hz = 1000.37
    amplitude = 1.0
    sample_rate_hz = 10000.0
    duration_seconds = 5.0

    time, samples = generate_sine(
        frequency_hz=frequency_hz,
        amplitude=amplitude,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )

    assert len(samples) == 50000
    assert len(time) == 50000

    expected_samples = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time
    )

    assert np.allclose(samples, expected_samples)

def test_create_signal_with_nan():
    _, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=1.0,
    )

    corrupted_samples = samples.copy()
    corrupted_samples[100] = np.nan

    assert len(corrupted_samples) == 10000
    assert np.isnan(corrupted_samples[100])

    assert np.sum(np.isnan(corrupted_samples)) == 1

def test_create_signal_with_inf():
    _, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=1.0,
    )

    corrupted_samples = samples.copy()
    corrupted_samples[100] = np.inf

    assert len(corrupted_samples) == 10000
    assert np.isinf(corrupted_samples[100])

    assert np.sum(np.isinf(corrupted_samples)) == 1

def test_create_empty_signal_reference_case():
    samples = np.array([], dtype=float)

    assert samples.size == 0
    assert samples.dtype == float

def test_generate_sine_rejects_zero_sample_rate():
    with pytest.raises(
        ValueError,
        match="sample_rate_hz must be greater than zero",
    ):
        generate_sine(
            frequency_hz=1000.0,
            amplitude=1.0,
            sample_rate_hz=0.0,
            duration_seconds=1.0,
        )

def test_generate_sine_rejects_negative_sample_rate():
    with pytest.raises(
        ValueError,
        match="sample_rate_hz must be greater than zero",
    ):
        generate_sine(
            frequency_hz=1000.0,
            amplitude=1.0,
            sample_rate_hz=-10000.0,
            duration_seconds=1.0,
        )

def test_create_too_short_signal_reference_case():
    time, samples = generate_sine(
        frequency_hz=100.0,
        amplitude=1.0,
        sample_rate_hz=1000.0,
        duration_seconds=0.002,
    )

    assert len(samples) == 2
    assert len(time) == 2

    assert np.isclose(time[0], 0.0)
    assert np.isclose(time[1], 0.001)

def test_create_nonuniform_timestamp_reference_case():
    sample_rate_hz = 1000.0
    duration_seconds = 1.0

    time, samples = generate_sine(
        frequency_hz=100.0,
        amplitude=1.0,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )

    nonuniform_time = time.copy()

    # Introduce one deliberate timing irregularity.
    nonuniform_time[500] += 0.0002

    time_steps = np.diff(nonuniform_time)

    assert len(samples) == 1000
    assert len(nonuniform_time) == 1000

    assert not np.allclose(
        time_steps,
        1.0 / sample_rate_hz,
        rtol=0.0,
        atol=1e-12,
    )

def test_fixed_seed_produces_identical_noisy_signal():
    time_1, samples_1 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=2.0,
        seed=42,
    )

    time_2, samples_2 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=2.0,
        seed=42,
    )

    assert np.array_equal(time_1, time_2)
    assert np.array_equal(samples_1, samples_2)