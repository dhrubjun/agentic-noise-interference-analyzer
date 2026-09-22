import numpy as np

from noise_analyzer.synthetic.signals import (
    generate_constant,
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