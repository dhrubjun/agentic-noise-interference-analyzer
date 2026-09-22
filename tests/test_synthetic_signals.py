import numpy as np

from noise_analyzer.synthetic.signals import generate_sine


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