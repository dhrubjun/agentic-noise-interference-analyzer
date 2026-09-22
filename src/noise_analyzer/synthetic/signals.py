import numpy as np


def generate_sine(
    frequency_hz: float,
    amplitude: float,
    sample_rate_hz: float,
    duration_seconds: float,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a uniformly sampled real-valued sinusoid."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than zero.")

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")

    number_of_samples = int(round(sample_rate_hz * duration_seconds))

    time = np.arange(number_of_samples, dtype=float) / sample_rate_hz

    samples = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time + phase_rad
    )

    return time, samples