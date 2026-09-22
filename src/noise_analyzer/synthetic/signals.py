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

def generate_constant(
    value: float,
    sample_rate_hz: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a uniformly sampled constant-valued signal."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than zero.")

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    number_of_samples = int(round(sample_rate_hz * duration_seconds))

    time = np.arange(number_of_samples, dtype=float) / sample_rate_hz

    samples = np.full(
        number_of_samples,
        fill_value=value,
        dtype=float,
    )

    return time, samples

def generate_two_tone(
    frequency_1_hz: float,
    amplitude_1: float,
    frequency_2_hz: float,
    amplitude_2: float,
    sample_rate_hz: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the sum of two uniformly sampled real-valued sinusoids."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than zero.")

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if frequency_1_hz < 0 or frequency_2_hz < 0:
        raise ValueError("Frequencies must be non-negative.")

    number_of_samples = int(round(sample_rate_hz * duration_seconds))

    time = np.arange(number_of_samples, dtype=float) / sample_rate_hz

    tone_1 = amplitude_1 * np.sin(
        2.0 * np.pi * frequency_1_hz * time
    )

    tone_2 = amplitude_2 * np.sin(
        2.0 * np.pi * frequency_2_hz * time
    )

    samples = tone_1 + tone_2

    return time, samples

def generate_noisy_sine(
    frequency_hz: float,
    amplitude: float,
    noise_std: float,
    sample_rate_hz: float,
    duration_seconds: float,
    seed: int | None = None,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a uniformly sampled sinusoid corrupted by additive Gaussian noise."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than zero.")

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    number_of_samples = int(round(sample_rate_hz * duration_seconds))

    time = np.arange(number_of_samples, dtype=float) / sample_rate_hz

    clean_signal = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time + phase_rad
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=number_of_samples,
    )

    samples = clean_signal + noise

    return time, samples

def generate_dc_offset_sine(
    dc_offset: float,
    frequency_hz: float,
    amplitude: float,
    sample_rate_hz: float,
    duration_seconds: float,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sinusoid with a constant DC offset."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than zero.")

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")

    number_of_samples = int(round(sample_rate_hz * duration_seconds))

    time = np.arange(number_of_samples, dtype=float) / sample_rate_hz

    samples = dc_offset + amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time + phase_rad
    )

    return time, samples

def generate_intermittent_interference(
    base_frequency_hz: float,
    base_amplitude: float,
    interference_frequency_hz: float,
    interference_amplitude: float,
    interference_start_seconds: float,
    interference_end_seconds: float,
    sample_rate_hz: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a base sinusoid with an intermittent interfering sinusoid."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than zero.")

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if base_frequency_hz < 0 or interference_frequency_hz < 0:
        raise ValueError("Frequencies must be non-negative.")

    if interference_start_seconds < 0:
        raise ValueError("interference_start_seconds must be non-negative.")

    if interference_end_seconds <= interference_start_seconds:
        raise ValueError(
            "interference_end_seconds must be greater than interference_start_seconds."
        )

    if interference_end_seconds > duration_seconds:
        raise ValueError(
            "interference_end_seconds must not exceed duration_seconds."
        )

    number_of_samples = int(round(sample_rate_hz * duration_seconds))

    time = np.arange(number_of_samples, dtype=float) / sample_rate_hz

    base_signal = base_amplitude * np.sin(
        2.0 * np.pi * base_frequency_hz * time
    )

    interference = interference_amplitude * np.sin(
        2.0 * np.pi * interference_frequency_hz * time
    )

    active_mask = (
        (time >= interference_start_seconds)
        & (time < interference_end_seconds)
    )

    samples = base_signal.copy()
    samples[active_mask] += interference[active_mask]

    return time, samples