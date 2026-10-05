import numpy as np
import pytest

from noise_analyzer.dsp.noise_floor import (
    estimate_noise_floor,
)
from noise_analyzer.dsp.psd import (
    calculate_welch_psd,
)
from noise_analyzer.models.signal import SignalRecord


def calculate_test_psd(
    samples: np.ndarray,
    sample_rate_hz: float = 10000.0,
):
    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    return calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
        window="hann",
    )


def test_noise_floor_is_stable_with_strong_tone():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_samples,
    )

    strong_tone = 10.0 * np.sin(
        2.0 * np.pi * 1000.0 * time
    )

    noise_psd = calculate_test_psd(
        noise,
        sample_rate_hz,
    )

    tone_psd = calculate_test_psd(
        noise + strong_tone,
        sample_rate_hz,
    )

    noise_floor = estimate_noise_floor(
        noise_psd
    )

    tone_floor = estimate_noise_floor(
        tone_psd
    )

    difference_db = abs(
        tone_floor.noise_floor_db
        - noise_floor.noise_floor_db
    )

    assert difference_db < 0.1


def test_noise_floor_is_stable_with_many_tones():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_samples,
    )

    frequencies_hz = [
        137.3,
        421.7,
        803.2,
        1199.4,
        1677.8,
        2143.6,
        2789.1,
        3321.5,
        3897.2,
        4471.3,
    ]

    many_tones = np.zeros_like(time)

    for frequency_hz in frequencies_hz:
        many_tones += 8.0 * np.sin(
            2.0 * np.pi * frequency_hz * time
        )

    noise_psd = calculate_test_psd(
        noise,
        sample_rate_hz,
    )

    mixed_psd = calculate_test_psd(
        noise + many_tones,
        sample_rate_hz,
    )

    noise_floor = estimate_noise_floor(
        noise_psd
    )

    mixed_floor = estimate_noise_floor(
        mixed_psd
    )

    difference_db = abs(
        mixed_floor.noise_floor_db
        - noise_floor.noise_floor_db
    )

    assert difference_db < 0.1


def test_noise_floor_increases_with_noise_level():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    rng = np.random.default_rng(42)

    base_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_samples,
    )

    low_noise = 0.5 * base_noise
    medium_noise = 1.0 * base_noise
    high_noise = 2.0 * base_noise

    low_floor = estimate_noise_floor(
        calculate_test_psd(
            low_noise,
            sample_rate_hz,
        )
    )

    medium_floor = estimate_noise_floor(
        calculate_test_psd(
            medium_noise,
            sample_rate_hz,
        )
    )

    high_floor = estimate_noise_floor(
        calculate_test_psd(
            high_noise,
            sample_rate_hz,
        )
    )

    assert (
        low_floor.noise_floor_db
        < medium_floor.noise_floor_db
        < high_floor.noise_floor_db
    )


def test_noise_floor_scaling_matches_expected_db_change():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_samples,
    )

    normal_floor = estimate_noise_floor(
        calculate_test_psd(
            noise,
            sample_rate_hz,
        )
    )

    doubled_floor = estimate_noise_floor(
        calculate_test_psd(
            2.0 * noise,
            sample_rate_hz,
        )
    )

    difference_db = (
        doubled_floor.noise_floor_db
        - normal_floor.noise_floor_db
    )

    assert np.isclose(
        difference_db,
        10.0 * np.log10(4.0),
        atol=0.1,
    )


def test_noise_floor_supports_frequency_range():
    sample_rate_hz = 10000.0

    rng = np.random.default_rng(42)

    samples = rng.normal(
        loc=0.0,
        scale=1.0,
        size=100000,
    )

    psd = calculate_test_psd(
        samples,
        sample_rate_hz,
    )

    result = estimate_noise_floor(
        psd,
        lower_frequency_hz=100.0,
        upper_frequency_hz=1000.0,
    )

    assert result.lower_frequency_hz == 100.0
    assert result.upper_frequency_hz == 1000.0
    assert result.number_of_bins > 0
    assert result.method == "median_psd_db"


def test_noise_floor_rejects_invalid_frequency_range():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    psd = calculate_test_psd(
        samples
    )

    with pytest.raises(
        ValueError,
        match="greater than lower_frequency_hz",
    ):
        estimate_noise_floor(
            psd,
            lower_frequency_hz=1000.0,
            upper_frequency_hz=500.0,
        )


def test_noise_floor_rejects_frequency_above_nyquist():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    psd = calculate_test_psd(
        samples
    )

    with pytest.raises(
        ValueError,
        match="Nyquist",
    ):
        estimate_noise_floor(
            psd,
            lower_frequency_hz=1000.0,
            upper_frequency_hz=6000.0,
        )