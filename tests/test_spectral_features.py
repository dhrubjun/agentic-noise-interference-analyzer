import numpy as np
import pytest

from noise_analyzer.dsp.noise_floor import (
    estimate_noise_floor,
)
from noise_analyzer.dsp.psd import (
    calculate_welch_psd,
)
from noise_analyzer.dsp.spectral_features import (
    calculate_occupied_bandwidth,
    calculate_psd_percentiles,
    calculate_spectral_centroid,
    calculate_spectral_flatness,
    calculate_spectral_spread,
)
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import (
    generate_sine,
)


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


def test_white_noise_has_higher_flatness_than_tone():
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

    _, tone = generate_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )

    noise_flatness = calculate_spectral_flatness(
        calculate_test_psd(
            noise,
            sample_rate_hz,
        )
    )

    tone_flatness = calculate_spectral_flatness(
        calculate_test_psd(
            tone,
            sample_rate_hz,
        )
    )

    assert (
        noise_flatness.spectral_flatness
        > tone_flatness.spectral_flatness
    )


def test_spectral_flatness_is_between_zero_and_one():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    result = calculate_spectral_flatness(
        calculate_test_psd(samples)
    )

    assert 0.0 <= result.spectral_flatness <= 1.0


def test_spectral_flatness_supports_frequency_range():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    result = calculate_spectral_flatness(
        calculate_test_psd(samples),
        lower_frequency_hz=100.0,
        upper_frequency_hz=4000.0,
    )

    assert result.lower_frequency_hz == 100.0
    assert result.upper_frequency_hz == 4000.0
    assert result.number_of_bins > 0


def test_spectral_flatness_rejects_invalid_range():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    psd = calculate_test_psd(samples)

    with pytest.raises(
        ValueError,
        match="greater than lower_frequency_hz",
    ):
        calculate_spectral_flatness(
            psd,
            lower_frequency_hz=2000.0,
            upper_frequency_hz=1000.0,
        )


def test_spectral_flatness_rejects_frequency_above_nyquist():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    psd = calculate_test_psd(samples)

    with pytest.raises(
        ValueError,
        match="Nyquist",
    ):
        calculate_spectral_flatness(
            psd,
            lower_frequency_hz=1000.0,
            upper_frequency_hz=6000.0,
        )


def test_high_frequency_tone_has_higher_centroid():
    sample_rate_hz = 10000.0
    duration_seconds = 5.0

    _, low_tone = generate_sine(
        frequency_hz=500.0,
        amplitude=1.0,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )

    _, high_tone = generate_sine(
        frequency_hz=3000.0,
        amplitude=1.0,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )

    low_result = calculate_spectral_centroid(
        calculate_test_psd(
            low_tone,
            sample_rate_hz,
        )
    )

    high_result = calculate_spectral_centroid(
        calculate_test_psd(
            high_tone,
            sample_rate_hz,
        )
    )

    assert (
        high_result.spectral_centroid_hz
        > low_result.spectral_centroid_hz
    )


def test_broadband_noise_has_larger_spread_than_tone():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        size=number_of_samples,
    )

    _, tone = generate_sine(
        frequency_hz=2000.0,
        amplitude=1.0,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )

    noise_spread = calculate_spectral_spread(
        calculate_test_psd(
            noise,
            sample_rate_hz,
        )
    )

    tone_spread = calculate_spectral_spread(
        calculate_test_psd(
            tone,
            sample_rate_hz,
        )
    )

    assert (
        noise_spread.spectral_spread_hz
        > tone_spread.spectral_spread_hz
    )


def test_psd_percentiles_are_ordered():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    result = calculate_psd_percentiles(
        calculate_test_psd(samples)
    )

    assert (
        result.percentile_10_db
        <= result.percentile_25_db
        <= result.percentile_50_db
        <= result.percentile_75_db
        <= result.percentile_90_db
    )


def test_psd_50th_percentile_matches_noise_floor_median():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    psd = calculate_test_psd(samples)

    percentiles = calculate_psd_percentiles(psd)

    noise_floor = estimate_noise_floor(psd)

    assert np.isclose(
        percentiles.percentile_50_db,
        noise_floor.noise_floor_db,
    )


def test_white_noise_has_large_occupied_bandwidth():
    rng = np.random.default_rng(42)

    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    samples = rng.normal(
        size=number_of_samples
    )

    psd = calculate_test_psd(
        samples,
        sample_rate_hz,
    )

    result = calculate_occupied_bandwidth(
        psd,
        power_fraction=0.90,
    )

    assert result.occupied_bandwidth_hz > 4000.0


def test_tone_has_small_occupied_bandwidth():
    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    samples = np.sin(
        2.0
        * np.pi
        * 1000.0
        * time
    )

    psd = calculate_test_psd(
        samples,
        sample_rate_hz,
    )

    result = calculate_occupied_bandwidth(
        psd,
        power_fraction=0.90,
    )

    assert result.occupied_bandwidth_hz < 10.0


def test_white_noise_has_wider_occupied_bandwidth_than_tone():
    rng = np.random.default_rng(42)

    sample_rate_hz = 10000.0
    duration_seconds = 10.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    noise = rng.normal(
        size=number_of_samples
    )

    tone = np.sin(
        2.0
        * np.pi
        * 1000.0
        * time
    )

    noise_psd = calculate_test_psd(
        noise,
        sample_rate_hz,
    )

    tone_psd = calculate_test_psd(
        tone,
        sample_rate_hz,
    )

    noise_bandwidth = calculate_occupied_bandwidth(
        noise_psd
    )

    tone_bandwidth = calculate_occupied_bandwidth(
        tone_psd
    )

    assert (
        noise_bandwidth.occupied_bandwidth_hz
        >
        tone_bandwidth.occupied_bandwidth_hz
    )


def test_occupied_bandwidth_rejects_invalid_fraction():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=10000
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=1000.0,
    )

    psd = calculate_welch_psd(
        record,
        nperseg=1000,
        noverlap=500,
        window="hann",
    )

    with pytest.raises(ValueError):
        calculate_occupied_bandwidth(
            psd,
            power_fraction=1.0,
        )