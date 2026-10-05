import numpy as np

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.pipeline.single_channel import (
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)


SAMPLE_RATE_HZ = 10000.0
DURATION_SECONDS = 10.0
NUMBER_OF_SAMPLES = int(
    SAMPLE_RATE_HZ * DURATION_SECONDS
)

TIME = (
    np.arange(NUMBER_OF_SAMPLES)
    / SAMPLE_RATE_HZ
)


def make_config() -> SingleChannelAnalysisConfig:
    return SingleChannelAnalysisConfig(
        spectrum_window="hann",

        psd_nperseg=10000,
        psd_noverlap=5000,
        psd_window="hann",

        spectrogram_nperseg=1000,
        spectrogram_noverlap=500,
        spectrogram_window="hann",

        peak_min_prominence=0.1,
        peak_min_distance_hz=20.0,

        band_power_ranges_hz=(
            (0.0, 1000.0),
            (1000.0, 5000.0),
        ),

        noise_floor_lower_frequency_hz=0.0,
        noise_floor_upper_frequency_hz=5000.0,

        narrowband_neighbourhood_width_hz=100.0,
        narrowband_excluded_peak_width_hz=5.0,

        harmonic_tolerance_hz=1.0,
        harmonic_max_order=10,
        harmonic_minimum_matches=3,

        characterization_broadband_flatness_threshold=0.80,
        characterization_tonal_flatness_threshold=0.10,
        characterization_strong_line_threshold_db=10.0,
        characterization_harmonic_minimum_matches=3,
    )


def analyze(samples: np.ndarray):
    record = SignalRecord(
        samples=samples,
        sample_rate_hz=SAMPLE_RATE_HZ,
    )

    return analyze_single_channel(
        record,
        make_config(),
    )


def test_v02_white_noise_is_broadband_dominant():
    rng = np.random.default_rng(42)

    noise = rng.normal(
        size=NUMBER_OF_SAMPLES,
    )

    result = analyze(noise)

    assert result.characterization is not None

    assert (
        result.characterization.label
        == "broadband-dominant"
    )

    assert result.spectral_flatness is not None

    assert (
        result.spectral_flatness.spectral_flatness
        >= 0.80
    )

    assert (
        result.characterization.strong_line_count
        == 0
    )


def test_v02_strong_tone_with_weak_noise_is_tonal():
    rng = np.random.default_rng(42)

    noise = 0.2 * rng.normal(
        size=NUMBER_OF_SAMPLES,
    )

    tone = 3.0 * np.sin(
        2.0 * np.pi * 1000.0 * TIME
    )

    result = analyze(
        noise + tone
    )

    assert result.characterization is not None

    assert (
        result.characterization.label
        == "tonal-dominant"
    )

    assert (
        result.characterization
        .strong_line_count
        >= 1
    )

    assert (
        result.characterization
        .strongest_line_to_floor_db
        > 10.0
    )


def test_v02_noise_plus_strong_tone_is_mixed():
    rng = np.random.default_rng(42)

    noise = rng.normal(
        size=NUMBER_OF_SAMPLES,
    )

    tone = 3.0 * np.sin(
        2.0 * np.pi * 1000.0 * TIME
    )

    result = analyze(
        noise + tone
    )

    assert result.characterization is not None

    assert (
        result.characterization.label
        == "mixed"
    )

    assert (
        result.characterization
        .strong_line_count
        >= 1
    )


def test_v02_harmonic_series_is_harmonic_rich():
    rng = np.random.default_rng(42)

    noise = rng.normal(
        size=NUMBER_OF_SAMPLES,
    )

    samples = (
        noise
        + 3.0 * np.sin(
            2.0 * np.pi * 200.0 * TIME
        )
        + 2.0 * np.sin(
            2.0 * np.pi * 400.0 * TIME
        )
        + 1.5 * np.sin(
            2.0 * np.pi * 600.0 * TIME
        )
        + 1.0 * np.sin(
            2.0 * np.pi * 800.0 * TIME
        )
    )

    result = analyze(samples)

    assert result.characterization is not None

    assert (
        result.characterization.label
        == "harmonic-rich"
    )

    assert result.harmonic_family is not None

    assert (
        result.harmonic_family.best_family
        is not None
    )

    assert np.isclose(
        result.harmonic_family
        .best_family
        .candidate_fundamental_hz,
        200.0,
        atol=1.0,
    )

    assert (
        len(
            result.harmonic_family
            .best_family
            .matches
        )
        >= 4
    )


def test_v02_detects_missing_fundamental():
    samples = (
        2.0 * np.sin(
            2.0 * np.pi * 100.0 * TIME
        )
        + 1.8 * np.sin(
            2.0 * np.pi * 150.0 * TIME
        )
        + 1.5 * np.sin(
            2.0 * np.pi * 200.0 * TIME
        )
        + 1.2 * np.sin(
            2.0 * np.pi * 250.0 * TIME
        )
    )

    result = analyze(samples)

    assert result.harmonic_family is not None

    assert (
        result.harmonic_family.best_family
        is not None
    )

    assert np.isclose(
        result.harmonic_family
        .best_family
        .candidate_fundamental_hz,
        50.0,
        atol=1.0,
    )

    assert (
        len(
            result.harmonic_family
            .best_family
            .matches
        )
        >= 4
    )


def test_v02_unrelated_tones_are_not_harmonic_rich():
    samples = (
        np.sin(
            2.0 * np.pi * 470.0 * TIME
        )
        + np.sin(
            2.0 * np.pi * 913.0 * TIME
        )
    )

    result = analyze(samples)

    assert result.characterization is not None

    assert (
        result.characterization.label
        != "harmonic-rich"
    )

    if result.harmonic_family is not None:
        assert (
            result.harmonic_family.best_family
            is None
        )


def test_v02_low_frequency_signal_has_lower_centroid():
    rng = np.random.default_rng(42)

    noise = 0.2 * rng.normal(
        size=NUMBER_OF_SAMPLES,
    )

    low_signal = (
        noise
        + 2.0 * np.sin(
            2.0 * np.pi * 300.0 * TIME
        )
    )

    high_signal = (
        noise
        + 2.0 * np.sin(
            2.0 * np.pi * 3000.0 * TIME
        )
    )

    low_result = analyze(
        low_signal
    )

    high_result = analyze(
        high_signal
    )

    assert (
        low_result.spectral_centroid
        is not None
    )

    assert (
        high_result.spectral_centroid
        is not None
    )

    assert (
        low_result.spectral_centroid
        .spectral_centroid_hz
        <
        high_result.spectral_centroid
        .spectral_centroid_hz
    )


def test_v02_band_power_tracks_frequency_location():
    rng = np.random.default_rng(42)

    noise = 0.1 * rng.normal(
        size=NUMBER_OF_SAMPLES,
    )

    low_signal = (
        noise
        + 3.0 * np.sin(
            2.0 * np.pi * 300.0 * TIME
        )
    )

    high_signal = (
        noise
        + 3.0 * np.sin(
            2.0 * np.pi * 3000.0 * TIME
        )
    )

    low_result = analyze(
        low_signal
    )

    high_result = analyze(
        high_signal
    )

    low_low_band = (
        low_result.band_powers[0]
        .fraction_of_total_power
    )

    low_high_band = (
        low_result.band_powers[1]
        .fraction_of_total_power
    )

    high_low_band = (
        high_result.band_powers[0]
        .fraction_of_total_power
    )

    high_high_band = (
        high_result.band_powers[1]
        .fraction_of_total_power
    )

    assert (
        low_low_band
        > low_high_band
    )

    assert (
        high_high_band
        > high_low_band
    )


def test_v02_noise_floor_increases_with_noise_amplitude():
    rng = np.random.default_rng(42)

    base_noise = rng.normal(
        size=NUMBER_OF_SAMPLES,
    )

    low_result = analyze(
        0.5 * base_noise
    )

    high_result = analyze(
        2.0 * base_noise
    )

    assert low_result.noise_floor is not None
    assert high_result.noise_floor is not None

    assert (
        high_result.noise_floor.noise_floor_db
        >
        low_result.noise_floor.noise_floor_db
    )


def test_v02_constant_signal_is_handled_safely():
    samples = np.full(
        NUMBER_OF_SAMPLES,
        3.0,
    )

    result = analyze(samples)

    assert result.is_constant

    assert result.noise_floor is None
    assert result.spectral_flatness is None
    assert result.spectral_centroid is None
    assert result.spectral_spread is None
    assert result.psd_percentiles is None
    assert result.narrowband_lines is None
    assert result.harmonic_family is None
    assert result.characterization is None