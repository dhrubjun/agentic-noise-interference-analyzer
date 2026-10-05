import numpy as np

from noise_analyzer.dsp.characterization import (
    characterize_spectral_behavior,
)
from noise_analyzer.dsp.harmonics import (
    find_best_harmonic_family,
)
from noise_analyzer.dsp.narrowband import (
    characterize_detected_lines,
)
from noise_analyzer.dsp.peaks import (
    detect_spectral_peaks,
)
from noise_analyzer.dsp.psd import (
    calculate_welch_psd,
)
from noise_analyzer.dsp.spectral_features import (
    calculate_spectral_flatness,
)
from noise_analyzer.dsp.spectrum import (
    calculate_amplitude_spectrum,
)
from noise_analyzer.models.signal import SignalRecord


def characterize_samples(
    samples: np.ndarray,
    sample_rate_hz: float = 10000.0,
):
    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
    )

    spectrum = calculate_amplitude_spectrum(
        record,
        window="hann",
    )

    psd = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
        window="hann",
    )

    peaks = detect_spectral_peaks(
        spectrum,
        min_prominence=0.1,
        min_distance_hz=20.0,
    )

    flatness = calculate_spectral_flatness(psd)

    narrowband = characterize_detected_lines(
        psd,
        peaks,
        neighbourhood_width_hz=100.0,
        excluded_peak_width_hz=5.0,
    )

    harmonics = find_best_harmonic_family(
        peaks,
        tolerance_hz=1.0,
        max_harmonic_order=10,
        minimum_matches=3,
    )

    return characterize_spectral_behavior(
        flatness,
        narrowband,
        harmonics,
    )

def test_white_noise_is_broadband_dominant():
    rng = np.random.default_rng(42)

    samples = rng.normal(
        size=100000,
    )

    result = characterize_samples(samples)

    assert result.label == "broadband-dominant"

def test_strong_tone_with_weak_noise_is_tonal_dominant():
    sample_rate_hz = 10000.0

    time = (
        np.arange(100000)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    noise = 0.2 * rng.normal(
        size=time.size,
    )

    samples = (
        noise
        + 3.0 * np.sin(
            2.0 * np.pi * 1000.0 * time
        )
    )

    result = characterize_samples(
        samples,
        sample_rate_hz,
    )

    assert result.label == "tonal-dominant"

def test_noise_plus_strong_tone_is_mixed():
    sample_rate_hz = 10000.0

    time = (
        np.arange(100000)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        size=time.size,
    )

    samples = (
        noise
        + 3.0 * np.sin(
            2.0 * np.pi * 1000.0 * time
        )
    )

    result = characterize_samples(
        samples,
        sample_rate_hz,
    )

    assert result.label == "mixed"

def test_harmonic_series_is_harmonic_rich():
    sample_rate_hz = 10000.0

    time = (
        np.arange(100000)
        / sample_rate_hz
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        size=time.size,
    )

    samples = (
        noise
        + 3.0 * np.sin(
            2.0 * np.pi * 200.0 * time
        )
        + 2.0 * np.sin(
            2.0 * np.pi * 400.0 * time
        )
        + 1.5 * np.sin(
            2.0 * np.pi * 600.0 * time
        )
        + 1.0 * np.sin(
            2.0 * np.pi * 800.0 * time
        )
    )

    result = characterize_samples(
        samples,
        sample_rate_hz,
    )

    assert result.label == "harmonic-rich"

    assert result.harmonic_match_count >= 3
    assert result.strong_line_count >= 3