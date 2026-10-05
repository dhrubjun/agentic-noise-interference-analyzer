import numpy as np

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


def analyze_signal(
    samples: np.ndarray,
    sample_rate_hz: float,
) -> dict:
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

    flatness = calculate_spectral_flatness(
        psd
    )

    narrowband = characterize_detected_lines(
        psd,
        peaks,
        neighbourhood_width_hz=100.0,
        excluded_peak_width_hz=5.0,
    )

    harmonic = find_best_harmonic_family(
        peaks,
        tolerance_hz=1.0,
        max_harmonic_order=10,
        minimum_matches=3,
    )

    if len(narrowband.lines) > 0:
        strongest_line_to_floor_db = max(
            line.line_to_floor_db
            for line in narrowband.lines
        )
    else:
        strongest_line_to_floor_db = None

    strong_line_count = sum(
        line.line_to_floor_db >= 10.0
        for line in narrowband.lines
    )

    if (
        harmonic.best_family is not None
    ):
        harmonic_match_count = len(
            harmonic.best_family.matches
        )

        harmonic_fundamental_hz = (
            harmonic.best_family
            .candidate_fundamental_hz
        )
    else:
        harmonic_match_count = 0
        harmonic_fundamental_hz = None

    return {
        "spectral_flatness": (
            flatness.spectral_flatness
        ),
        "detected_peak_count": len(
            peaks.peaks
        ),
        "strong_line_count": (
            strong_line_count
        ),
        "strongest_line_to_floor_db": (
            strongest_line_to_floor_db
        ),
        "harmonic_match_count": (
            harmonic_match_count
        ),
        "harmonic_fundamental_hz": (
            harmonic_fundamental_hz
        ),
    }


def main():
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

    pure_noise = noise

    tone_weak_noise = (
        3.0
        * np.sin(
            2.0 * np.pi * 1000.0 * time
        )
        + 0.2 * noise
    )

    noise_plus_strong_tone = (
        noise
        + 3.0
        * np.sin(
            2.0 * np.pi * 1000.0 * time
        )
    )

    harmonic_signal = (
        noise
        + 3.0
        * np.sin(
            2.0 * np.pi * 200.0 * time
        )
        + 2.0
        * np.sin(
            2.0 * np.pi * 400.0 * time
        )
        + 1.5
        * np.sin(
            2.0 * np.pi * 600.0 * time
        )
        + 1.0
        * np.sin(
            2.0 * np.pi * 800.0 * time
        )
    )

    signals = {
        "white_noise": pure_noise,
        "tone_plus_weak_noise": (
            tone_weak_noise
        ),
        "noise_plus_strong_tone": (
            noise_plus_strong_tone
        ),
        "noise_plus_harmonic_series": (
            harmonic_signal
        ),
    }

    print()
    print(
        "V0.2 characterization-rule experiment"
    )
    print("=" * 72)

    for name, samples in signals.items():
        result = analyze_signal(
            samples,
            sample_rate_hz,
        )

        print()
        print(name)

        print(
            "  spectral_flatness        : "
            f"{result['spectral_flatness']:.6f}"
        )

        print(
            "  detected_peak_count      : "
            f"{result['detected_peak_count']}"
        )

        print(
            "  strong_line_count        : "
            f"{result['strong_line_count']}"
        )

        strongest = (
            result[
                "strongest_line_to_floor_db"
            ]
        )

        if strongest is None:
            print(
                "  strongest_line_to_floor  : "
                "None"
            )
        else:
            print(
                "  strongest_line_to_floor  : "
                f"{strongest:.3f} dB"
            )

        print(
            "  harmonic_match_count     : "
            f"{result['harmonic_match_count']}"
        )

        fundamental = (
            result[
                "harmonic_fundamental_hz"
            ]
        )

        if fundamental is None:
            print(
                "  harmonic_fundamental     : "
                "None"
            )
        else:
            print(
                "  harmonic_fundamental     : "
                f"{fundamental:.3f} Hz"
            )


if __name__ == "__main__":
    main()