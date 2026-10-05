import numpy as np

from characterization_rules_experiment import analyze_signal


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

    base_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_samples,
    )

    tone_amplitudes = [
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    ]

    noise_levels = [
        0.2,
        0.5,
        1.0,
        2.0,
    ]

    print()
    print("Characterization threshold sweep")
    print("=" * 90)

    print(
        f"{'Tone':>8}"
        f"{'Noise':>10}"
        f"{'Flatness':>14}"
        f"{'Line/Floor':>16}"
        f"{'Strong Lines':>16}"
    )

    print("-" * 90)

    for noise_level in noise_levels:
        for tone_amplitude in tone_amplitudes:
            samples = (
                noise_level * base_noise
                + tone_amplitude
                * np.sin(
                    2.0
                    * np.pi
                    * 1000.0
                    * time
                )
            )

            result = analyze_signal(
                samples,
                sample_rate_hz,
            )

            strongest = result[
                "strongest_line_to_floor_db"
            ]

            if strongest is None:
                strongest_text = "None"
            else:
                strongest_text = (
                    f"{strongest:.2f}"
                )

            print(
                f"{tone_amplitude:8.2f}"
                f"{noise_level:10.2f}"
                f"{result['spectral_flatness']:14.4f}"
                f"{strongest_text:>16}"
                f"{result['strong_line_count']:16d}"
            )


if __name__ == "__main__":
    main()