import numpy as np

from noise_analyzer.dsp.psd import calculate_welch_psd
from noise_analyzer.models.signal import SignalRecord


def estimate_levels(psd_values: np.ndarray) -> dict[str, float]:
    """Compare several candidate global noise-floor estimators."""

    positive_psd = psd_values[psd_values > 0]

    psd_db = 10.0 * np.log10(positive_psd)

    return {
        "mean_db": float(np.mean(psd_db)),
        "median_db": float(np.median(psd_db)),
        "percentile_25_db": float(
            np.percentile(psd_db, 25)
        ),
        "percentile_50_db": float(
            np.percentile(psd_db, 50)
        ),
    }


def calculate_psd(
    samples: np.ndarray,
    sample_rate_hz: float,
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

    tone_1 = 10.0 * np.sin(
        2.0 * np.pi * 1000.0 * time
    )

    tone_2 = 8.0 * np.sin(
        2.0 * np.pi * 2000.0 * time
    )

    tone_3 = 6.0 * np.sin(
        2.0 * np.pi * 3000.0 * time
    )

    many_tones = np.zeros_like(time)

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

    for frequency_hz in frequencies_hz:
        many_tones += 8.0 * np.sin(
            2.0 * np.pi * frequency_hz * time
        )

    signals = {
        "noise_only": noise,
        "noise_plus_one_tone": (
            noise + tone_1
        ),
        "noise_plus_three_tones": (
            noise
            + tone_1
            + tone_2
            + tone_3
        ),
        "noise_plus_many_tones": (
            noise
            + many_tones
        ),
    }

    print()
    print("Noise-floor estimator comparison")
    print("=" * 60)

    for name, samples in signals.items():
        psd = calculate_psd(
            samples,
            sample_rate_hz,
        )

        levels = estimate_levels(
            psd.psd
        )

        print()
        print(name)

        for estimator, value in levels.items():
            print(
                f"  {estimator:20s}: "
                f"{value:8.3f} dB"
            )


if __name__ == "__main__":
    main()