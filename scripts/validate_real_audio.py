import numpy as np
from scipy.io import wavfile

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.output.writer import (
    save_analysis_json,
    save_analysis_plots,
)
from noise_analyzer.pipeline.single_channel import (
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)


sample_rate_hz, samples = wavfile.read(
    "test_data/real/1-137296-A-16.wav"
)

samples = np.asarray(
    samples,
    dtype=float,
)

if samples.ndim == 2:
    samples = np.mean(
        samples,
        axis=1,
    )

peak_abs = np.max(
    np.abs(samples)
)

if peak_abs > 0:
    samples = samples / peak_abs


record = SignalRecord(
    samples=samples,
    sample_rate_hz=float(sample_rate_hz),
    channel_name="1-137296-A-16",
    units="normalized amplitude",
)


print("Loaded 1-137-A-32 audio")
print("-------------------")
print(f"Samples: {record.number_of_samples}")
print(f"Sample rate: {record.sample_rate_hz} Hz")
print(f"Duration: {record.duration_seconds:.3f} s")
print(f"Nyquist: {record.nyquist_frequency_hz} Hz")


config = SingleChannelAnalysisConfig(
    spectrum_window="hann",

    psd_nperseg=8192,
    psd_noverlap=4096,
    psd_window="hann",

    spectrogram_nperseg=2048,
    spectrogram_noverlap=1024,
    spectrogram_window="hann",

    peak_min_prominence=0.01,
    peak_min_distance_hz=20.0,

    band_power_ranges_hz=(
        (0.0, 250.0),
        (250.0, 1000.0),
        (1000.0, 4000.0),
        (4000.0, 10000.0),
        (10000.0, 22050.0),
    ),

    noise_floor_lower_frequency_hz=20.0,
    noise_floor_upper_frequency_hz=20000.0,

    narrowband_neighbourhood_width_hz=200.0,
    narrowband_excluded_peak_width_hz=20.0,

    harmonic_tolerance_hz=5.0,
    harmonic_max_order=10,
    harmonic_minimum_matches=3,
    harmonic_relative_tolerance_fraction=0.01,
    harmonic_minimum_match_fraction=0.5,
    harmonic_minimum_consecutive_matches=3,

    characterization_broadband_flatness_threshold=0.80,
    characterization_tonal_flatness_threshold=0.10,
    characterization_strong_line_threshold_db=10.0,
    characterization_harmonic_minimum_matches=3,
)


result = analyze_single_channel(
    record,
    config,
)


print()
print("Time-domain statistics")
print("----------------------")
print(f"Mean: {result.statistics.mean}")
print(f"RMS: {result.statistics.rms}")
print(
    f"Standard deviation: "
    f"{result.statistics.standard_deviation}"
)


print()
print("V0.2 characterization")
print("---------------------")

if result.noise_floor is None:
    print("Noise floor: unavailable")
else:
    print(
        f"Noise floor: "
        f"{result.noise_floor.noise_floor_db:.3f} dB"
    )

if result.spectral_flatness is not None:
    print(
        f"Spectral flatness: "
        f"{result.spectral_flatness.spectral_flatness:.6f}"
    )

if result.spectral_centroid is not None:
    print(
        f"Spectral centroid: "
        f"{result.spectral_centroid.spectral_centroid_hz:.3f} Hz"
    )

if result.spectral_spread is not None:
    print(
        f"Spectral spread: "
        f"{result.spectral_spread.spectral_spread_hz:.3f} Hz"
    )


print()
print("Band powers")
print("-----------")

for band in result.band_powers:
    print(
        f"{band.lower_frequency_hz:.0f}"
        f"–{band.upper_frequency_hz:.0f} Hz"
        f" | fraction="
        f"{100.0 * band.fraction_of_total_power:.2f}%"
    )


print()
print("Narrowband lines")
print("----------------")

if result.narrowband_lines is None:
    print("Unavailable")
else:
    strong_lines = [
        line
        for line in result.narrowband_lines.lines
        if line.line_to_floor_db >= 10.0
    ]

    for line in strong_lines:
        print(
            f"{line.frequency_hz:.1f} Hz"
            f" | line/floor="
            f"{line.line_to_floor_db:.2f} dB"
        )

    print(
        f"Strong-line count: "
        f"{len(strong_lines)}"
    )


print()
print("Harmonic analysis")
print("-----------------")

if (
    result.harmonic_family is None
    or result.harmonic_family.best_family is None
):
    print("No convincing harmonic family detected.")
else:
    family = result.harmonic_family.best_family

    print(
        f"Candidate fundamental: "
        f"{family.candidate_fundamental_hz:.2f} Hz"
    )

    print(
        f"Matches: "
        f"{len(family.matches)}"
    )

    print(
        f"Match fraction: "
        f"{family.match_fraction:.3f}"
    )

    print(
        f"Longest consecutive run: "
        f"{family.longest_consecutive_run}"
    )


print()
print("Final characterization")
print("----------------------")

if result.characterization is None:
    print("Unavailable")
else:
    print(
        f"Label: "
        f"{result.characterization.label}"
    )

    print(
        f"Strong narrowband lines: "
        f"{result.characterization.strong_line_count}"
    )

    print(
        f"Harmonic matches: "
        f"{result.characterization.harmonic_match_count}"
    )

if result.occupied_bandwidth is None:
    print("Occupied bandwidth: unavailable")
else:
    print(
        f"90% occupied bandwidth: "
        f"{result.occupied_bandwidth.occupied_bandwidth_hz:.3f} Hz"
    )

    print(
        f"Occupied range: "
        f"{result.occupied_bandwidth.lower_edge_hz:.3f}"
        f"–"
        f"{result.occupied_bandwidth.upper_edge_hz:.3f} Hz"
    )


output_dir = "outputs/real_esc50_sample"

save_analysis_json(
    record,
    result,
    output_dir,
)

save_analysis_plots(
    record,
    result,
    output_dir,
)

print()
print(
    f"Analysis saved to: "
    f"{output_dir}"
)