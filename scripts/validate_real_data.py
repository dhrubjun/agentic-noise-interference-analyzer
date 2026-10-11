from noise_analyzer.io.loaders import load_signal_csv
from noise_analyzer.output.writer import (
    save_analysis_json,
    save_analysis_plots,
)
from noise_analyzer.pipeline.single_channel import (
    EventAnalysisConfig,
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)


# Load real vibration measurement
record = load_signal_csv(
    file_path="test_data/real/H_WD1_WS1_01.csv",
    sample_rate_hz=1024.0,
    channel_name="H_WD1_WS1_01",
    units="V",
    has_header=False,
)


print("Loaded real-world signal")
print("------------------------")
print(f"Samples: {record.number_of_samples}")
print(f"Sample rate: {record.sample_rate_hz} Hz")
print(f"Duration: {record.duration_seconds} s")
print(f"Nyquist: {record.nyquist_frequency_hz} Hz")


config = SingleChannelAnalysisConfig(
    spectrum_window="hann",

    psd_nperseg=4096,
    psd_noverlap=2048,
    psd_window="hann",

    spectrogram_nperseg=1024,
    spectrogram_noverlap=512,
    spectrogram_window="hann",

    peak_min_prominence=5e-5,
    peak_min_distance_hz=1.0,

    band_power_ranges_hz=(
        (0.0, 10.0),
        (10.0, 50.0),
        (50.0, 100.0),
        (100.0, 250.0),
        (250.0, 512.0),
    ),

    noise_floor_lower_frequency_hz=1.0,
    noise_floor_upper_frequency_hz=500.0,

    narrowband_neighbourhood_width_hz=10.0,
    narrowband_excluded_peak_width_hz=1.0,

    harmonic_tolerance_hz=0.5,
    harmonic_max_order=10,
    harmonic_minimum_matches=3,

    characterization_broadband_flatness_threshold=0.80,
    characterization_tonal_flatness_threshold=0.10,
    characterization_strong_line_threshold_db=10.0,
    characterization_harmonic_minimum_matches=3,

    event_analysis_config=EventAnalysisConfig(
        frame_length_samples=1024,
        hop_length_samples=256,
        start_threshold=6.0,
        end_threshold=3.0,
        minimum_event_duration_s=0.5,
        merge_gap_s=0.25,
    ),

    
)


result = analyze_single_channel(
    record=record,
    config=config,
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
print("Detected spectral peaks")
print("-----------------------")

for peak in result.peaks.peaks:
    print(
        f"{peak.frequency_hz:.3f} Hz"
        f" | amplitude={peak.amplitude:.6g}"
        f" | prominence={peak.prominence:.6g}"
    )

print()
print("V0.2 characterization")
print("---------------------")

if result.noise_floor is None:
    print("Noise floor: unavailable")
else:
    print(
        f"Global noise floor: "
        f"{result.noise_floor.noise_floor_db:.3f} dB"
    )

if result.spectral_flatness is None:
    print("Spectral flatness: unavailable")
else:
    print(
        f"Spectral flatness: "
        f"{result.spectral_flatness.spectral_flatness:.6f}"
    )

if result.spectral_centroid is None:
    print("Spectral centroid: unavailable")
else:
    print(
        f"Spectral centroid: "
        f"{result.spectral_centroid.spectral_centroid_hz:.3f} Hz"
    )

if result.spectral_spread is None:
    print("Spectral spread: unavailable")
else:
    print(
        f"Spectral spread: "
        f"{result.spectral_spread.spectral_spread_hz:.3f} Hz"
    )

print()
print("Band powers")
print("-----------")

for band in result.band_powers:
    print(
        f"{band.lower_frequency_hz:.1f}"
        f"–{band.upper_frequency_hz:.1f} Hz"
        f" | power={band.band_power:.6g}"
        f" | fraction="
        f"{100.0 * band.fraction_of_total_power:.2f}%"
    )

print()
print("Narrowband characterization")
print("---------------------------")

if result.narrowband_lines is None:
    print("Unavailable")
else:
    for line in result.narrowband_lines.lines:
        print(
            f"{line.frequency_hz:.3f} Hz"
            f" | peak={line.peak_level_db:.2f} dB"
            f" | local floor="
            f"{line.local_noise_floor_db:.2f} dB"
            f" | line/floor="
            f"{line.line_to_floor_db:.2f} dB"
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
        f"{family.candidate_fundamental_hz:.3f} Hz"
    )

    print(
        f"Harmonic matches: "
        f"{len(family.matches)}"
    )

    for match in family.matches:
        print(
            f"  order {match.harmonic_order}"
            f" | expected="
            f"{match.expected_frequency_hz:.3f} Hz"
            f" | detected="
            f"{match.detected_frequency_hz:.3f} Hz"
            f" | error="
            f"{match.frequency_error_hz:.3f} Hz"
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

    if (
        result.characterization
        .strongest_line_to_floor_db
        is not None
    ):
        print(
            f"Strongest line/floor: "
            f"{result.characterization.strongest_line_to_floor_db:.2f} dB"
        )

print()
print("V0.3 event and anomaly analysis")
print("-------------------------------")

if result.event_analysis is None:
    print("Event analysis unavailable.")

else:
    event_analysis = result.event_analysis

    print(
        f"Feature: short-time RMS"
    )

    print(
        f"Baseline RMS: "
        f"{event_analysis.baseline.location:.8g}"
    )

    print(
        f"Baseline MAD: "
        f"{event_analysis.baseline.mad:.8g}"
    )

    print(
        f"Robust scale: "
        f"{event_analysis.baseline.robust_scale:.8g}"
    )

    print(
        f"Baseline status: "
        f"{event_analysis.baseline.status}"
    )

    print(
        f"Raw events: "
        f"{len(event_analysis.raw_events)}"
    )

    print(
        f"Final events: "
        f"{len(event_analysis.events)}"
    )

    if len(event_analysis.events) == 0:
        print(
            "No RMS events exceeded the configured "
            "detection thresholds."
        )

    else:
        print()
        print("Detected events")
        print("---------------")

        for event_number, (
            event,
            measurement,
        ) in enumerate(
            zip(
                event_analysis.events,
                event_analysis.measurements,
            ),
            start=1,
        ):
            print(
                f"Event {event_number}"
            )

            print(
                f"  Start: "
                f"{event.start_time_s:.3f} s"
            )

            print(
                f"  End: "
                f"{event.end_time_s:.3f} s"
            )

            print(
                f"  Support duration: "
                f"{measurement.support_duration_s:.3f} s"
            )

            print(
                f"  Peak time: "
                f"{event.peak_time_s:.3f} s"
            )

            print(
                f"  Peak deviation score: "
                f"{event.peak_score:.3f}"
            )

            print(
                f"  Baseline RMS: "
                f"{measurement.baseline_rms:.8g}"
            )

            print(
                f"  Mean event RMS: "
                f"{measurement.mean_rms:.8g}"
            )

            print(
                f"  Peak event RMS: "
                f"{measurement.peak_rms:.8g}"
            )

            print(
                f"  Peak RMS / baseline: "
                f"{measurement.rms_ratio_to_baseline:.3f}x"
            )


output_dir = "outputs/real_H_WD1_WS1_01"

save_analysis_json(
    record=record,
    result=result,
    output_dir=output_dir,
)

save_analysis_plots(
    record=record,
    result=result,
    output_dir=output_dir,
)


print()
print(f"Analysis saved to: {output_dir}")