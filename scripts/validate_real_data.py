from noise_analyzer.io.loaders import load_signal_csv
from noise_analyzer.output.writer import (
    save_analysis_json,
    save_analysis_plots,
)
from noise_analyzer.pipeline.single_channel import (
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