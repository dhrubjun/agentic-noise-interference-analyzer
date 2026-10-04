from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.output.writer import (
    save_analysis_json,
    save_analysis_plots,
)
from noise_analyzer.pipeline.single_channel import (
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)
from noise_analyzer.synthetic.signals import generate_two_tone


# Create a known two-tone test signal
_, samples = generate_two_tone(
    frequency_1_hz=1000.0,
    amplitude_1=1.0,
    frequency_2_hz=1800.0,
    amplitude_2=0.5,
    sample_rate_hz=10000.0,
    duration_seconds=5.0,
)

# Put the samples into our standard SignalRecord
record = SignalRecord(
    samples=samples,
    sample_rate_hz=10000.0,
    channel_name="two_tone_reference",
    units="V",
)

# Define the analysis settings
config = SingleChannelAnalysisConfig(
    spectrum_window=None,
    psd_nperseg=10000,
    psd_noverlap=5000,
    psd_window="hann",
    spectrogram_nperseg=1000,
    spectrogram_noverlap=500,
    spectrogram_window="hann",
    peak_min_prominence=0.1,
)

# Run the full single-channel analysis
result = analyze_single_channel(
    record=record,
    config=config,
)

# Save JSON results
config_path, results_path = save_analysis_json(
    record=record,
    result=result,
    output_dir="outputs/test_run",
)

# Save all plots
(
    waveform_path,
    spectrum_path,
    psd_path,
    asd_path,
    spectrogram_path,
) = save_analysis_plots(
    record=record,
    result=result,
    output_dir="outputs/test_run",
)

# Show what was created
print("Saved JSON:")
print(config_path)
print(results_path)

print()

print("Saved plots:")
print(waveform_path)
print(spectrum_path)
print(psd_path)
print(asd_path)
print(spectrogram_path)