from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.output.writer import save_analysis_json
from noise_analyzer.pipeline.single_channel import (
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)
from noise_analyzer.synthetic.signals import generate_two_tone


_, samples = generate_two_tone(
    frequency_1_hz=1000.0,
    amplitude_1=1.0,
    frequency_2_hz=1800.0,
    amplitude_2=0.5,
    sample_rate_hz=10000.0,
    duration_seconds=5.0,
)

record = SignalRecord(
    samples=samples,
    sample_rate_hz=10000.0,
    channel_name="two_tone_reference",
    units="V",
)

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

result = analyze_single_channel(
    record=record,
    config=config,
)

config_path, results_path = save_analysis_json(
    record=record,
    result=result,
    output_dir="outputs/test_run",
)

print("Saved:")
print(config_path)
print(results_path)