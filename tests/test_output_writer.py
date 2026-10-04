import json

import numpy as np

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

def test_save_analysis_json_creates_expected_files(tmp_path):
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
        output_dir=tmp_path,
    )

    assert config_path.exists()
    assert results_path.exists()


def test_saved_config_contains_analysis_parameters(tmp_path):
    _, samples = generate_two_tone(
        frequency_1_hz=1000.0,
        amplitude_1=1.0,
        frequency_2_hz=1800.0,
        amplitude_2=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=2.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    config = SingleChannelAnalysisConfig(
        spectrum_window="hann",
        psd_nperseg=2000,
        psd_noverlap=1000,
        psd_window="hann",
        spectrogram_nperseg=1000,
        spectrogram_noverlap=500,
        spectrogram_window="hann",
        peak_min_prominence=0.1,
        peak_min_distance_hz=100.0,
    )

    result = analyze_single_channel(
        record,
        config,
    )

    config_path, _ = save_analysis_json(
        record,
        result,
        tmp_path,
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_config = json.load(file)

    assert saved_config["spectrum_window"] == "hann"
    assert saved_config["psd_nperseg"] == 2000
    assert saved_config["psd_noverlap"] == 1000
    assert saved_config["peak_min_prominence"] == 0.1
    assert saved_config["peak_min_distance_hz"] == 100.0

def test_saved_config_contains_analysis_parameters(tmp_path):
    _, samples = generate_two_tone(
        frequency_1_hz=1000.0,
        amplitude_1=1.0,
        frequency_2_hz=1800.0,
        amplitude_2=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=2.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    config = SingleChannelAnalysisConfig(
        spectrum_window="hann",
        psd_nperseg=2000,
        psd_noverlap=1000,
        psd_window="hann",
        spectrogram_nperseg=1000,
        spectrogram_noverlap=500,
        spectrogram_window="hann",
        peak_min_prominence=0.1,
        peak_min_distance_hz=100.0,
    )

    result = analyze_single_channel(
        record,
        config,
    )

    config_path, _ = save_analysis_json(
        record,
        result,
        tmp_path,
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_config = json.load(file)

    assert saved_config["spectrum_window"] == "hann"
    assert saved_config["psd_nperseg"] == 2000
    assert saved_config["psd_noverlap"] == 1000
    assert saved_config["peak_min_prominence"] == 0.1
    assert saved_config["peak_min_distance_hz"] == 100.0

def test_saved_results_contain_expected_signal_and_peaks(tmp_path):
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
        record,
        config,
    )

    _, results_path = save_analysis_json(
        record,
        result,
        tmp_path,
    )

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_results = json.load(file)

    signal = saved_results["signal"]

    assert signal["channel_name"] == "two_tone_reference"
    assert signal["units"] == "V"
    assert signal["number_of_samples"] == 50000

    assert np.isclose(
        signal["sample_rate_hz"],
        10000.0,
    )

    saved_peak_frequencies = np.array(
        [
            peak["frequency_hz"]
            for peak in saved_results["peaks"]
        ]
    )

    assert np.any(
        np.isclose(
            saved_peak_frequencies,
            1000.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

    assert np.any(
        np.isclose(
            saved_peak_frequencies,
            1800.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

def test_save_analysis_plots_creates_expected_files(tmp_path):
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
        output_dir=tmp_path,
    )

    waveform_path, spectrum_path = save_analysis_plots(
        record=record,
        result=result,
        output_dir=tmp_path,
    )

    assert config_path.exists()
    assert results_path.exists()

    assert waveform_path.exists()
    assert spectrum_path.exists()

    assert waveform_path.stat().st_size > 0
    assert spectrum_path.stat().st_size > 0

    assert waveform_path.name == "waveform.png"
    assert spectrum_path.name == "spectrum.png"