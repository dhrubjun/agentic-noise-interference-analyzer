import json

import numpy as np

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.output.writer import (
    save_analysis_json,
    save_analysis_plots,
)
from noise_analyzer.pipeline.single_channel import (
    EventAnalysisConfig,
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


def test_saved_config_contains_basic_analysis_parameters(
    tmp_path,
):
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


def test_saved_config_contains_characterization_parameters(
    tmp_path,
):
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

    assert (
        saved_config[
            "characterization_broadband_flatness_threshold"
        ]
        == 0.80
    )

    assert (
        saved_config[
            "characterization_tonal_flatness_threshold"
        ]
        == 0.10
    )

    assert (
        saved_config[
            "characterization_strong_line_threshold_db"
        ]
        == 10.0
    )

    assert (
        saved_config["harmonic_minimum_matches"]
        == 3
    )


def test_saved_results_contain_expected_signal_and_peaks(
    tmp_path,
):
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


def test_save_analysis_plots_creates_expected_files(
    tmp_path,
):
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

    (
        waveform_path,
        spectrum_path,
        psd_path,
        asd_path,
        spectrogram_path,
    ) = save_analysis_plots(
        record=record,
        result=result,
        output_dir=tmp_path,
    )

    expected_paths = [
        waveform_path,
        spectrum_path,
        psd_path,
        asd_path,
        spectrogram_path,
    ]

    for path in expected_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    assert waveform_path.name == "waveform.png"
    assert spectrum_path.name == "spectrum.png"
    assert psd_path.name == "psd.png"
    assert asd_path.name == "asd.png"
    assert spectrogram_path.name == "spectrogram.png"


def test_saved_results_contain_v02_characterization(
    tmp_path,
):
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

    samples = (
        rng.normal(
            size=number_of_samples,
        )
        + 3.0 * np.sin(
            2.0 * np.pi * 1000.0 * time
        )
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        channel_name="v02_test_signal",
        units="V",
    )

    config = SingleChannelAnalysisConfig(
        spectrum_window="hann",
        psd_nperseg=10000,
        psd_noverlap=5000,
        psd_window="hann",
        spectrogram_nperseg=1000,
        spectrogram_noverlap=500,
        spectrogram_window="hann",
        peak_min_prominence=0.1,
        peak_min_distance_hz=20.0,
        band_power_ranges_hz=(
            (0.0, 1000.0),
            (1000.0, 5000.0),
        ),
        noise_floor_lower_frequency_hz=100.0,
        noise_floor_upper_frequency_hz=4000.0,
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

    assert len(
        saved_results["band_powers"]
    ) == 2

    assert (
        saved_results["noise_floor"]
        is not None
    )

    assert np.isfinite(
        saved_results[
            "noise_floor"
        ]["noise_floor_db"]
    )

    spectral_features = (
        saved_results["spectral_features"]
    )

    assert (
        spectral_features["flatness"]
        is not None
    )

    assert (
        spectral_features["centroid_hz"]
        is not None
    )

    assert (
        spectral_features["spread_hz"]
        is not None
    )

    assert (
        spectral_features[
            "psd_percentiles_db"
        ]
        is not None
    )

    assert (
        saved_results["narrowband_lines"]
        is not None
    )

    assert len(
        saved_results["narrowband_lines"]
    ) >= 1

    assert (
        saved_results["characterization"]
        is not None
    )

    assert (
        saved_results[
            "characterization"
        ]["label"]
        == "mixed"
    )


def test_saved_constant_signal_uses_null_for_v02_features(
    tmp_path,
):
    samples = np.full(
        2000,
        3.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=1000.0,
    )

    config = SingleChannelAnalysisConfig(
        spectrum_window=None,
        psd_nperseg=1000,
        psd_noverlap=500,
        psd_window="hann",
        spectrogram_nperseg=200,
        spectrogram_noverlap=100,
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

    assert saved_results["noise_floor"] is None

    assert (
        saved_results[
            "spectral_features"
        ]["flatness"]
        is None
    )

    assert (
        saved_results["narrowband_lines"]
        is None
    )

    assert (
        saved_results["harmonic_analysis"]
        is None
    )

    assert (
        saved_results["characterization"]
        is None
    )


def test_saved_json_contains_v03_event_analysis(
    tmp_path,
):
    sample_rate_hz = 1000.0
    duration_seconds = 4.0

    time = np.arange(
        int(sample_rate_hz * duration_seconds)
    ) / sample_rate_hz

    samples = (
        np.sin(
            2.0 * np.pi * 100.0 * time
        )
        + 0.02 * np.sin(
            2.0 * np.pi * 17.0 * time
        )
    )

    event_mask = (
        (time >= 2.0)
        & (time < 2.5)
    )

    samples[event_mask] *= 4.0

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        channel_name="v03_event_test",
        units="V",
    )

    config = SingleChannelAnalysisConfig(
        spectrum_window="hann",
        psd_nperseg=1000,
        psd_noverlap=500,
        psd_window="hann",
        spectrogram_nperseg=200,
        spectrogram_noverlap=100,
        spectrogram_window="hann",
        peak_min_prominence=0.1,
        event_analysis_config=EventAnalysisConfig(
            frame_length_samples=100,
            hop_length_samples=50,
            start_threshold=6.0,
            end_threshold=3.0,
            minimum_event_duration_s=0.2,
            merge_gap_s=0.1,
        ),
    )

    result = analyze_single_channel(
        record,
        config,
    )

    config_path, results_path = save_analysis_json(
        record,
        result,
        tmp_path,
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_config = json.load(file)

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_results = json.load(file)

    assert saved_config["event_analysis"] is not None

    assert (
        saved_config[
            "event_analysis"
        ]["frame_length_samples"]
        == 100
    )

    assert (
        saved_config[
            "event_analysis"
        ]["hop_length_samples"]
        == 50
    )

    assert (
        saved_config[
            "event_analysis"
        ]["start_threshold"]
        == 6.0
    )

    assert (
        saved_config[
            "event_analysis"
        ]["end_threshold"]
        == 3.0
    )

    assert (
        saved_config[
            "event_analysis"
        ]["minimum_event_duration_s"]
        == 0.2
    )

    assert (
        saved_config[
            "event_analysis"
        ]["merge_gap_s"]
        == 0.1
    )

    event_analysis = saved_results[
        "event_analysis"
    ]

    assert event_analysis is not None

    assert (
        event_analysis["feature"]
        == "short_time_rms"
    )

    assert (
        event_analysis[
            "baseline"
        ]["status"]
        == "valid"
    )

    assert (
        event_analysis[
            "baseline"
        ]["method"]
        == "median_mad"
    )

    assert (
        event_analysis[
            "frame"
        ]["frame_length_samples"]
        == 100
    )

    assert (
        event_analysis[
            "frame"
        ]["hop_length_samples"]
        == 50
    )

    assert (
        event_analysis[
            "number_of_events"
        ]
        == 1
    )

    assert len(
        event_analysis["events"]
    ) == 1

    event = event_analysis["events"][0]

    assert event["start_time_s"] <= 2.1
    assert event["end_time_s"] >= 2.4

    assert event["peak_score"] > 6.0

    assert (
        event["support_duration_s"]
        >= 0.2
    )

    assert (
        event["rms_ratio_to_baseline"]
        > 3.0
    )


def test_saved_json_uses_null_when_v03_not_enabled(
    tmp_path,
):
    _, samples = generate_two_tone(
        frequency_1_hz=100.0,
        amplitude_1=1.0,
        frequency_2_hz=200.0,
        amplitude_2=0.5,
        sample_rate_hz=1000.0,
        duration_seconds=2.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=1000.0,
    )

    config = SingleChannelAnalysisConfig(
        spectrum_window="hann",
        psd_nperseg=1000,
        psd_noverlap=500,
        psd_window="hann",
        spectrogram_nperseg=200,
        spectrogram_noverlap=100,
        spectrogram_window="hann",
        peak_min_prominence=0.1,
    )

    result = analyze_single_channel(
        record,
        config,
    )

    config_path, results_path = save_analysis_json(
        record,
        result,
        tmp_path,
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_config = json.load(file)

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_results = json.load(file)

    assert saved_config["event_analysis"] is None
    assert saved_results["event_analysis"] is None