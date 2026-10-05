import numpy as np

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.pipeline.single_channel import (
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)
from noise_analyzer.synthetic.signals import generate_two_tone
from noise_analyzer.synthetic.signals import (
    generate_constant,
    generate_two_tone,
)


def test_single_channel_pipeline_with_two_tone_signal():
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

    assert result.number_of_samples == 50000
    assert np.isclose(result.sample_rate_hz, 10000.0)
    assert np.isclose(result.duration_seconds, 5.0)
    assert np.isclose(result.nyquist_frequency_hz, 5000.0)

    assert result.is_constant is False

    detected_frequencies = np.array(
        [
            peak.frequency_hz
            for peak in result.peaks.peaks
        ]
    )

    assert np.any(
        np.isclose(
            detected_frequencies,
            1000.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

    assert np.any(
        np.isclose(
            detected_frequencies,
            1800.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

    assert np.allclose(
        result.asd.asd**2,
        result.psd.psd,
    )

def test_single_channel_pipeline_flags_constant_signal():
    _, samples = generate_constant(
        value=3.0,
        sample_rate_hz=1000.0,
        duration_seconds=2.0,
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
        record=record,
        config=config,
    )

    assert result.is_constant is True

    assert np.isclose(
        result.statistics.mean,
        3.0,
    )

    assert np.isclose(
        result.statistics.standard_deviation,
        0.0,
    )

def test_single_channel_pipeline_calculates_band_power():
    _, samples = generate_two_tone(
        frequency_1_hz=300.0,
        amplitude_1=1.0,
        frequency_2_hz=3000.0,
        amplitude_2=2.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
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
        band_power_ranges_hz=(
            (0.0, 1000.0),
            (1000.0, 5000.0),
        ),
    )

    result = analyze_single_channel(
        record,
        config,
    )

    assert len(result.band_powers) == 2

    assert np.isclose(
        result.band_powers[0].band_power,
        0.5,
        rtol=0.02,
    )

    assert np.isclose(
        result.band_powers[1].band_power,
        2.0,
        rtol=0.02,
    )