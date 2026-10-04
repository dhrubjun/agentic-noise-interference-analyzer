import numpy as np
import pytest

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.pipeline.single_channel import (
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)
from noise_analyzer.synthetic.signals import (
    generate_constant,
    generate_dc_offset_sine,
    generate_intermittent_interference,
    generate_noisy_sine,
    generate_sine,
    generate_two_tone,
)

def make_config():
    return SingleChannelAnalysisConfig(
        spectrum_window="hann",
        psd_nperseg=2000,
        psd_noverlap=1000,
        psd_window="hann",
        spectrogram_nperseg=1000,
        spectrogram_noverlap=500,
        spectrogram_window="hann",
        peak_min_prominence=0.05,
    )

def test_matrix_pure_sine():
    _, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=2.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    assert np.isclose(
        result.statistics.rms,
        2.0 / np.sqrt(2.0),
        rtol=1e-3,
    )

    peak_frequencies = np.array(
        [peak.frequency_hz for peak in result.peaks.peaks]
    )

    assert np.any(
        np.isclose(
            peak_frequencies,
            1000.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

def test_matrix_constant_signal():
    _, samples = generate_constant(
        value=3.0,
        sample_rate_hz=10000.0,
        duration_seconds=1.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    assert result.is_constant is True
    assert np.isclose(result.statistics.mean, 3.0)
    assert np.isclose(result.statistics.rms, 3.0)
    assert np.isclose(result.statistics.standard_deviation, 0.0)

def test_matrix_two_tone():
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
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    peak_frequencies = np.array(
        [peak.frequency_hz for peak in result.peaks.peaks]
    )

    assert np.any(
        np.isclose(
            peak_frequencies,
            1000.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

    assert np.any(
        np.isclose(
            peak_frequencies,
            1800.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

def test_matrix_dc_offset_sine():
    _, samples = generate_dc_offset_sine(
        dc_offset=2.0,
        frequency_hz=1000.0,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    assert np.isclose(
        result.statistics.mean,
        2.0,
        atol=1e-12,
    )

def test_matrix_low_noise_signal():
    _, samples = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
        seed=42,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    peak_frequencies = np.array(
        [peak.frequency_hz for peak in result.peaks.peaks]
    )

    assert np.any(
        np.isclose(
            peak_frequencies,
            1000.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

    assert result.statistics.standard_deviation > 0

def test_matrix_higher_noise_signal():
    _, samples = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
        seed=42,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    peak_frequencies = np.array(
        [peak.frequency_hz for peak in result.peaks.peaks]
    )

    assert np.any(
        np.isclose(
            peak_frequencies,
            1000.0,
            atol=result.spectrum.frequency_resolution_hz,
        )
    )

def test_matrix_intermittent_interference():
    _, samples = generate_intermittent_interference(
        base_frequency_hz=1000.0,
        base_amplitude=1.0,
        interference_frequency_hz=1800.0,
        interference_amplitude=0.5,
        interference_start_seconds=4.0,
        interference_end_seconds=6.0,
        sample_rate_hz=10000.0,
        duration_seconds=10.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    index_1800 = np.argmin(
        np.abs(
            result.spectrogram.frequencies_hz - 1800.0
        )
    )

    power_1800 = (
        result.spectrogram.power_spectral_density[
            index_1800, :
        ]
    )

    before_mask = (
        result.spectrogram.times_seconds < 3.5
    )

    during_mask = (
        (result.spectrogram.times_seconds >= 4.2)
        & (result.spectrogram.times_seconds <= 5.8)
    )

    after_mask = (
        result.spectrogram.times_seconds > 6.5
    )

    before_power = np.mean(
        power_1800[before_mask]
    )

    during_power = np.mean(
        power_1800[during_mask]
    )

    after_power = np.mean(
        power_1800[after_mask]
    )

    assert during_power > before_power
    assert during_power > after_power

def test_matrix_non_bin_centered_sine():
    _, samples = generate_sine(
        frequency_hz=1000.37,
        amplitude=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    result = analyze_single_channel(
        record,
        make_config(),
    )

    dominant_index = np.argmax(
        result.spectrum.amplitudes
    )

    dominant_frequency = (
        result.spectrum.frequencies_hz[
            dominant_index
        ]
    )

    assert np.isclose(
        dominant_frequency,
        1000.37,
        atol=result.spectrum.frequency_resolution_hz,
    )

def test_matrix_rejects_nan_samples():
    samples = np.array([0.0, 1.0, np.nan, 2.0])

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=1000.0,
    )

    with pytest.raises(ValueError):
        analyze_single_channel(
            record,
            make_config(),
        )

def test_matrix_rejects_inf_samples():
    samples = np.array([0.0, 1.0, np.inf, 2.0])

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=1000.0,
    )

    with pytest.raises(ValueError):
        analyze_single_channel(
            record,
            make_config(),
        )

def test_matrix_rejects_empty_signal():
    record = SignalRecord(
        samples=np.array([], dtype=float),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(ValueError):
        analyze_single_channel(
            record,
            make_config(),
        )

@pytest.mark.parametrize(
    "sample_rate_hz",
    [0.0, -1000.0],
)
def test_matrix_rejects_invalid_sample_rate(sample_rate_hz):
    with pytest.raises(ValueError):
        SignalRecord(
            samples=np.array([0.0, 1.0]),
            sample_rate_hz=sample_rate_hz,
        )

def test_matrix_rejects_too_short_signal():
    record = SignalRecord(
        samples=np.array([1.0]),
        sample_rate_hz=1000.0,
    )

    with pytest.raises(ValueError):
        analyze_single_channel(
            record,
            make_config(),
        )

def test_matrix_noise_generation_is_reproducible():
    _, samples_1 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
        seed=42,
    )

    _, samples_2 = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
        seed=42,
    )

    assert np.array_equal(
        samples_1,
        samples_2,
    )

def test_matrix_higher_noise_increases_rms_std_and_psd_floor():
    _, low_noise_samples = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.1,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
        seed=42,
    )

    _, high_noise_samples = generate_noisy_sine(
        frequency_hz=1000.0,
        amplitude=1.0,
        noise_std=0.5,
        sample_rate_hz=10000.0,
        duration_seconds=5.0,
        seed=42,
    )

    low_record = SignalRecord(
        samples=low_noise_samples,
        sample_rate_hz=10000.0,
    )

    high_record = SignalRecord(
        samples=high_noise_samples,
        sample_rate_hz=10000.0,
    )

    low_result = analyze_single_channel(
        low_record,
        make_config(),
    )

    high_result = analyze_single_channel(
        high_record,
        make_config(),
    )

    # Time-domain noise increase
    assert (
        high_result.statistics.rms
        > low_result.statistics.rms
    )

    assert (
        high_result.statistics.standard_deviation
        > low_result.statistics.standard_deviation
    )

    # Estimate PSD noise floor away from the 1 kHz tone.
    frequencies = low_result.psd.frequencies_hz

    noise_band_mask = (
        (frequencies >= 2000.0)
        & (frequencies <= 4000.0)
    )

    low_noise_floor = np.median(
        low_result.psd.psd[noise_band_mask]
    )

    high_noise_floor = np.median(
        high_result.psd.psd[noise_band_mask]
    )

    assert high_noise_floor > low_noise_floor

    tone_index_low = np.argmin(
        np.abs(
            low_result.psd.frequencies_hz - 1000.0
        )
    )

    tone_index_high = np.argmin(
        np.abs(
            high_result.psd.frequencies_hz - 1000.0
        )
    )

    low_tone_to_floor_ratio = (
        low_result.psd.psd[tone_index_low]
        / low_noise_floor
    )

    high_tone_to_floor_ratio = (
        high_result.psd.psd[tone_index_high]
        / high_noise_floor
    )

    assert (
        high_tone_to_floor_ratio
        < low_tone_to_floor_ratio
    )