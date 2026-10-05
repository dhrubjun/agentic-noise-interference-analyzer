import numpy as np
import pytest

from noise_analyzer.dsp.band_power import (
    calculate_band_power,
)
from noise_analyzer.dsp.psd import (
    calculate_welch_psd,
)
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.synthetic.signals import (
    generate_two_tone,
)


def test_band_power_separates_two_tones():
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

    psd = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    low_band = calculate_band_power(
        psd,
        lower_frequency_hz=0.0,
        upper_frequency_hz=1000.0,
    )

    high_band = calculate_band_power(
        psd,
        lower_frequency_hz=1000.0,
        upper_frequency_hz=5000.0,
    )

    assert high_band.band_power > low_band.band_power


def test_band_power_matches_known_tone_powers():
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

    psd = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    low_band = calculate_band_power(
        psd,
        lower_frequency_hz=0.0,
        upper_frequency_hz=1000.0,
    )

    high_band = calculate_band_power(
        psd,
        lower_frequency_hz=1000.0,
        upper_frequency_hz=5000.0,
    )

    assert np.isclose(
        low_band.band_power,
        0.5,
        rtol=0.02,
    )

    assert np.isclose(
        high_band.band_power,
        2.0,
        rtol=0.02,
    )


def test_band_power_fractions_sum_to_one():
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

    psd = calculate_welch_psd(
        record,
        nperseg=10000,
        noverlap=5000,
    )

    low_band = calculate_band_power(
        psd,
        lower_frequency_hz=0.0,
        upper_frequency_hz=1000.0,
    )

    high_band = calculate_band_power(
        psd,
        lower_frequency_hz=1000.0,
        upper_frequency_hz=5000.0,
    )

    total_fraction = (
        low_band.fraction_of_total_power
        + high_band.fraction_of_total_power
    )

    assert np.isclose(
        total_fraction,
        1.0,
        rtol=0.01,
    )


def test_band_power_rejects_invalid_frequency_range():
    _, samples = generate_two_tone(
        frequency_1_hz=300.0,
        amplitude_1=1.0,
        frequency_2_hz=3000.0,
        amplitude_2=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=1.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    psd = calculate_welch_psd(
        record,
        nperseg=1000,
    )

    with pytest.raises(
        ValueError,
        match="greater than lower_frequency_hz",
    ):
        calculate_band_power(
            psd,
            lower_frequency_hz=1000.0,
            upper_frequency_hz=500.0,
        )


def test_band_power_rejects_frequency_above_nyquist():
    _, samples = generate_two_tone(
        frequency_1_hz=300.0,
        amplitude_1=1.0,
        frequency_2_hz=3000.0,
        amplitude_2=1.0,
        sample_rate_hz=10000.0,
        duration_seconds=1.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
    )

    psd = calculate_welch_psd(
        record,
        nperseg=1000,
    )

    with pytest.raises(
        ValueError,
        match="Nyquist",
    ):
        calculate_band_power(
            psd,
            lower_frequency_hz=1000.0,
            upper_frequency_hz=6000.0,
        )