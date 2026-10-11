import numpy as np

from noise_analyzer.dsp.anomaly import (
    calculate_robust_deviation_scores,
)
from noise_analyzer.dsp.baseline import (
    estimate_robust_baseline,
)
from noise_analyzer.dsp.event_comparison import (
    compare_event_to_baseline_spectrum,
)
from noise_analyzer.dsp.events import (
    detect_events_with_hysteresis,
    postprocess_events,
)
from noise_analyzer.dsp.short_time import (
    calculate_short_time_band_power,
)
from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.pipeline.single_channel import (
    EventAnalysisConfig,
    SingleChannelAnalysisConfig,
    analyze_single_channel,
)


def test_v03_synthetic_end_to_end_acceptance():
    sample_rate_hz = 1000.0
    duration_seconds = 6.0

    number_of_samples = int(
        sample_rate_hz * duration_seconds
    )

    time = (
        np.arange(number_of_samples)
        / sample_rate_hz
    )

    rng = np.random.default_rng(12345)

    # Normal baseline:
    # weak 300 Hz tone + weak broadband noise
    samples = (
        0.2 * np.sin(
            2.0 * np.pi * 300.0 * time
        )
        + rng.normal(
            loc=0.0,
            scale=0.02,
            size=number_of_samples,
        )
    )

    # Event 1:
    # broadband/amplitude increase
    amplitude_event_mask = (
        (time >= 1.5)
        & (time < 2.1)
    )

    samples[amplitude_event_mask] *= 4.0

    # Event 2:
    # new narrowband 100 Hz component
    frequency_event_mask = (
        (time >= 4.0)
        & (time < 4.6)
    )

    samples[frequency_event_mask] += (
        1.0
        * np.sin(
            2.0
            * np.pi
            * 100.0
            * time[frequency_event_mask]
        )
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        channel_name="v03_acceptance_signal",
        units="V",
    )

    # -------------------------------------------------
    # PART 1:
    # Main RMS-based V0.3 pipeline
    # -------------------------------------------------

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

    assert result.event_analysis is not None

    rms_events = result.event_analysis.events

    assert len(rms_events) >= 2

    # Find an RMS event overlapping the known
    # amplitude disturbance.
    amplitude_events = [
        event
        for event in rms_events
        if (
            event.start_time_s <= 2.0
            and event.end_time_s >= 1.6
        )
    ]

    assert len(amplitude_events) >= 1

    # Find an RMS event overlapping the known
    # frequency disturbance.
    frequency_rms_events = [
        event
        for event in rms_events
        if (
            event.start_time_s <= 4.5
            and event.end_time_s >= 4.1
        )
    ]

    assert len(frequency_rms_events) >= 1

    # Both disturbances must produce strong
    # deviations from the normal RMS baseline.
    measurements = (
        result.event_analysis.measurements
    )

    assert any(
        measurement.rms_ratio_to_baseline
        > 3.0
        for measurement in measurements
    )

    # -------------------------------------------------
    # PART 2:
    # Frequency-selective V0.3 event detection
    # -------------------------------------------------

    band_track = calculate_short_time_band_power(
        record,
        lower_frequency_hz=80.0,
        upper_frequency_hz=120.0,
        frame_length_samples=200,
        hop_length_samples=100,
    )

    band_baseline = estimate_robust_baseline(
        band_track.band_power
    )

    assert band_baseline.status == "valid"

    band_scores = (
        calculate_robust_deviation_scores(
            band_track.band_power,
            band_baseline,
        )
    )

    raw_band_events = (
        detect_events_with_hysteresis(
            times_seconds=(
                band_track.times_seconds
            ),
            scores=band_scores.scores,
            start_threshold=6.0,
            end_threshold=3.0,
        )
    )

    band_events = postprocess_events(
        raw_band_events,
        frame_length_seconds=(
            band_track.frame_length_seconds
        ),
        minimum_event_duration_s=0.2,
        merge_gap_s=0.1,
    )

    # The 100 Hz disturbance must produce
    # a frequency-selective event.
    frequency_band_events = [
        event
        for event in band_events
        if (
            event.start_time_s <= 4.5
            and event.end_time_s >= 4.1
        )
    ]

    assert len(frequency_band_events) >= 1

    strongest_frequency_event = max(
        frequency_band_events,
        key=lambda event: event.peak_score,
    )

    assert (
        strongest_frequency_event.peak_score
        > 6.0
    )

    # -------------------------------------------------
    # PART 3:
    # Spectral evidence for the frequency event
    # -------------------------------------------------

    baseline_record = SignalRecord(
        samples=samples[
            0:int(1.0 * sample_rate_hz)
        ],
        sample_rate_hz=sample_rate_hz,
        channel_name="baseline",
        units="V",
    )

    frequency_event_record = SignalRecord(
        samples=samples[
            int(4.0 * sample_rate_hz):
            int(4.6 * sample_rate_hz)
        ],
        sample_rate_hz=sample_rate_hz,
        channel_name="frequency_event",
        units="V",
    )

    comparison = (
        compare_event_to_baseline_spectrum(
            baseline_record=baseline_record,
            event_record=frequency_event_record,
            lower_frequency_hz=80.0,
            upper_frequency_hz=120.0,
            nperseg=200,
        )
    )

    assert (
        comparison.event_band_power
        > comparison.baseline_band_power
    )

    assert comparison.band_power_ratio > 20.0

    # The new 100 Hz tone shifts the spectrum
    # strongly toward lower frequency.
    assert comparison.centroid_shift_hz < 0.0