from dataclasses import dataclass

from noise_analyzer.dsp.anomaly import (
    DeviationScoreResult,
    calculate_robust_deviation_scores,
)
from noise_analyzer.dsp.asd import (
    ASDResult,
    calculate_asd_from_psd,
)
from noise_analyzer.dsp.band_power import (
    BandPowerResult,
    calculate_band_power,
)
from noise_analyzer.dsp.baseline import (
    BaselineResult,
    estimate_robust_baseline,
)
from noise_analyzer.dsp.characterization import (
    CharacterizationResult,
    characterize_spectral_behavior,
)
from noise_analyzer.dsp.events import (
    EventInterval,
    EventMeasurement,
    detect_events_with_hysteresis,
    measure_rms_event,
    postprocess_events,
)
from noise_analyzer.dsp.harmonics import (
    AutomaticHarmonicSearchResult,
    find_best_harmonic_family,
)
from noise_analyzer.dsp.narrowband import (
    NarrowbandCharacterizationResult,
    characterize_detected_lines,
)
from noise_analyzer.dsp.noise_floor import (
    NoiseFloorResult,
    estimate_noise_floor,
)
from noise_analyzer.dsp.peaks import (
    PeakDetectionResult,
    detect_spectral_peaks,
)
from noise_analyzer.dsp.psd import (
    PSDResult,
    calculate_welch_psd,
)
from noise_analyzer.dsp.short_time import (
    ShortTimeRMSResult,
    calculate_short_time_rms,
)
from noise_analyzer.dsp.spectral_features import (
    OccupiedBandwidthResult,
    PSDPercentilesResult,
    SpectralCentroidResult,
    SpectralFlatnessResult,
    SpectralSpreadResult,
    calculate_occupied_bandwidth,
    calculate_psd_percentiles,
    calculate_spectral_centroid,
    calculate_spectral_flatness,
    calculate_spectral_spread,
)
from noise_analyzer.dsp.spectrogram import (
    SpectrogramResult,
    calculate_spectrogram,
)
from noise_analyzer.dsp.spectrum import (
    SpectrumResult,
    calculate_amplitude_spectrum,
)
from noise_analyzer.dsp.time_domain import (
    TimeDomainStatistics,
    calculate_time_domain_statistics,
)
from noise_analyzer.io.validation import (
    is_constant_signal,
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord


@dataclass(frozen=True)
class EventAnalysisConfig:
    """Configuration for V0.3 RMS-based event analysis."""

    frame_length_samples: int
    hop_length_samples: int

    start_threshold: float
    end_threshold: float

    minimum_event_duration_s: float = 0.0
    merge_gap_s: float = 0.0


@dataclass(frozen=True)
class EventAnalysisResult:
    """Structured V0.3 RMS event-analysis result."""

    feature_track: ShortTimeRMSResult
    baseline: BaselineResult
    deviation_scores: DeviationScoreResult

    raw_events: tuple[EventInterval, ...]
    events: tuple[EventInterval, ...]
    measurements: tuple[EventMeasurement, ...]


@dataclass(frozen=True)
class SingleChannelAnalysisConfig:
    """Configuration for the single-channel analysis pipeline."""

    spectrum_window: str | None

    psd_nperseg: int
    psd_noverlap: int | None
    psd_window: str

    spectrogram_nperseg: int
    spectrogram_noverlap: int | None
    spectrogram_window: str

    peak_min_prominence: float
    peak_min_distance_hz: float | None = None

    band_power_ranges_hz: tuple[tuple[float, float], ...] = ()

    noise_floor_lower_frequency_hz: float | None = None
    noise_floor_upper_frequency_hz: float | None = None

    narrowband_neighbourhood_width_hz: float = 100.0
    narrowband_excluded_peak_width_hz: float = 5.0

    harmonic_tolerance_hz: float = 1.0
    harmonic_max_order: int = 10
    harmonic_minimum_matches: int = 3

    characterization_broadband_flatness_threshold: float = 0.80
    characterization_tonal_flatness_threshold: float = 0.10
    characterization_strong_line_threshold_db: float = 10.0
    characterization_harmonic_minimum_matches: int = 3

    harmonic_relative_tolerance_fraction: float = 0.01
    harmonic_minimum_match_fraction: float = 0.5
    harmonic_minimum_consecutive_matches: int = 3

    occupied_bandwidth_power_fraction: float = 0.90

    event_analysis_config: EventAnalysisConfig | None = None


@dataclass(frozen=True)
class SingleChannelAnalysisResult:
    """Structured result from the single-channel analysis pipeline."""

    number_of_samples: int
    sample_rate_hz: float
    duration_seconds: float
    nyquist_frequency_hz: float
    is_constant: bool

    statistics: TimeDomainStatistics
    spectrum: SpectrumResult
    psd: PSDResult
    asd: ASDResult
    spectrogram: SpectrogramResult
    peaks: PeakDetectionResult

    config: SingleChannelAnalysisConfig

    band_powers: tuple[BandPowerResult, ...]

    noise_floor: NoiseFloorResult | None

    spectral_flatness: SpectralFlatnessResult | None
    spectral_centroid: SpectralCentroidResult | None
    spectral_spread: SpectralSpreadResult | None
    psd_percentiles: PSDPercentilesResult | None

    narrowband_lines: NarrowbandCharacterizationResult | None

    harmonic_family: AutomaticHarmonicSearchResult | None

    characterization: CharacterizationResult | None

    occupied_bandwidth: OccupiedBandwidthResult | None

    event_analysis: EventAnalysisResult | None


def analyze_single_channel(
    record: SignalRecord,
    config: SingleChannelAnalysisConfig,
) -> SingleChannelAnalysisResult:
    """Run the validated single-channel analysis pipeline."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    constant_signal = is_constant_signal(record)

    statistics = calculate_time_domain_statistics(record)

    spectrum = calculate_amplitude_spectrum(
        record,
        window=config.spectrum_window,
    )

    psd = calculate_welch_psd(
        record,
        nperseg=config.psd_nperseg,
        noverlap=config.psd_noverlap,
        window=config.psd_window,
    )

    if constant_signal or psd.integrated_power <= 0:
        spectral_flatness = None
        spectral_centroid = None
        spectral_spread = None
        psd_percentiles = None
    else:
        spectral_flatness = calculate_spectral_flatness(
            psd
        )

        spectral_centroid = calculate_spectral_centroid(
            psd
        )

        spectral_spread = calculate_spectral_spread(
            psd
        )

        psd_percentiles = calculate_psd_percentiles(
            psd
        )

    if constant_signal or psd.integrated_power <= 0:
        noise_floor = None
    else:
        noise_floor = estimate_noise_floor(
            psd,
            lower_frequency_hz=(
                config.noise_floor_lower_frequency_hz
            ),
            upper_frequency_hz=(
                config.noise_floor_upper_frequency_hz
            ),
        )

    band_powers = tuple(
        calculate_band_power(
            psd,
            lower_frequency_hz=lower_frequency_hz,
            upper_frequency_hz=upper_frequency_hz,
        )
        for lower_frequency_hz, upper_frequency_hz
        in config.band_power_ranges_hz
    )

    asd = calculate_asd_from_psd(
        psd
    )

    spectrogram = calculate_spectrogram(
        record,
        nperseg=config.spectrogram_nperseg,
        noverlap=config.spectrogram_noverlap,
        window=config.spectrogram_window,
    )

    peaks = detect_spectral_peaks(
        spectrum,
        min_prominence=config.peak_min_prominence,
        min_distance_hz=config.peak_min_distance_hz,
    )

    if constant_signal or len(peaks.peaks) < 2:
        harmonic_family = None
    else:
        harmonic_family = find_best_harmonic_family(
            peaks,
            tolerance_hz=config.harmonic_tolerance_hz,
            max_harmonic_order=config.harmonic_max_order,
            minimum_matches=config.harmonic_minimum_matches,
            frequency_resolution_hz=(
                spectrum.frequency_resolution_hz
            ),
            relative_tolerance_fraction=(
                config.harmonic_relative_tolerance_fraction
            ),
            minimum_match_fraction=(
                config.harmonic_minimum_match_fraction
            ),
            minimum_consecutive_matches=(
                config.harmonic_minimum_consecutive_matches
            ),
        )

    if constant_signal or psd.integrated_power <= 0:
        narrowband_lines = None
    else:
        narrowband_lines = characterize_detected_lines(
            psd,
            peaks,
            neighbourhood_width_hz=(
                config.narrowband_neighbourhood_width_hz
            ),
            excluded_peak_width_hz=(
                config.narrowband_excluded_peak_width_hz
            ),
        )

    if (
        constant_signal
        or spectral_flatness is None
        or narrowband_lines is None
    ):
        characterization = None
    else:
        characterization = characterize_spectral_behavior(
            spectral_flatness,
            narrowband_lines,
            harmonic_family,
            broadband_flatness_threshold=(
                config.characterization_broadband_flatness_threshold
            ),
            tonal_flatness_threshold=(
                config.characterization_tonal_flatness_threshold
            ),
            strong_line_threshold_db=(
                config.characterization_strong_line_threshold_db
            ),
            harmonic_minimum_matches=(
                config.characterization_harmonic_minimum_matches
            ),
        )

    if constant_signal or psd.integrated_power <= 0:
        occupied_bandwidth = None
    else:
        occupied_bandwidth = calculate_occupied_bandwidth(
            psd,
            power_fraction=(
                config.occupied_bandwidth_power_fraction
            ),
        )

    # V0.3 RMS-based event analysis
    if config.event_analysis_config is None:
        event_analysis = None

    elif constant_signal:
        event_analysis = None

    else:
        event_config = config.event_analysis_config

        feature_track = calculate_short_time_rms(
            record,
            frame_length_samples=(
                event_config.frame_length_samples
            ),
            hop_length_samples=(
                event_config.hop_length_samples
            ),
        )

        baseline = estimate_robust_baseline(
            feature_track.rms
        )

        if baseline.status != "valid":
            event_analysis = None

        else:
            deviation_scores = (
                calculate_robust_deviation_scores(
                    feature_track.rms,
                    baseline,
                )
            )

            raw_events_list = (
                detect_events_with_hysteresis(
                    times_seconds=(
                        feature_track.times_seconds
                    ),
                    scores=deviation_scores.scores,
                    start_threshold=(
                        event_config.start_threshold
                    ),
                    end_threshold=(
                        event_config.end_threshold
                    ),
                )
            )

            final_events_list = postprocess_events(
                raw_events_list,
                frame_length_seconds=(
                    feature_track.frame_length_seconds
                ),
                minimum_event_duration_s=(
                    event_config.minimum_event_duration_s
                ),
                merge_gap_s=(
                    event_config.merge_gap_s
                ),
            )

            measurements = tuple(
                measure_rms_event(
                    event=event,
                    rms_values=feature_track.rms,
                    baseline_rms=baseline.location,
                    frame_length_seconds=(
                        feature_track.frame_length_seconds
                    ),
                )
                for event in final_events_list
            )

            event_analysis = EventAnalysisResult(
                feature_track=feature_track,
                baseline=baseline,
                deviation_scores=deviation_scores,
                raw_events=tuple(raw_events_list),
                events=tuple(final_events_list),
                measurements=measurements,
            )

    return SingleChannelAnalysisResult(
        number_of_samples=record.number_of_samples,
        sample_rate_hz=record.sample_rate_hz,
        duration_seconds=record.duration_seconds,
        nyquist_frequency_hz=record.nyquist_frequency_hz,
        is_constant=constant_signal,
        statistics=statistics,
        spectrum=spectrum,
        psd=psd,
        asd=asd,
        spectrogram=spectrogram,
        peaks=peaks,
        config=config,
        band_powers=band_powers,
        noise_floor=noise_floor,
        spectral_flatness=spectral_flatness,
        spectral_centroid=spectral_centroid,
        spectral_spread=spectral_spread,
        psd_percentiles=psd_percentiles,
        narrowband_lines=narrowband_lines,
        harmonic_family=harmonic_family,
        characterization=characterization,
        occupied_bandwidth=occupied_bandwidth,
        event_analysis=event_analysis,
    )