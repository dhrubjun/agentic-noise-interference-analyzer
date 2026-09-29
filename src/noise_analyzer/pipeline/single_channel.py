from dataclasses import dataclass

from noise_analyzer.dsp.asd import (
    ASDResult,
    calculate_asd_from_psd,
)
from noise_analyzer.dsp.peaks import (
    PeakDetectionResult,
    detect_spectral_peaks,
)
from noise_analyzer.dsp.psd import (
    PSDResult,
    calculate_welch_psd,
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


def analyze_single_channel(
    record: SignalRecord,
    config: SingleChannelAnalysisConfig,
) -> SingleChannelAnalysisResult:
    """Run the validated V0.1 analysis modules on one signal."""

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

    asd = calculate_asd_from_psd(psd)

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
    )