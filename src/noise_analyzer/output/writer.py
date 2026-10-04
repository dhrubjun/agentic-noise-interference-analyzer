import json
from pathlib import Path

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.pipeline.single_channel import (
    SingleChannelAnalysisResult,
)


def save_analysis_json(
    record: SignalRecord,
    result: SingleChannelAnalysisResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save analysis configuration and summary results as JSON."""

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = output_path / "analysis_config.json"
    results_path = output_path / "results.json"

    config_data = {
        "spectrum_window": result.config.spectrum_window,
        "psd_nperseg": result.config.psd_nperseg,
        "psd_noverlap": result.config.psd_noverlap,
        "psd_window": result.config.psd_window,
        "spectrogram_nperseg": result.config.spectrogram_nperseg,
        "spectrogram_noverlap": result.config.spectrogram_noverlap,
        "spectrogram_window": result.config.spectrogram_window,
        "peak_min_prominence": result.config.peak_min_prominence,
        "peak_min_distance_hz": result.config.peak_min_distance_hz,
    }

    results_data = {
        "signal": {
            "channel_name": record.channel_name,
            "units": record.units,
            "sample_rate_hz": result.sample_rate_hz,
            "number_of_samples": result.number_of_samples,
            "duration_seconds": result.duration_seconds,
            "nyquist_frequency_hz": result.nyquist_frequency_hz,
            "is_constant": result.is_constant,
        },
        "statistics": {
            "mean": result.statistics.mean,
            "median": result.statistics.median,
            "rms": result.statistics.rms,
            "standard_deviation": result.statistics.standard_deviation,
            "variance": result.statistics.variance,
            "minimum": result.statistics.minimum,
            "maximum": result.statistics.maximum,
            "peak_to_peak": result.statistics.peak_to_peak,
        },
        "psd": {
            "frequency_resolution_hz": (
                result.psd.frequency_resolution_hz
            ),
            "integrated_power": result.psd.integrated_power,
        },
        "asd": {
            "frequency_resolution_hz": (
                result.asd.frequency_resolution_hz
            ),
        },
        "spectrogram": {
            "frequency_resolution_hz": (
                result.spectrogram.frequency_resolution_hz
            ),
            "time_step_seconds": (
                result.spectrogram.time_step_seconds
            ),
        },
        "peaks": [
            {
                "frequency_hz": peak.frequency_hz,
                "amplitude": peak.amplitude,
                "prominence": peak.prominence,
            }
            for peak in result.peaks.peaks
        ],
    }

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_data,
            file,
            indent=2,
        )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results_data,
            file,
            indent=2,
        )

    return config_path, results_path