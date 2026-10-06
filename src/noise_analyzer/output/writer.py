import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

        "band_power_ranges_hz": [
            list(frequency_range)
            for frequency_range
            in result.config.band_power_ranges_hz
        ],

        "noise_floor_lower_frequency_hz": (
            result.config.noise_floor_lower_frequency_hz
        ),
        "noise_floor_upper_frequency_hz": (
            result.config.noise_floor_upper_frequency_hz
        ),

        "narrowband_neighbourhood_width_hz": (
            result.config.narrowband_neighbourhood_width_hz
        ),
        "narrowband_excluded_peak_width_hz": (
            result.config.narrowband_excluded_peak_width_hz
        ),

        "harmonic_tolerance_hz": (
            result.config.harmonic_tolerance_hz
        ),
        "harmonic_max_order": (
            result.config.harmonic_max_order
        ),
        "harmonic_minimum_matches": (
            result.config.harmonic_minimum_matches
        ),

        "characterization_broadband_flatness_threshold": (
            result.config
            .characterization_broadband_flatness_threshold
        ),
        "characterization_tonal_flatness_threshold": (
            result.config
            .characterization_tonal_flatness_threshold
        ),
        "characterization_strong_line_threshold_db": (
            result.config
            .characterization_strong_line_threshold_db
        ),
        "characterization_harmonic_minimum_matches": (
            result.config
            .characterization_harmonic_minimum_matches
        ),

        "harmonic_relative_tolerance_fraction": (
            result.config.harmonic_relative_tolerance_fraction
        ),

        "harmonic_minimum_match_fraction": (
            result.config.harmonic_minimum_match_fraction
        ),

        "harmonic_minimum_consecutive_matches": (
            result.config.harmonic_minimum_consecutive_matches
        ),

        "occupied_bandwidth_power_fraction": (
            result.config.occupied_bandwidth_power_fraction
        ),
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

        "band_powers": [
            {
                "lower_frequency_hz": band.lower_frequency_hz,
                "upper_frequency_hz": band.upper_frequency_hz,
                "band_power": band.band_power,
                "fraction_of_total_power": (
                    band.fraction_of_total_power
                ),
            }
            for band in result.band_powers
        ],

        "noise_floor": (
            None
            if result.noise_floor is None
            else {
                "noise_floor_db": (
                    result.noise_floor.noise_floor_db
                ),
                "method": (
                    result.noise_floor.method
                ),
                "lower_frequency_hz": (
                    result.noise_floor.lower_frequency_hz
                ),
                "upper_frequency_hz": (
                    result.noise_floor.upper_frequency_hz
                ),
                "number_of_bins": (
                    result.noise_floor.number_of_bins
                ),
            }
        ),

        "spectral_features": {
            "flatness": (
                None
                if result.spectral_flatness is None
                else result.spectral_flatness.spectral_flatness
            ),

            "centroid_hz": (
                None
                if result.spectral_centroid is None
                else (
                    result.spectral_centroid
                    .spectral_centroid_hz
                )
            ),

            "spread_hz": (
                None
                if result.spectral_spread is None
                else (
                    result.spectral_spread
                    .spectral_spread_hz
                )
            ),

            "psd_percentiles_db": (
                None
                if result.psd_percentiles is None
                else {
                    "p10": (
                        result.psd_percentiles
                        .percentile_10_db
                    ),
                    "p25": (
                        result.psd_percentiles
                        .percentile_25_db
                    ),
                    "p50": (
                        result.psd_percentiles
                        .percentile_50_db
                    ),
                    "p75": (
                        result.psd_percentiles
                        .percentile_75_db
                    ),
                    "p90": (
                        result.psd_percentiles
                        .percentile_90_db
                    ),
                }
            ),
        },

        "narrowband_lines": (
            None
            if result.narrowband_lines is None
            else [
                {
                    "frequency_hz": line.frequency_hz,
                    "peak_level_db": line.peak_level_db,
                    "local_noise_floor_db": (
                        line.local_noise_floor_db
                    ),
                    "line_to_floor_db": (
                        line.line_to_floor_db
                    ),
                    "neighbourhood_lower_hz": (
                        line.neighbourhood_lower_hz
                    ),
                    "neighbourhood_upper_hz": (
                        line.neighbourhood_upper_hz
                    ),
                    "excluded_peak_width_hz": (
                        line.excluded_peak_width_hz
                    ),
                    "number_of_background_bins": (
                        line.number_of_background_bins
                    ),
                }
                for line in result.narrowband_lines.lines
            ]
        ),

        "harmonic_analysis": (
            None
            if result.harmonic_family is None
            else {
                "candidate_fundamentals_hz": list(
                    result.harmonic_family
                    .candidate_fundamentals_hz
                ),
                "best_family": (
                    None
                    if (
                        result.harmonic_family
                        .best_family is None
                    )
                    else {
                        "candidate_fundamental_hz": (
                            result.harmonic_family
                            .best_family
                            .candidate_fundamental_hz
                        ),
                        "tolerance_hz": (
                            result.harmonic_family
                            .best_family
                            .tolerance_hz
                        ),
                        "matches": [
                            {
                                "harmonic_order": (
                                    match.harmonic_order
                                ),
                                "expected_frequency_hz": (
                                    match.expected_frequency_hz
                                ),
                                "detected_frequency_hz": (
                                    match.detected_frequency_hz
                                ),
                                "frequency_error_hz": (
                                    match.frequency_error_hz
                                ),
                            }
                            for match
                            in (
                                result.harmonic_family
                                .best_family
                                .matches
                            )
                        ],
                    }
                ),
            }
        ),

        "characterization": (
            None
            if result.characterization is None
            else {
                "label": (
                    result.characterization.label
                ),
                "spectral_flatness": (
                    result.characterization
                    .spectral_flatness
                ),
                "strong_line_count": (
                    result.characterization
                    .strong_line_count
                ),
                "strongest_line_to_floor_db": (
                    result.characterization
                    .strongest_line_to_floor_db
                ),
                "harmonic_match_count": (
                    result.characterization
                    .harmonic_match_count
                ),
                "thresholds": {
                    "broadband_flatness": (
                        result.characterization
                        .broadband_flatness_threshold
                    ),
                    "tonal_flatness": (
                        result.characterization
                        .tonal_flatness_threshold
                    ),
                    "strong_line_db": (
                        result.characterization
                        .strong_line_threshold_db
                    ),
                    "harmonic_minimum_matches": (
                        result.characterization
                        .harmonic_minimum_matches
                    ),
                },
            }
        ),

        "occupied_bandwidth": (
            None
            if result.occupied_bandwidth is None
            else {
                "occupied_bandwidth_hz": (
                    result.occupied_bandwidth.occupied_bandwidth_hz
                ),
                "lower_edge_hz": (
                    result.occupied_bandwidth.lower_edge_hz
                ),
                "upper_edge_hz": (
                    result.occupied_bandwidth.upper_edge_hz
                ),
                "power_fraction": (
                    result.occupied_bandwidth.power_fraction
                ),
            }
        ),


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

def save_analysis_plots(
    record: SignalRecord,
    result: SingleChannelAnalysisResult,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Save waveform, spectrum, PSD, ASD, and spectrogram plots."""

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    waveform_path = output_path / "waveform.png"
    spectrum_path = output_path / "spectrum.png"
    psd_path = output_path / "psd.png"
    asd_path = output_path / "asd.png"
    spectrogram_path = output_path / "spectrogram.png"

    # Waveform
    time_seconds = (
        np.arange(record.number_of_samples)
        / record.sample_rate_hz
    )

    fig, ax = plt.subplots()

    ax.plot(
        time_seconds,
        record.samples,
    )

    ax.set_xlabel("Time [s]")

    if record.units == "unknown":
        ax.set_ylabel("Amplitude [arbitrary units]")
    else:
        ax.set_ylabel(
            f"Amplitude [{record.units}]"
        )

    ax.set_title(record.channel_name)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        waveform_path,
        dpi=150,
    )
    plt.close(fig)

    # Amplitude spectrum
    fig, ax = plt.subplots()

    ax.plot(
        result.spectrum.frequencies_hz,
        result.spectrum.amplitudes,
    )

    ax.set_xlabel("Frequency [Hz]")

    if record.units == "unknown":
        ax.set_ylabel("Amplitude [arbitrary units]")
    else:
        ax.set_ylabel(
            f"Amplitude [{record.units}]"
        )

    ax.set_title(
        f"{record.channel_name} - Amplitude Spectrum"
    )

    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        spectrum_path,
        dpi=150,
    )
    plt.close(fig)

    # PSD
    fig, ax = plt.subplots()

    ax.semilogy(
        result.psd.frequencies_hz,
        result.psd.psd,
    )

    ax.set_xlabel("Frequency [Hz]")

    if record.units == "unknown":
        ax.set_ylabel("PSD [units²/Hz]")
    else:
        ax.set_ylabel(
            f"PSD [{record.units}²/Hz]"
        )

    ax.set_title(
        f"{record.channel_name} - Power Spectral Density"
    )

    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        psd_path,
        dpi=150,
    )
    plt.close(fig)

    # ASD
    fig, ax = plt.subplots()

    ax.semilogy(
        result.asd.frequencies_hz,
        result.asd.asd,
    )

    ax.set_xlabel("Frequency [Hz]")

    if record.units == "unknown":
        ax.set_ylabel("ASD [units/√Hz]")
    else:
        ax.set_ylabel(
            f"ASD [{record.units}/√Hz]"
        )

    ax.set_title(
        f"{record.channel_name} - Amplitude Spectral Density"
    )

    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        asd_path,
        dpi=150,
    )
    plt.close(fig)

    # Spectrogram
    spectrogram_db = 10.0 * np.log10(
        np.maximum(
            result.spectrogram.power_spectral_density,
            1e-20,
        )
    )

    fig, ax = plt.subplots()

    image = ax.pcolormesh(
        result.spectrogram.times_seconds,
        result.spectrogram.frequencies_hz,
        spectrogram_db,
        shading="auto",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    ax.set_title(
        f"{record.channel_name} - Spectrogram"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    if record.units == "unknown":
        colorbar.set_label(
            "PSD [dB re 1 arbitrary-unit²/Hz]"
        )
    else:
        colorbar.set_label(
            f"PSD [dB re 1 {record.units}²/Hz]"
        )

    fig.tight_layout()
    fig.savefig(
        spectrogram_path,
        dpi=150,
    )
    plt.close(fig)

    return (
        waveform_path,
        spectrum_path,
        psd_path,
        asd_path,
        spectrogram_path,
    )