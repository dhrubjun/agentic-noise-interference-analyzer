from dataclasses import dataclass

import numpy as np

from noise_analyzer.io.validation import (
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord

from noise_analyzer.dsp.band_power import calculate_band_power
from noise_analyzer.dsp.psd import calculate_welch_psd


@dataclass(frozen=True)
class FrameResult:
    """Fixed-length frames extracted from a sampled signal."""

    frames: np.ndarray
    times_seconds: np.ndarray
    frame_length_samples: int
    hop_length_samples: int
    frame_length_seconds: float
    hop_length_seconds: float

@dataclass(frozen=True)
class ShortTimeRMSResult:
    """Short-time RMS measurements for a framed signal."""

    times_seconds: np.ndarray
    rms: np.ndarray
    frame_length_samples: int
    hop_length_samples: int
    frame_length_seconds: float
    hop_length_seconds: float

@dataclass(frozen=True)
class ShortTimeBandPowerResult:
    """Time-varying power measured inside one frequency band."""

    times_seconds: np.ndarray
    band_power: np.ndarray
    lower_frequency_hz: float
    upper_frequency_hz: float
    frame_length_samples: int
    hop_length_samples: int
    frame_length_seconds: float
    hop_length_seconds: float


def frame_signal(
    record: SignalRecord,
    frame_length_samples: int,
    hop_length_samples: int,
) -> FrameResult:
    """Divide a signal into fixed-length overlapping frames."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    if frame_length_samples < 1:
        raise ValueError(
            "frame_length_samples must be at least 1."
        )

    if hop_length_samples < 1:
        raise ValueError(
            "hop_length_samples must be at least 1."
        )

    if frame_length_samples > record.number_of_samples:
        raise ValueError(
            "frame_length_samples must not exceed the number of signal samples."
        )

    number_of_frames = (
        1
        + (
            record.number_of_samples
            - frame_length_samples
        )
        // hop_length_samples
    )

    frames = np.empty(
        (number_of_frames, frame_length_samples),
        dtype=float,
    )

    times_seconds = np.empty(
        number_of_frames,
        dtype=float,
    )

    for frame_index in range(number_of_frames):
        start_sample = frame_index * hop_length_samples
        end_sample = start_sample + frame_length_samples

        frames[frame_index] = record.samples[
            start_sample:end_sample
        ]

        centre_sample = (
            start_sample + frame_length_samples / 2.0
        )

        times_seconds[frame_index] = (
            centre_sample / record.sample_rate_hz
        )

    return FrameResult(
        frames=frames,
        times_seconds=times_seconds,
        frame_length_samples=frame_length_samples,
        hop_length_samples=hop_length_samples,
        frame_length_seconds=(
            frame_length_samples / record.sample_rate_hz
        ),
        hop_length_seconds=(
            hop_length_samples / record.sample_rate_hz
        ),
    )

def calculate_short_time_rms(
    record: SignalRecord,
    frame_length_samples: int,
    hop_length_samples: int,
) -> ShortTimeRMSResult:
    """Calculate RMS independently for each signal frame."""

    framed = frame_signal(
        record,
        frame_length_samples=frame_length_samples,
        hop_length_samples=hop_length_samples,
    )

    rms = np.sqrt(
        np.mean(
            framed.frames**2,
            axis=1,
        )
    )

    return ShortTimeRMSResult(
        times_seconds=framed.times_seconds,
        rms=rms,
        frame_length_samples=framed.frame_length_samples,
        hop_length_samples=framed.hop_length_samples,
        frame_length_seconds=framed.frame_length_seconds,
        hop_length_seconds=framed.hop_length_seconds,
    )

def calculate_short_time_band_power(
    record: SignalRecord,
    lower_frequency_hz: float,
    upper_frequency_hz: float,
    frame_length_samples: int,
    hop_length_samples: int,
    window: str = "hann",
) -> ShortTimeBandPowerResult:
    """Calculate band power independently for each signal frame."""

    framed = frame_signal(
        record,
        frame_length_samples=frame_length_samples,
        hop_length_samples=hop_length_samples,
    )

    band_powers = np.empty(
        framed.frames.shape[0],
        dtype=float,
    )

    for frame_index, frame_samples in enumerate(
        framed.frames
    ):
        frame_record = SignalRecord(
            samples=frame_samples,
            sample_rate_hz=record.sample_rate_hz,
            channel_name=record.channel_name,
            units=record.units,
        )

        psd_result = calculate_welch_psd(
            frame_record,
            nperseg=frame_length_samples,
            noverlap=0,
            window=window,
        )

        band_result = calculate_band_power(
            psd_result,
            lower_frequency_hz=lower_frequency_hz,
            upper_frequency_hz=upper_frequency_hz,
        )

        band_powers[frame_index] = (
            band_result.band_power
        )

    return ShortTimeBandPowerResult(
        times_seconds=framed.times_seconds,
        band_power=band_powers,
        lower_frequency_hz=lower_frequency_hz,
        upper_frequency_hz=upper_frequency_hz,
        frame_length_samples=frame_length_samples,
        hop_length_samples=hop_length_samples,
        frame_length_seconds=framed.frame_length_seconds,
        hop_length_seconds=framed.hop_length_seconds,
    )