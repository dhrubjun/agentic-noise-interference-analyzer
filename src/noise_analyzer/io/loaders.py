from pathlib import Path

import numpy as np
import pandas as pd

from noise_analyzer.models.signal import SignalRecord


def load_signal_csv(
    file_path: str | Path,
    sample_rate_hz: float | None = None,
    signal_column: str = "signal",
    time_column: str = "time",
    channel_name: str = "signal",
    units: str = "unknown",
) -> SignalRecord:
    """Load a single real-valued signal from a CSV file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = pd.read_csv(path)

    if signal_column not in dataframe.columns:
        raise ValueError(
            f"Signal column '{signal_column}' was not found in the CSV file."
        )

    try:
        samples = pd.to_numeric(
            dataframe[signal_column],
            errors="raise",
        ).to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Signal column '{signal_column}' must contain numeric values."
        ) from exc

    if time_column in dataframe.columns:
        sample_rate_hz = _infer_sample_rate_from_time(
            dataframe=dataframe,
            time_column=time_column,
        )

    elif sample_rate_hz is None:
        raise ValueError(
            "sample_rate_hz must be provided when no time column is present."
        )

    return SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        channel_name=channel_name,
        units=units,
        source_file=path,
    )


def _infer_sample_rate_from_time(
    dataframe: pd.DataFrame,
    time_column: str,
) -> float:
    """Infer sampling frequency from a uniformly spaced time column."""

    try:
        time = pd.to_numeric(
            dataframe[time_column],
            errors="raise",
        ).to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Time column '{time_column}' must contain numeric values."
        ) from exc

    if len(time) < 2:
        raise ValueError(
            "At least two timestamps are required to infer sample_rate_hz."
        )

    if not np.all(np.isfinite(time)):
        raise ValueError("Time column must contain only finite values.")

    time_steps = np.diff(time)

    if np.any(time_steps <= 0):
        raise ValueError(
            "Time values must be strictly increasing."
        )

    reference_step = np.median(time_steps)

    if not np.allclose(
        time_steps,
        reference_step,
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError(
            "Time column is not uniformly sampled."
        )

    return 1.0 / reference_step