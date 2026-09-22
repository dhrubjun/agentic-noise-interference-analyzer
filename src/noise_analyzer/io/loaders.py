from pathlib import Path

import pandas as pd

from noise_analyzer.models.signal import SignalRecord


def load_signal_csv(
    file_path: str | Path,
    sample_rate_hz: float,
    signal_column: str = "signal",
    channel_name: str = "signal",
    units: str = "unknown",
) -> SignalRecord:
    """Load a single real-valued signal column from a CSV file."""

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

    return SignalRecord(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        channel_name=channel_name,
        units=units,
        source_file=path,
    )