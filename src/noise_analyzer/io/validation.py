import numpy as np

from noise_analyzer.models.signal import SignalRecord


def validate_finite_samples(record: SignalRecord) -> None:
    """Reject signals containing NaN or infinite sample values."""

    if not np.all(np.isfinite(record.samples)):
        raise ValueError(
            "Signal contains non-finite sample values (NaN or Inf)."
        )

def validate_nonempty_signal(record: SignalRecord) -> None:
    """Reject signals containing no samples."""

    if record.number_of_samples == 0:
        raise ValueError("Signal contains no samples.")

def is_constant_signal(record: SignalRecord) -> bool:
    """Return True when all signal samples have the same value."""

    if record.number_of_samples == 0:
        return False

    return bool(np.all(record.samples == record.samples[0]))