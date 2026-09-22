import numpy as np

from noise_analyzer.models.signal import SignalRecord


def validate_finite_samples(record: SignalRecord) -> None:
    """Reject signals containing NaN or infinite sample values."""

    if not np.all(np.isfinite(record.samples)):
        raise ValueError(
            "Signal contains non-finite sample values (NaN or Inf)."
        )