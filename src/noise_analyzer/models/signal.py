from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SignalRecord:
    """Container for one uniformly sampled signal and its metadata."""

    samples: np.ndarray
    sample_rate_hz: float
    channel_name: str = "signal"
    units: str = "unknown"
    source_file: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fundamental properties required by the signal record."""

        if not np.isfinite(self.sample_rate_hz):
            raise ValueError("sample_rate_hz must be finite.")

        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be greater than zero.")

    @property
    def number_of_samples(self) -> int:
        return int(self.samples.size)

    @property
    def duration_seconds(self) -> float:
        return self.number_of_samples / self.sample_rate_hz

    @property
    def nyquist_frequency_hz(self) -> float:
        return self.sample_rate_hz / 2.0

    @property
    def sampling_interval_seconds(self) -> float:
        return 1.0 / self.sample_rate_hz