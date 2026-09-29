import matplotlib.pyplot as plt
import numpy as np

from noise_analyzer.io.validation import (
    validate_finite_samples,
    validate_nonempty_signal,
)
from noise_analyzer.models.signal import SignalRecord


def plot_waveform(record: SignalRecord):
    """Create a time-domain waveform plot for a signal."""

    validate_nonempty_signal(record)
    validate_finite_samples(record)

    time = np.arange(record.number_of_samples) / record.sample_rate_hz

    fig, ax = plt.subplots()

    ax.plot(time, record.samples)

    ax.set_xlabel("Time [s]")

    if record.units == "unknown":
        ax.set_ylabel("Amplitude [arbitrary units]")
    else:
        ax.set_ylabel(f"Amplitude [{record.units}]")

    ax.set_title(record.channel_name)

    return fig, ax