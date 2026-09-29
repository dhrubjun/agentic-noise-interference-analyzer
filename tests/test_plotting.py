import matplotlib.pyplot as plt
import numpy as np

from noise_analyzer.models.signal import SignalRecord
from noise_analyzer.plotting.plots import plot_waveform
from noise_analyzer.synthetic.signals import generate_sine


def test_plot_waveform_creates_expected_axes():
    _, samples = generate_sine(
        frequency_hz=1000.0,
        amplitude=2.0,
        sample_rate_hz=10000.0,
        duration_seconds=1.0,
    )

    record = SignalRecord(
        samples=samples,
        sample_rate_hz=10000.0,
        channel_name="reference_sine",
        units="V",
    )

    fig, ax = plot_waveform(record)

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Amplitude [V]"
    assert ax.get_title() == "reference_sine"

    plotted_line = ax.lines[0]

    assert len(plotted_line.get_xdata()) == 10000
    assert len(plotted_line.get_ydata()) == 10000

    assert np.allclose(
        plotted_line.get_ydata(),
        samples,
    )

    plt.close(fig)

def test_plot_waveform_uses_arbitrary_units_when_unknown():
    record = SignalRecord(
        samples=np.array([1.0, 2.0, 3.0]),
        sample_rate_hz=1000.0,
    )

    fig, ax = plot_waveform(record)

    assert ax.get_ylabel() == "Amplitude [arbitrary units]"

    plt.close(fig)