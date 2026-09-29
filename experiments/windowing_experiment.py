import numpy as np
import matplotlib.pyplot as plt

from noise_analyzer.synthetic.signals import generate_sine


# -------------------------------------------------
# Reference signal settings
# -------------------------------------------------

sample_rate_hz = 10000.0
duration_seconds = 5.0
frequency_hz = 1000.37
amplitude = 1.0


# -------------------------------------------------
# Generate non-bin-centered sine wave
# -------------------------------------------------

time, samples = generate_sine(
    frequency_hz=frequency_hz,
    amplitude=amplitude,
    sample_rate_hz=sample_rate_hz,
    duration_seconds=duration_seconds,
)

number_of_samples = len(samples)

frequency_resolution_hz = sample_rate_hz / number_of_samples

frequencies_hz = np.fft.rfftfreq(
    number_of_samples,
    d=1.0 / sample_rate_hz,
)


# -------------------------------------------------
# Case 1: No window
# -------------------------------------------------

fft_no_window = np.fft.rfft(samples)

amplitude_no_window = (
    np.abs(fft_no_window)
    / number_of_samples
)

# Convert to one-sided amplitude spectrum.
if number_of_samples % 2 == 0:
    amplitude_no_window[1:-1] *= 2.0
else:
    amplitude_no_window[1:] *= 2.0


# -------------------------------------------------
# Case 2: Hann window
# -------------------------------------------------

hann_window = np.hanning(number_of_samples)

windowed_samples = samples * hann_window

fft_hann = np.fft.rfft(windowed_samples)

# The Hann window reduces the amplitude of the signal.
# Coherent-gain correction compensates for this effect.
coherent_gain = np.mean(hann_window)

amplitude_hann = (
    np.abs(fft_hann)
    / number_of_samples
    / coherent_gain
)

# Convert to one-sided amplitude spectrum.
if number_of_samples % 2 == 0:
    amplitude_hann[1:-1] *= 2.0
else:
    amplitude_hann[1:] *= 2.0


# -------------------------------------------------
# Convert amplitudes to dB
# -------------------------------------------------

epsilon = 1e-12

amplitude_no_window_db = 20.0 * np.log10(
    np.maximum(amplitude_no_window, epsilon)
)

amplitude_hann_db = 20.0 * np.log10(
    np.maximum(amplitude_hann, epsilon)
)


# -------------------------------------------------
# Inspect dominant FFT bins
# -------------------------------------------------

peak_index_no_window = np.argmax(amplitude_no_window)
peak_index_hann = np.argmax(amplitude_hann)

print("Signal frequency:", frequency_hz, "Hz")
print("Frequency resolution:", frequency_resolution_hz, "Hz")

print()

print("No window:")
print(
    "Peak frequency:",
    frequencies_hz[peak_index_no_window],
    "Hz",
)
print(
    "Peak amplitude:",
    amplitude_no_window[peak_index_no_window],
)

print()

print("Hann window:")
print(
    "Peak frequency:",
    frequencies_hz[peak_index_hann],
    "Hz",
)
print(
    "Peak amplitude:",
    amplitude_hann[peak_index_hann],
)


# -------------------------------------------------
# Plot 1: Linear amplitude spectrum
# -------------------------------------------------

fig1, ax1 = plt.subplots()

ax1.plot(
    frequencies_hz,
    amplitude_no_window,
    label="No window",
)

ax1.plot(
    frequencies_hz,
    amplitude_hann,
    label="Hann window",
)

ax1.set_xlim(995, 1006)

ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Amplitude")
ax1.set_title(
    "Spectral Leakage: No Window vs Hann Window"
)

ax1.legend()
ax1.grid(True)


# -------------------------------------------------
# Plot 2: dB amplitude spectrum
# -------------------------------------------------

fig2, ax2 = plt.subplots()

ax2.plot(
    frequencies_hz,
    amplitude_no_window_db,
    label="No window",
)

ax2.plot(
    frequencies_hz,
    amplitude_hann_db,
    label="Hann window",
)

ax2.set_xlim(990, 1010)
ax2.set_ylim(-100, 5)

ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Amplitude [dB]")
ax2.set_title("Spectral Leakage in dB")

ax2.legend()
ax2.grid(True)


# -------------------------------------------------
# Save both figures
# -------------------------------------------------

from pathlib import Path

output_dir = Path(__file__).resolve().parent

linear_path = output_dir / "windowing_linear.png"
db_path = output_dir / "windowing_db.png"

fig1.savefig(
    linear_path,
    dpi=150,
    bbox_inches="tight",
)

fig2.savefig(
    db_path,
    dpi=150,
    bbox_inches="tight",
)

print()
print("Saved:")
print(linear_path)
print(db_path)

plt.close(fig1)
plt.close(fig2)