# V0.1 DSP Validation Matrix



## Agentic Noise & Interference Analyzer



****Version:**** V0.1  

**Status:** V0.1 complete



**---**



## 1. Purpose



This document defines how the numerical DSP components of V0.1 are validated and records the completed V0.1 validation results.



The central rule is:



> ****A reasonable-looking plot is not sufficient evidence that a DSP implementation is correct.****



Validation should proceed through three levels:



```text

Analytical Validation

        ↓

Controlled Synthetic Validation

        ↓

Real-World Validation

```



### Level 1 — Analytical Validation



Use signals for which the expected mathematical result can be calculated.



### Level 2 — Synthetic System Validation



Use controlled signals containing known tones, noise, offsets, and time-varying events.



### Level 3 — Real-World Validation



Only after Levels 1 and 2 pass should the analyzer be evaluated using public or experimental measurement datasets.



**---**



# 2. Synthetic Reference Signals



## TEST-SIG-001 — Pure Sinusoid



Parameters:



```text

Sampling frequency: 10000 Hz

Duration:           5 s

Amplitude:          2

Frequency:          1000 Hz

Number of samples:  50000

```



Signal:



$$

x(t)=2\sin(2\pi1000t)

$$



Expected:



```text

Mean ≈ 0

RMS  = 2/√2

     ≈ 1.41421356



Dominant frequency ≈ 1000 Hz

```



Because the tone is bin-centered for this record length, it provides a clean initial FFT validation case.



**---**



## TEST-SIG-002 — Constant Signal



Signal:



$$

x[n]=3

$$



Expected:



```text

Mean             = 3

Median           = 3

RMS              = 3

Standard deviation = 0

Variance         = 0

Minimum          = 3

Maximum          = 3

Peak-to-peak     = 0

```



The analyzer should also identify the signal as constant.



**---**



## TEST-SIG-003 — Two-Tone Signal



Signal:



$$

x(t)=1.0\sin(2\pi1000t)+0.5\sin(2\pi1800t)

$$



Expected spectral components:



```text

1000 Hz

1800 Hz

```



The 1000 Hz component should have greater amplitude than the 1800 Hz component.



This signal validates frequency-axis generation and multiple-peak detection.



**---**



## TEST-SIG-004 — Sinusoid with Gaussian Noise



Parameters:



```text

Sampling frequency: 10000 Hz

Duration:           10 s

Tone frequency:     1000 Hz

Tone amplitude:     1

Noise sigma:        0.1

Random seed:        fixed

```



Expected:



- clear spectral component near 1000 Hz;

- broadband noise floor;

- PSD noise floor greater than the corresponding clean sinusoid.



An exact PSD-floor acceptance value should not be frozen until the estimator normalization and averaging behavior are derived and verified.



**---**



## TEST-SIG-005 — Increased Noise



Same signal as TEST-SIG-004, except:



```text

Noise sigma:

0.1 → 0.5

```



Expected:



- RMS increases;

- standard deviation increases;

- PSD noise floor increases;

- 1000 Hz tone becomes less prominent relative to the noise floor.



**---**



## TEST-SIG-006 — DC Offset and Sinusoid



Signal:



$$

x(t)=2+1.0\sin(2\pi1000t)

$$



Expected:



```text

Mean ≈ 2

Strong DC component

Spectral component at 1000 Hz

```



This validates separation between DC content and oscillatory components.



**---**



## TEST-SIG-007 — Intermittent Interference



Duration:



```text

10 s

```



Components:



```text

1000 Hz tone: entire record

1800 Hz tone: only from approximately 4 s to 6 s

```



Expected spectrogram:



```text

1000 Hz → visible throughout record



1800 Hz → visible approximately between

          4 s and 6 s

```



The acceptable event-boundary error should be related to the spectrogram's time resolution.



**---**



## TEST-SIG-008 — Non-Bin-Centered Sinusoid



Use a sinusoid such as:



```text

Frequency: 1000.37 Hz

```



with an FFT configuration where the tone does not fall exactly on an FFT bin.



Expected:



### Without Window



- visible spectral leakage;

- energy distributed across multiple bins.



### Hann Window



- reduced sidelobe leakage;

- broader main lobe.



This validates the effect of windowing and demonstrates why FFT and window parameters must be recorded.



**---**



## TEST-SIG-009 — NaN Input



Insert one or more NaN samples.



Expected:



```text

Validation failure

```



The analyzer must not silently interpolate or remove NaN values.



**---**



## TEST-SIG-010 — Infinite Input



Insert positive or negative infinity.



Expected:



```text

Validation failure

```



**---**



## TEST-SIG-011 — Empty Input



Provide an empty signal.



Expected:



```text

Input rejected

Clear validation reason returned

```



**---**



## TEST-SIG-012 — Invalid Sampling Frequency



Test:



```text

Fs = 0

Fs < 0

```



Expected:



```text

Input rejected

```



Sampling frequency must be finite and greater than zero.



**---**



## TEST-SIG-013 — Too-Short Signal



Example:



```text

N = 2

```



or another deliberately short input.



Expected:



- loader may accept valid numerical samples;

- analysis requiring more samples must report insufficient data clearly;

- raw low-level NumPy/SciPy exceptions should not be exposed as the normal user-facing result.



Minimum sample requirements should be defined per DSP operation during implementation.



**---**



## TEST-SIG-014 — Nonuniform Timestamps



Construct a signal with intentionally irregular time spacing.



Expected:



```text

Nonuniform sampling detected

```



The analyzer must not silently assume a constant sampling frequency or silently resample the signal.



The exact numerical uniformity tolerance will be determined and documented during implementation.



**---**



## TEST-SIG-015 — Reproducible Random Signal



Generate a noisy signal using a fixed random seed.



Initial seed:



```text

42

```



Expected:



Repeated generation with identical parameters and seed should produce identical samples.



This test supports reproducible numerical validation.



**---**



# 3. Numerical Validation Matrix



| ID | Measurement | Reference | Acceptance Principle |

| --- | --- | --- | --- |

| STAT-001 | Mean of zero-centered pure sinusoid | 0 | Numerical tolerance |

| STAT-002 | RMS of amplitude-2 sinusoid | 1.41421356... | Tight floating-point tolerance |

| STAT-003 | Constant-signal mean | 3 | Numerical equality |

| STAT-004 | Constant-signal standard deviation | 0 | Numerical equality |

| STAT-005 | Constant-signal variance | 0 | Numerical equality |

| STAT-006 | Constant-signal peak-to-peak | 0 | Numerical equality |

| FFT-001 | Pure-tone frequency | 1000 Hz | Error tied to FFT frequency resolution |

| FFT-002 | Two-tone frequencies | 1000 Hz, 1800 Hz | Both components detected |

| FFT-003 | DC + sinusoid | DC and 1000 Hz | Both components represented |

| FFT-004 | Non-bin-centered tone | Spectral leakage | Verified experimentally and quantitatively where practical |

| PSD-001 | Tone + noise | Tone above broadband floor | Pass |

| PSD-002 | Increased noise | Higher PSD floor | Must increase consistently |

| ASD-001 | ASD squared | PSD | Numerical equality within tolerance |

| SPEC-001 | Continuous 1000 Hz component | Entire record | Visible throughout |

| SPEC-002 | Intermittent 1800 Hz component | Approximately 4–6 s | Error tied to spectrogram time resolution |

| PEAK-001 | Two known tones | 1000 Hz, 1800 Hz | Both significant peaks detected |

| IO-001 | Valid CSV | Valid load | Pass |

| IO-002 | Empty input | Reject | Pass |

| IO-003 | NaN | Reject | Pass |

| IO-004 | Inf | Reject | Pass |

| IO-005 | Fs ≤ 0 | Reject | Pass |

| IO-006 | Nonuniform timestamps | Detect and reject/warn according to defined policy | Pass |



**---**



# 4. Tolerance Philosophy



Numerical tolerances should be derived from the properties of the algorithm whenever possible.



For example, FFT frequency resolution is:



$$

\Delta f=\frac{F_s}{N}

$$



Therefore, frequency-estimation acceptance should be related to \\(\Delta f\\) rather than using an arbitrary tolerance such as:



```text

±1 Hz

```



Similarly, spectrogram event timing should be evaluated relative to the time resolution produced by its segment and overlap parameters.



Floating-point equality tests should use appropriate numerical tolerances rather than direct equality where necessary.



**---**



# 5. PSD Power Consistency



Where applicable, the integrated PSD should be compared with the corresponding time-domain power.



Conceptually:



$$

P\_{\mathrm{spectral}}\approx\int PSD(f)\\,df

$$



For a suitable zero-mean signal, this should correspond to its time-domain mean-square power.



The exact acceptance tolerance will be established experimentally based on the Welch configuration, signal length, and numerical integration method.



This provides an important independent consistency check beyond inspecting the PSD plot.



**---**



# 6. ASD Consistency



ASD is calculated from PSD:



$$

ASD(f)=\sqrt{PSD(f)}

$$



Therefore:



$$

ASD(f)^2=PSD(f)

$$



The implementation should verify this numerically within floating-point tolerance.



**---**



# 7. Validation of Reused DSP Code



Some DSP implementations may be adapted from the earlier \`dsp-ai-agent\` project.



Existing code is not automatically considered validated for this project.



Every reused component must follow:



```text

Existing implementation

        ↓

Review algorithm and assumptions

        ↓

Adapt to new DSP architecture

        ↓

Run V0.1 reference tests

        ↓

Compare with expected results

        ↓

PASS

        ↓

Integrate

```



Possible outcomes are:



```text

Reuse

Refactor

Reimplement

Defer to later version

```



This ensures that previous work can be reused without weakening the validation standard of the new project.



**---**



# 8. Definition of Done for a DSP Module



A DSP module is considered complete only when:



1. its engineering purpose is understood;

2. its mathematical/DSP principle is documented;

3. inputs and outputs are defined;

4. a controlled reference signal exists;

5. the expected result is known;

6. the implementation is complete;

7. automated numerical tests pass;

8. plots are inspected where relevant;

9. important edge cases are tested;

10. important parameters are recorded;

11. documentation is updated.



The implementation itself is therefore only one part of completing a DSP module.



**---**



# 9. Validation Order



The recommended validation sequence is:



```text

Synthetic Signal Generator

        ↓

Input / Sampling Validation

        ↓

Time-Domain Statistics

        ↓

FFT

        ↓

Windowing

        ↓

PSD

        ↓

ASD

        ↓

Spectrogram

        ↓

Peak Detection

        ↓

Integrated Pipeline

        ↓

Public Real-World Dataset

```



A later module should not be used to compensate for an unvalidated earlier module.



**---**



# 10. Validation Record



As the project develops, each implemented module should have automated tests under:



```text

tests/

```



Engineering experiments that help establish or understand the implementation may be stored under:



```text

experiments/

```



Controlled synthetic data may be stored under:



```text

test_data/synthetic/

```



Generated analysis outputs should remain under:



```text

outputs/

```



The validation matrix should be updated as tests are implemented.



Eventually, each validation item should have an explicit status such as:



```text

NOT IMPLEMENTED

IN PROGRESS

PASS

FAIL

```



This document should therefore evolve from a validation plan into a record of the validated DSP foundation.



**---**



# 11. V0.1 Validation Status



## V0.1 Single-Channel Analyzer



| ID | Test Case | Expected Behavior | Status |

|---|---|---|---|

| TEST-SIG-001 | Pure sine, 1 kHz, A=2 | Correct RMS and dominant spectral peak | PASS |

| TEST-SIG-002 | Constant signal, value=3 | Mean=3, RMS=3, std=0, constant detected | PASS |

| TEST-SIG-003 | Two-tone, 1 kHz + 1.8 kHz | Both tones detected at correct amplitudes | PASS |

| TEST-SIG-004 | Sine + Gaussian noise, σ=0.1 | 1 kHz tone remains detectable and a broadband PSD noise floor is present | PASS |

| TEST-SIG-005 | Sine + Gaussian noise, σ=0.5 | RMS, standard deviation, and PSD noise floor increase; tone-to-floor ratio decreases | PASS |

| TEST-SIG-006 | DC offset + sine | Mean correctly reflects DC offset | PASS |

| TEST-SIG-007 | Intermittent 1.8 kHz interference | Spectrogram localizes interference to 4–6 s | PASS |

| TEST-SIG-008 | Non-bin-centered 1000.37 Hz sine | Dominant spectral bin lies within FFT resolution | PASS |

| TEST-SIG-009 | NaN samples | Analysis rejected | PASS |

| TEST-SIG-010 | Inf samples | Analysis rejected | PASS |

| TEST-SIG-011 | Empty signal | Analysis rejected | PASS |

| TEST-SIG-012 | Invalid sample rate | SignalRecord rejected | PASS |

| TEST-SIG-013 | Too-short signal | Analysis rejected | PASS |

| TEST-SIG-014 | Nonuniform timestamps | CSV loader rejected input | PASS |

| TEST-SIG-015 | Fixed random seed | Synthetic noisy signal reproducible | PASS |





## Notes



- Numerical tolerances are tied to the properties of the algorithms being tested.

- FFT frequency accuracy is evaluated relative to the frequency resolution Δf = Fs / N.

- Welch PSD was independently validated by comparing integrated PSD power with time-domain signal power.

- ASD was validated using ASD² = PSD.

- Spectrogram behavior was validated using a known intermittent interference interval.

- Nonuniform timestamp validation is performed at the CSV input layer because timestamps are not retained in SignalRecord after successful loading.

---

# 12. Real-World Validation

V0.1 was also evaluated using a public real-world vibration measurement from the University of Patras Floating Wind Turbine dataset.

The selected file was:

```text
H_WD1_WS1_01.csv
```

The measurement was a headerless single-column CSV containing voltage samples.

Known dataset properties used for validation:

```text
Sampling frequency: 1024 Hz
Number of samples:  30720
Duration:           30 s
Nyquist frequency:  512 Hz
Units:              V
```

The analyzer successfully:

- loaded the headerless CSV;
- preserved the full sample count;
- used the externally supplied sampling frequency;
- calculated time-domain statistics;
- generated an amplitude spectrum;
- calculated Welch PSD;
- calculated ASD;
- generated a spectrogram;
- detected spectral peaks;
- saved analysis configuration and numerical results as JSON;
- saved waveform, spectrum, PSD, ASD, and spectrogram plots.

Observed time-domain values for the validation record were approximately:

```text
Mean:               -1.63e-4 V
RMS:                 1.81e-3 V
Standard deviation:  1.80e-3 V
```

The real-world spectrum contained multiple low-frequency components and additional spectral structure at higher frequencies.

A peak-detection configuration of:

```text
Minimum prominence: 5e-5 V
Minimum spacing:     1 Hz
```

produced a practical reduced set of detected spectral peaks.

Examples of detected peaks included components near:

```text
4.7 Hz
13.3 Hz
14.6 Hz
28.8 Hz
50.0 Hz
102.5 Hz
```

These values are reported only as detected spectral peaks.

No physical interpretation was assigned to these peaks during V0.1 validation. A detected spectral peak should not automatically be interpreted as:

- a physical noise source;
- a rotor frequency;
- a structural resonance;
- a harmonic;
- a fault;
- a causal mechanism.

Additional physical information and engineering evidence would be required for those conclusions.

## 12.1 Real-World Validation Findings

The real-world evaluation exposed practical issues that were not fully visible when using controlled synthetic data.

### Headerless Measurement Files

The original CSV loader assumed that the CSV file contained a header row.

However, the real vibration dataset contained numerical samples starting directly from the first row.

For example:

```text
-0.001878473993
-0.001975826339
-0.002335281155
-0.002462588069
...
```

The loader was therefore extended to explicitly support headerless single-column CSV files.

The real dataset was loaded using:

```python
record = load_signal_csv(
    file_path="test_data/real/H_WD1_WS1_01.csv",
    sample_rate_hz=1024.0,
    channel_name="H_WD1_WS1_01",
    units="V",
    has_header=False,
)
```

This preserved all 30720 samples.

### Signal-Scale-Dependent Peak Detection

The first real-world analysis used:

```text
peak_min_prominence = 0.01 V
```

No spectral peaks were detected.

This was not because the signal contained no frequency structure. The measured signal itself had an RMS value of approximately:

```text
0.00181 V
```

Therefore, an absolute prominence requirement of `0.01 V` was much too large for this signal.

The prominence threshold was subsequently reduced to:

```text
peak_min_prominence = 5e-5 V
```

This revealed many valid local spectral maxima. However, a large number of closely spaced peaks were then detected, particularly in the low-frequency region.

A minimum frequency-spacing requirement was therefore also introduced:

```text
peak_min_distance_hz = 1.0
```

This reduced closely spaced detections and resulted in a more manageable peak list.

The final real-world validation configuration therefore used:

```text
peak_min_prominence = 5e-5 V
peak_min_distance_hz = 1.0 Hz
```

This experiment demonstrated that an absolute peak-prominence threshold is signal-scale dependent.

Automatic noise-floor-aware or adaptive peak selection is deferred to a later version.

## 12.2 Real-World Analysis Configuration

The real-world validation used the following analysis configuration:

```text
Sampling frequency:          1024 Hz
Number of samples:           30720
Duration:                    30 s
Nyquist frequency:           512 Hz

Spectrum window:             Hann

PSD nperseg:                 4096
PSD overlap:                 2048
PSD window:                  Hann

Spectrogram nperseg:         1024
Spectrogram overlap:         512
Spectrogram window:          Hann

Peak minimum prominence:     5e-5 V
Peak minimum distance:       1.0 Hz
```

The analysis generated:

```text
analysis_config.json
results.json
waveform.png
spectrum.png
psd.png
asd.png
spectrogram.png
```

This demonstrated that a real measurement can pass through the complete V0.1 analysis pipeline and produce a reproducible engineering analysis package.

## 12.3 What the Real-World Test Validated

The real-world validation demonstrated the following complete workflow:

```text
Real measurement CSV
        ↓
CSV loader
        ↓
SignalRecord
        ↓
Input validation
        ↓
Time-domain statistics
        ↓
Amplitude spectrum
        ↓
Welch PSD
        ↓
ASD
        ↓
Spectrogram
        ↓
Spectral peak detection
        ↓
Structured analysis result
        ↓
JSON results
        ↓
Saved engineering plots
```

The purpose of this experiment was not to diagnose the physical system. Its purpose was to determine whether the validated V0.1 DSP pipeline could operate sensibly on a real measurement rather than only on controlled synthetic signals.

That objective was achieved.

---

# 13. V0.1 Limitations

V0.1 intentionally provides a validated single-channel DSP analysis foundation.

It is not yet intended to perform complete noise-source investigation or autonomous engineering diagnosis.

## 13.1 Single-Channel Analysis

V0.1 analyzes one signal channel at a time.

It does not yet perform:

- cross-correlation between channels;
- cross-spectral density;
- coherence analysis;
- transfer-function estimation;
- multichannel source comparison;
- multichannel source localization.

These capabilities are planned for later versions.

## 13.2 Real-Valued Signals

V0.1 currently assumes real-valued sampled signals.

Complex IQ data is not yet supported by the main analysis pipeline. Complex SDR/IQ analysis may be introduced later if required by the application.

## 13.3 Uniform Sampling

V0.1 assumes uniformly sampled data.

When timestamps are supplied, the loader checks whether the sampling intervals are sufficiently uniform. Nonuniformly sampled data is rejected rather than silently resampled.

Automatic resampling is not implemented in V0.1.

## 13.4 CSV Input

The primary supported input format in V0.1 is CSV.

The loader currently supports:

- CSV files containing a signal column;
- CSV files containing time and signal columns;
- headerless single-column CSV measurement files.

Additional formats such as HDF5, MAT, NPY, NPZ, TDMS, WAV, and GWpy-compatible data may be added in later versions when required.

## 13.5 Sampling Frequency

If timestamps are unavailable, the sampling frequency must be supplied externally.

The analyzer must not guess the sampling frequency. Without a valid sampling frequency, frequency-domain quantities cannot be interpreted correctly.

## 13.6 Peak Detection

Peak detection currently uses explicitly configured parameters, including:

- minimum prominence;
- minimum frequency distance.

The real-world validation demonstrated that absolute peak-prominence thresholds are signal-scale dependent.

A threshold appropriate for a 1 V synthetic signal may be inappropriate for a millivolt-level sensor measurement.

V0.1 therefore does not claim to provide a universal automatic peak-detection threshold.

Automatic noise-floor-aware peak detection is deferred to a later version.

## 13.7 Peak Detection Is Not Source Identification

A detected spectral peak is a measurement.

It does not automatically identify the physical cause of that component.

The following chain must remain conceptually separate:

```text
Measurement
    ↓
Detection
    ↓
Interpretation
    ↓
Hypothesis
    ↓
Physical validation
```

For example, observing a spectral peak at 50 Hz does not by itself prove that the source is mains interference.

Additional evidence would be required.

## 13.8 No Automatic Harmonic Interpretation

V0.1 does not automatically determine whether spectral peaks form:

- harmonic families;
- intermodulation products;
- sidebands;
- mechanical orders;
- switching-frequency families;
- modulation products.

These analyses may be added during noise and interference characterization.

## 13.9 No Automatic Noise-Floor Estimation

Although Welch PSD is available, V0.1 does not yet automatically estimate a representative broadband noise floor.

Therefore, quantities such as automatic SNR, signal-to-noise-floor ratio, peak significance, and band-specific noise level are not yet part of the standard pipeline.

## 13.10 No Automatic Event Detection

The spectrogram can reveal time-varying interference.

However, V0.1 does not automatically identify or classify transient events.

Event detection will be introduced in a later version.

## 13.11 No Physical Source Attribution

V0.1 does not determine which physical device or mechanism produced an observed component.

It does not yet perform:

- source attribution;
- causal inference;
- coupling-path identification;
- environmental source matching;
- sensor-to-sensor evidence combination.

These tasks require additional measurements and later multichannel investigation tools.

## 13.12 No Noise Suppression

V0.1 performs measurement and characterization only.

It does not currently perform:

- adaptive filtering;
- notch-filter recommendation;
- interference cancellation;
- spectral subtraction;
- regression subtraction;
- Wiener filtering;
- reference-channel subtraction.

Suppression techniques will only be considered after reliable detection and characterization are established.

## 13.13 No Machine Learning

V0.1 uses deterministic classical DSP methods.

No machine-learning model is required for the current analysis.

Machine learning may be introduced later only where it provides a clear advantage over classical DSP.

## 13.14 No LLM Analysis

V0.1 does not use a large language model for numerical signal processing.

All numerical measurements are produced by validated Python DSP functions.

Future LLM components may receive structured results such as statistics, spectral peaks, PSD measurements, spectrogram events, coherence results, and band-power measurements, and use them to help decide what validated analysis tool should run next.

The LLM will not be treated as a numerical DSP engine.

## 13.15 No Agentic Investigation Yet

V0.1 is not yet the full agentic analyzer.

The current architecture provides the deterministic DSP foundation required before agentic decision-making is introduced.

A future investigation agent may follow a process such as:

```text
Observe measurement
        ↓
Request validated DSP analysis
        ↓
Receive structured result
        ↓
Evaluate evidence
        ↓
Choose next analysis tool
        ↓
Repeat investigation
        ↓
Generate engineering interpretation
```

The agent will orchestrate validated tools rather than replace them.

---

# 14. V0.1 Completion Status

V0.1 provides a validated single-channel signal-analysis foundation.

The completed capabilities include:

- synthetic reference signal generation;
- CSV loading;
- headerless CSV loading;
- sampling-frequency handling;
- timestamp-based sampling-frequency inference;
- nonuniform timestamp detection;
- signal metadata representation using `SignalRecord`;
- input validation;
- time-domain statistics;
- waveform generation;
- one-sided amplitude spectrum;
- Hann window support;
- Welch PSD;
- ASD;
- spectrogram analysis;
- spectral peak detection;
- integrated single-channel analysis pipeline;
- reproducible JSON configuration;
- reproducible JSON numerical results;
- saved waveform plots;
- saved spectrum plots;
- saved PSD plots;
- saved ASD plots;
- saved spectrogram plots;
- synthetic validation matrix;
- real-world measurement validation.

The implementation has been evaluated using both controlled synthetic signals and real experimental vibration data.

V0.1 is therefore considered complete as the validated DSP foundation of the Agentic Noise & Interference Analyzer.

Later versions will build more advanced characterization, multichannel analysis, source investigation, machine learning where justified, and agentic decision-making on top of this foundation.

---

# Final Validation Principle

The purpose of the validation process is not merely to show that the software runs.

It is to establish that:

> **When the analyzer reports a numerical signal measurement, we have engineering evidence that the measurement can be trusted within clearly defined assumptions and limitations.**

This principle will remain important when AI and agentic investigation are introduced in later versions.