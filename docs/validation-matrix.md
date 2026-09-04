# V0.1 DSP Validation Matrix

## Agentic Noise & Interference Analyzer

**Version:** V0.1  
**Status:** Approved baseline

---

## 1. Purpose

This document defines how the numerical DSP components of V0.1 will be validated.

The central rule is:

> **A reasonable-looking plot is not sufficient evidence that a DSP implementation is correct.**

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

---

# 2. Synthetic Reference Signals

## TEST-SIG-001 — Pure Sinusoid

Parameters:

```text
Sampling frequency: 10000 Hz
Duration:           5 s
Amplitude:          2
Frequency:          1000 Hz
Number of samples:  50000
```

Signal:

$$
x(t)=2\sin(2\pi1000t)
$$

Expected:

```text
Mean ≈ 0
RMS  = 2/√2
     ≈ 1.41421356

Dominant frequency ≈ 1000 Hz
```

Because the tone is bin-centered for this record length, it provides a clean initial FFT validation case.

---

## TEST-SIG-002 — Constant Signal

Signal:

$$
x[n]=3
$$

Expected:

```text
Mean             = 3
Median           = 3
RMS              = 3
Standard deviation = 0
Variance         = 0
Minimum          = 3
Maximum          = 3
Peak-to-peak     = 0
```

The analyzer should also identify the signal as constant.

---

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

---

## TEST-SIG-004 — Sinusoid with Gaussian Noise

Parameters:

```text
Sampling frequency: 10000 Hz
Duration:           10 s
Tone frequency:     1000 Hz
Tone amplitude:     1
Noise sigma:        0.1
Random seed:        fixed
```

Expected:

- clear spectral component near 1000 Hz;
- broadband noise floor;
- PSD noise floor greater than the corresponding clean sinusoid.

An exact PSD-floor acceptance value should not be frozen until the estimator normalization and averaging behavior are derived and verified.

---

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

---

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

---

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

---

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

---

## TEST-SIG-009 — NaN Input

Insert one or more NaN samples.

Expected:

```text
Validation failure
```

The analyzer must not silently interpolate or remove NaN values.

---

## TEST-SIG-010 — Infinite Input

Insert positive or negative infinity.

Expected:

```text
Validation failure
```

---

## TEST-SIG-011 — Empty Input

Provide an empty signal.

Expected:

```text
Input rejected
Clear validation reason returned
```

---

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

---

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

---

## TEST-SIG-014 — Nonuniform Timestamps

Construct a signal with intentionally irregular time spacing.

Expected:

```text
Nonuniform sampling detected
```

The analyzer must not silently assume a constant sampling frequency or silently resample the signal.

The exact numerical uniformity tolerance will be determined and documented during implementation.

---

## TEST-SIG-015 — Reproducible Random Signal

Generate a noisy signal using a fixed random seed.

Initial seed:

```text
42
```

Expected:

Repeated generation with identical parameters and seed should produce identical samples.

This test supports reproducible numerical validation.

---

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

---

# 4. Tolerance Philosophy

Numerical tolerances should be derived from the properties of the algorithm whenever possible.

For example, FFT frequency resolution is:

$$
\Delta f=\frac{F_s}{N}
$$

Therefore, frequency-estimation acceptance should be related to \(\Delta f\) rather than using an arbitrary tolerance such as:

```text
±1 Hz
```

Similarly, spectrogram event timing should be evaluated relative to the time resolution produced by its segment and overlap parameters.

Floating-point equality tests should use appropriate numerical tolerances rather than direct equality where necessary.

---

# 5. PSD Power Consistency

Where applicable, the integrated PSD should be compared with the corresponding time-domain power.

Conceptually:

$$
P_{\mathrm{spectral}}\approx\int PSD(f)\,df
$$

For a suitable zero-mean signal, this should correspond to its time-domain mean-square power.

The exact acceptance tolerance will be established experimentally based on the Welch configuration, signal length, and numerical integration method.

This provides an important independent consistency check beyond inspecting the PSD plot.

---

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

---

# 7. Validation of Reused DSP Code

Some DSP implementations may be adapted from the earlier `dsp-ai-agent` project.

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

---

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

---

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

---

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

---

## Final Validation Principle

The purpose of the validation process is not merely to show that the software runs.

It is to establish that:

> **When the analyzer reports a numerical signal measurement, we have engineering evidence that the measurement can be trusted within clearly defined assumptions and limitations.**

That principle will remain important when AI and agentic investigation are introduced in later versions.