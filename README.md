# Agentic Noise & Interference Analyzer

A DSP-based framework for detecting, characterizing, and investigating noise and interference in sampled signals, with a long-term goal of supporting agentic AI-assisted signal investigation.

> **Project status:** Early development - V0.1 specification and validation framework defined.

---

## Overview

The **Agentic Noise & Interference Analyzer** is being developed as a general-purpose engineering tool for analyzing noise, interference, and other disturbances in recorded signals.

The project begins with a validated classical Digital Signal Processing (DSP) foundation. Later versions will introduce multichannel analysis, automated investigation workflows, local LLM integration, and agentic selection of validated DSP tools.

A central design principle is:

> **The DSP tools perform the measurements. The AI decides which validated measurement to request and helps interpret the resulting evidence.**

The AI layer will not be allowed to invent numerical signal-analysis results.

---

## Project Goals

The long-term system is intended to support tasks such as:

- inspecting recorded signals;
- detecting dominant spectral components;
- characterizing broadband and narrowband noise;
- identifying intermittent disturbances;
- analyzing signals in the time, frequency, and time-frequency domains;
- comparing multiple measurement channels;
- investigating possible relationships between noise sources and observed disturbances;
- assisting engineers in deciding which analysis should be performed next;
- preserving analysis parameters and numerical evidence for reproducibility;
- supporting future experiments in noise suppression and interference mitigation.

The project is designed to remain useful as a standalone DSP analyzer even without the AI layer.

---

## Engineering Philosophy

The project follows the development sequence:

```text
Understand the engineering problem
        ↓
Define the DSP measurement
        ↓
Create a controlled reference signal
        ↓
Implement the DSP method
        ↓
Verify the numerical result
        ↓
Integrate the validated tool
```

Every important DSP operation should be validated using signals for which the expected result is known.

A reasonable-looking plot alone is not considered sufficient validation.

---

## Scientific Interpretation Principle

The analyzer will maintain a distinction between:

```text
Measurement
    ↓
Detection
    ↓
Interpretation
    ↓
Hypothesis
```

For example, detecting strong coherence between two measurement channels does not by itself prove that one signal caused the other.

Source attribution and causal conclusions require additional engineering evidence and experimental validation.

---

# V0.1 - Core Single-Channel Signal Analyzer

The first version focuses entirely on building and validating the DSP foundation.

The V0.1 objective is:

> Given one sampled signal and sufficient sampling information, produce trustworthy and reproducible time-domain, frequency-domain, and time-frequency-domain measurements.

V0.1 initially focuses on **real-valued, uniformly sampled, single-channel signals stored in CSV files**.

---

## Planned V0.1 Capabilities

### Input and Validation

- CSV signal loading
- sampling-frequency handling
- signal metadata
- NaN and Inf detection
- empty-signal detection
- constant-signal detection
- sampling validation
- nonuniform timestamp detection

### Time-Domain Analysis

- mean
- median
- RMS
- variance
- standard deviation
- minimum
- maximum
- peak-to-peak amplitude

### Frequency-Domain Analysis

- FFT
- one-sided amplitude spectrum
- frequency-axis generation
- frequency-resolution reporting
- windowing
- dominant spectral-component detection

### Noise Spectral Analysis

- Welch Power Spectral Density (PSD)
- Amplitude Spectral Density (ASD)

### Time-Frequency Analysis

- spectrogram

### Visualization

- waveform
- amplitude spectrum
- PSD
- ASD
- spectrogram

### Reproducibility

Analysis runs will preserve important information such as:

- input information;
- sampling frequency;
- sample count;
- signal duration;
- analysis methods;
- DSP parameters;
- software version;
- numerical results.

---

## Validation Strategy

DSP modules will be tested using controlled synthetic signals before being used on real-world measurements.

Initial reference signals will include:

1. pure sinusoid;
2. constant signal;
3. two-tone signal;
4. sinusoid with Gaussian noise;
5. increased-noise signal;
6. DC offset with sinusoid;
7. intermittent interference;
8. non-bin-centered sinusoid;
9. signal containing NaN;
10. signal containing Inf;
11. empty signal;
12. invalid sampling frequency;
13. very short signal;
14. nonuniformly sampled signal.

Random test signals will use fixed seeds where appropriate so that experiments remain reproducible.

---

## Planned Architecture

```text
Input Signal
     ↓
Input Validation
     ↓
Signal Representation
     ↓
Validated DSP Tools
     ↓
Structured Numerical Results
     ↓
Visualization
     ↓
Reproducible Analysis Record
```

Later versions will extend this architecture toward:

```text
User
  ↓
Investigation Agent
  ↓
Local LLM
  ↓
Validated DSP Tool Registry
  ↓
DSP Measurements
  ↓
Structured Evidence
  ↓
Next Investigation Step
```

---

## Planned Technology Stack

Initial development:

- Python
- NumPy
- SciPy
- pandas
- Matplotlib
- pytest

Later versions may introduce:

- Streamlit or another lightweight user interface
- Ollama for local LLM integration
- machine-learning tools where scientifically justified
- MCP-based tool exposure if useful

---

## Project Structure

```text
agentic-noise-interference-analyzer/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── v0.1-engineering-spec.md
│   ├── validation-matrix.md
│   └── roadmap.md
├── src/
│   └── noise_analyzer/
│       └── __init__.py
├── tests/
├── test_data/
│   └── synthetic/
├── experiments/
└── outputs/
```

The architecture will grow progressively. Directories for future AI, agent, ML, or MCP components will be added only when those components are actually implemented.

---

## Relationship to Previous Work

Some deterministic DSP components developed in the earlier **DSP AI Agent** project may be reviewed and adapted for this project.

Existing implementations will not automatically be considered validated for this application. Reused components must pass the validation requirements of this project before integration.

This allows previous work to be reused while maintaining an independent scientific validation standard.

---

## Development Roadmap

The current high-level roadmap is:

```text
V0.1  Core Single-Channel Signal Analyzer
  ↓
V0.2  Spectral and Noise Characterization
  ↓
V0.3  Event and Anomaly Detection
  ↓
V0.4  Multichannel Analysis
  ↓
V0.5  Noise-Source Investigation
  ↓
V0.6  Local LLM Engineering Assistant
  ↓
V0.7  Agentic DSP Tool Selection
  ↓
V0.8  ML-Assisted Analysis
  ↓
V0.9  Source-Attribution and Suppression Experiments
  ↓
V1.0  Stable Analyzer
```

The roadmap may evolve as the DSP validation work and experimental requirements become clearer.

---

## Current Status

The project is currently at the beginning of **V0.1**.

Completed:

- initial project concept;
- V0.1 scope definition;
- V0.1 engineering specification;
- initial validation matrix;
- review of potentially reusable DSP components from previous work.

Next:

- establish the Python project environment;
- implement deterministic synthetic reference-signal generation;
- begin V0.1 module development and numerical validation.

---

## License

License information will be added according to the selected project license.