# Real-Time Audio Signal Graph & Spatial DSP Architecture

## 1. Executive Summary

This document specifies the real-time audio synthesis, adaptive acoustic masking, and spatial binaural rendering graph deployed across `RainAI` and integrated with `spodeian-core`. The engine provides sample-accurate DSP processing, SIMD vectorization, sub-millisecond adaptive filtering, and telemetry streaming into `spodeian-lake`.

```mermaid
flowchart TD
    subgraph SoundSources ["Procedural Sound Generators"]
        S1["Rain Impact Synthesis<br/>(Poisson droplet events)"]
        S2["Wind & Aeroacoustics<br/>(Pink noise + Resonant bandpass)"]
        S3["Thunder Physical Model<br/>(Nonlinear shockwave dispersion)"]
    end

    subgraph SpatialGraph ["Binaural & Environmental DSP"]
        Mix["Dynamic Mixer"]
        HRTF["KEMAR HRTF Binaural Convolver<br/>(Azimuth / Elevation Filtering)"]
        Rev["Schroeder-Moorer Feedback Reverb<br/>(Late diffuse field)"]
    end

    subgraph AdaptiveAcoustics ["Adaptive Noise Masker"]
        Mic["Microphone Reference Input"]
        RLS["Recursive Least Squares (RLS) Filter<br/>(Dynamic spectral weight adaptation)"]
    end

    subgraph Output ["Audio Output & Telemetry"]
        DAC["Audio Hardware (WASAPI / ASIO)"]
        Snap["AcousticTelemetrySnapshot<br/>(RMS, Centroid, ZCR, Underruns)"]
        Lake["spodeian-lake<br/>(Parquet Store)"]
    end

    S1 --> Mix
    S2 --> Mix
    S3 --> Mix
    Mix --> HRTF
    HRTF --> Rev
    Rev --> RLS
    Mic --> RLS
    RLS --> DAC
    RLS --> Snap
    Snap --> Lake
```

---

## 2. DSP Graph Stages

### 2.1 Procedural Synthesis Engine
- **Poisson Droplet Generator**: Stochastic arrival times with droplet mass distributions mapped to damped sinusoidal impact resonances:
  $$s(t) = A e^{-\gamma t} \sin(\omega_0 t + \phi)$$
- **Wind Turbulence**: Multi-band resonant noise filtering with low-frequency oscillator (LFO) modulation for gust dynamics.
- **Physical Shockwaves**: Thunder synthesis using acoustic wave propagation delay models with high-frequency absorption ($1/f^\alpha$).

### 2.2 Spatialization & HRTF Convolution
- **Binaural Impulse Responses**: KEMAR measurements parameterized by azimuth $\theta \in [-180^\circ, 180^\circ]$ and elevation $\phi \in [-40^\circ, 90^\circ]$.
- **Time-Domain vs. Frequency-Domain**: Partitioned convolution using overlap-add (OLA) or direct time-domain FIR for zero-latency interactive head tracking.
- **Interaural Delays**: Accurate Interaural Time Difference (ITD) and Interaural Level Difference (ILD) rendering.

### 2.3 Adaptive Acoustic Masker (RLS)
- **Objective**: Mask ambient background disturbances using procedural soundscape shaping.
- **Recursive Least Squares (RLS) Algorithm**:
  $$k(n) = \frac{\lambda^{-1} P(n-1) x(n)}{1 + \lambda^{-1} x^T(n) P(n-1) x(n)}$$
  $$\xi(n) = d(n) - w^T(n-1) x(n)$$
  $$w(n) = w(n-1) + k(n) \xi(n)$$
  $$P(n) = \lambda^{-1} P(n-1) - \lambda^{-1} k(n) x^T(n) P(n-1)$$
  where $\lambda \in (0, 1]$ is the forgetting factor.

### 2.4 Diagnostic Telemetry & Telemetry Ingestion
- **Snapshot Schema**: [`AcousticTelemetrySnapshot`](file:///crates/spodeian-lake/src/schema.rs):
  - `timestamp`: i64 epoch ms
  - `frame_idx`: Audio buffer sequence index
  - `rms_db`, `peak_db`: Output power metrics
  - `spectral_centroid`: Spectral brightness indicator
  - `zero_crossing_rate`: High-frequency noise proxy
  - `active_voices`, `buffer_underruns`: Real-time engine health metrics

---

## 3. Real-Time Processing Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant AudioThread as Audio Callback Thread
    participant Synth as Sound Synthesis Graph
    participant HRTF as HRTF Spatializer
    participant RLS as RLS Adaptive Masker
    participant Telemetry as Telemetry Ring Buffer
    participant Lake as Windows Data Lake

    AudioThread->>Synth: Request audio frame (e.g. 512 samples)
    Synth->>Synth: Synthesize droplet & turbulence samples (SIMD)
    Synth->>HRTF: Send interleaved multi-channel stream
    HRTF->>HRTF: Convolve with head orientation impulse response
    HRTF->>RLS: Spatialized stereo stream
    RLS->>RLS: Adapt filter weights to ambient mic noise
    RLS-->>AudioThread: Rendered stereo audio buffer
    RLS->>Telemetry: Push AcousticTelemetrySnapshot (lock-free)
    Note over Telemetry,Lake: Asynchronous background worker
    Telemetry->>Lake: Batch Parquet store
```

---

## 4. Performance Guarantees
- **Zero Allocations**: Audio callback never allocates, locks, or performs filesystem I/O.
- **Lock-Free Communication**: DSP thread pushes diagnostic events via bounded crossbeam lock-free ring buffers.
- **Hardware Acceleration**: Autovectorized loops utilizing AVX2 / SSE4.1 / NEON instructions.
