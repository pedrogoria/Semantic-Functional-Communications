# Configuration Reference

This document mirrors the of the configuration structure.This document mirrors the intent of:

---

## Main Sections Covered

- Figure / experiment metadata
- Execution modes
- System parameters
- Signal parameters
- Signal models for comparison experiments
- DC handling
- Global quantization policy
- Benchmark settings
- Acquisition methods
  - Nyquist
  - Compressive Sensing / CS
  - PPM
  - Send-on-Delta / SoD
  - Finite Rate of Innovation / FRI-inspired
- RbCP settings
- RbCP_time settings
- Channel / PHY / SFC detection settings
- Semantic-based error detection / SED
- Duplicate RX settings
- Sweep parameters
- Monte Carlo
- Metrics
- Output
- Data format
- Plot settings
- Comparison settings
- Reproducibility

---

# Notes on Current Conventions

## Signal Period

Prefer:

```yaml
signal:
  tau: 1.0
```

in new configs.

The legacy alias:

```yaml
signal:
  T: 1.0
```

is still documented because some older figure pipelines accept it.

If both are present, keep both values numerically consistent.

---

## Bandwidth

`system.B` is the **total system bandwidth**.

Per-sensor bandwidth slices are derived inside the core from:

```yaml
system:
  bandwidth_allocation: null
```

If `bandwidth_allocation` is `null`, the system uses equal bandwidth sharing among all `S` sensors.

If `bandwidth_allocation` is provided, it must satisfy:

```text
length = S
all entries >= 0
sum = 1
```

Example:

```yaml
system:
  S: 4
  B: 10000.0
  bandwidth_allocation:
    - 0.25
    - 0.25
    - 0.25
    - 0.25
```

---

## Source Bandwidth

The signal/source bandwidth is configured through:

```yaml
signal:
  W: 10.4
```

The source signal filter should use `signal.W` directly.

Do **not** redefine filtering bandwidth from the number of harmonics as:

```text
W_eff = 2 * N / tau
```

The current convention is:

```text
W controls source filtering.
N controls representation harmonics.
```

---

## Power and Noise Conventions

Several experiments use different physical regimes.

### Fixed-SNR Regime

Some Figure-5-style pipelines keep SNR fixed and derive power as bandwidth changes:

```text
P(B) = SNR * B * N0
```

Relevant config helpers:

```yaml
system:
  manuscript_faithful_power_model: true
  derive_P_from_fixed_N0: true
```

### Fixed-Power Regime

Some Figure-6 / Figure-7-style pipelines keep `P` and `N0` fixed and derive SNR as bandwidth changes:

```text
SNR(B) = P / (B * N0)
```

Relevant config helpers:

```yaml
system:
  fixed_power_model: true
  derive_snr_from_fixed_P_and_N0: true
```

### SFC Physical Channel Noise

For SFC physical simulation, the project convention is:

```text
y = sqrt(E_chip) * superposed + n
n ~ CN(0, N0)
```

Therefore, SFC physical AWGN simulation uses `N0` directly at the matched-filter / resource-output level. It does not generate AWGN with variance `B*N0` or `B_s*N0`.

---

# Figure / Experiment Metadata

Section:

```yaml
figure:
```

Purpose: store human-readable metadata for plots, output folders, and paper tracking.

Typical keys:

```yaml
figure:
  id: "template_experiment"
  manuscript_figure: null
  title: "Template experiment"
  description: >
    Central master catalog of configurable keys supported by the current
    SFC/RbCP core, acquisition baselines, and figure pipelines.
```

## Keys

### `figure.id`

Free string identifier.

Examples:

```yaml
id: "rbcp_mse_vs_B"
id: "sfc_duplicate_rx"
id: "fair_methods_comparison_vs_B"
id: "fri_cs_rbcp_comparison_vs_B"
```

---

### `figure.manuscript_figure`

Optional manuscript figure number.

Examples:

```yaml
manuscript_figure: 4
manuscript_figure: 5
manuscript_figure: null
```

---

### `figure.title`

Human-readable title used by runners and plots.

---

### `figure.description`

Long-form description for documentation and metadata.

---

# Execution Modes

Section:

```yaml
mode:
```

Purpose: enable or disable branches in experiment pipelines.

Typical catalog block:

```yaml
mode:
  run_benchmark: false
  run_nyquist: false
  run_rbcp: false
  run_rbcp_time: false
  run_sfc: false
  run_sfc_sed: false

  run_cs: false
  run_ppm: false
  run_sod: false
  run_fri: false

  semantic_error_detection: false

  monte_carlo_random_signals: false
  monte_carlo_uniform_t: false
```

## Main Branches

### `mode.run_benchmark`

Enables benchmark branches in pipelines that support benchmark / Nyquist-style baselines.

---

### `mode.run_nyquist`

Enables explicit Nyquist acquisition branches in newer acquisition-based pipelines.

---

### `mode.run_rbcp`

Enables RbCP branches.

---

### `mode.run_rbcp_time`

Enables RbCP_time branches.

RbCP_time is typically an error-free time-grid reference:

```text
signal -> ta/tb -> events -> ta/tb -> signal
```

---

### `mode.run_sfc`

Enables SFC branches using the physical SFC channel stack.

---

### `mode.run_sfc_sed`

Enables SFC + semantic error detection if supported by the pipeline.

---

## Additional Acquisition Baselines

### `mode.run_cs`

Enables Compressive Sensing baseline branches.

---

### `mode.run_ppm`

Enables PPM baseline branches.

---

### `mode.run_sod`

Enables Send-on-Delta baseline branches.

---

### `mode.run_fri`

Enables FRI-inspired sparse-innovation baseline branches.

---

## Semantic Error Detection

### `mode.semantic_error_detection`

If true, the pipeline applies semantic-based error detection after SFC detection.

Current SED logic:

```text
duplicate t_z^(s,n)        -> invalid period
ta without tb              -> invalid period
tb without ta              -> invalid period
both ta and tb absent      -> valid absent pair
```

SED does **not** require exactly `2*N*S` detected events per period.

---

## Duplicate RX Monte Carlo Modes

### `mode.monte_carlo_random_signals`

Used by duplicate-reception experiments.

Enables Monte Carlo trials with random signals.

---

### `mode.monte_carlo_uniform_t`

Used by duplicate-reception experiments.

Enables Monte Carlo trials with uniformly distributed `ta/tb` phase parameters.

---

# System Parameters

Section:

```yaml
system:
```

Purpose: define global physical communication parameters.

Typical block:

```yaml
system:
  S: 8
  P: 1.0
  B: 10000.0
  R: 12
  L: 4
  SNR_dB: -16.0
  N0: 0.007

  manuscript_faithful_power_model: false
  derive_P_from_fixed_N0: false

  fixed_power_model: false
  derive_snr_from_fixed_P_and_N0: false

  bandwidth_allocation: null
```

---

## `system.S`

Number of sensors / signals.

Type:

```text
int >= 1
```

---

## `system.P`

Average transmit power parameter.

Type:

```text
float > 0
```

Usage depends on the physical regime:

- fixed-power experiments keep `P` constant;
- fixed-SNR experiments may derive `P(B)` pointwise.

---

## `system.B`

Total system bandwidth.

Type:

```text
float > 0
```

Important:

```text
system.B is total bandwidth, not per-sensor bandwidth.
```

---

## `system.R`

Number of resources / subcarriers used by SFC maps.

Type:

```text
int >= 1
```

For theorem-consistent SFC maps, the map generator partitions `R` resources into `L` row-resource groups:

```text
D_1, D_2, ..., D_L
```

Each row `l` can activate a resource from `D_l`.

The number of theorem-valid maps is:

```text
|D_1| * |D_2| * ... * |D_L|
```

For standard Fourier/phase SFC event IDs:

```text
num_event_ids = 2 * N * S
```

A valid SFC map configuration should satisfy:

```text
product_l |D_l| >= 2 * N * S
```

---

## `system.L`

Number of temporal resources / sub-symbols / rows per SFC map.

Type:

```text
int >= 1
```

---

## `system.SNR_dB`

Signal-to-noise ratio in dB.

Used directly in some pipelines and overwritten/derived in others.

---

## `system.N0`

Noise parameter.

Type:

```text
float > 0
```

In total-band diagnostic formulas:

```text
SNR = P / (B * N0)
```

In SFC physical AWGN simulation, `N0` is used directly at the matched-filter / resource-output level.

---

## Power-Model Helpers

### `system.manuscript_faithful_power_model`

Boolean helper for Figure-5-style behavior.

---

### `system.derive_P_from_fixed_N0`

If true, a pipeline may derive:

```text
P(B) = SNR * B * N0
```

---

### `system.fixed_power_model`

Boolean helper for fixed-power experiments.

---

### `system.derive_snr_from_fixed_P_and_N0`

If true, a pipeline may derive:

```text
SNR(B) = P / (B * N0)
```

---

## `system.bandwidth_allocation`

Per-sensor bandwidth allocation.

Examples:

```yaml
bandwidth_allocation: null
```

or:

```yaml
bandwidth_allocation:
  - 0.25
  - 0.25
  - 0.25
  - 0.25
```

Rules:

```text
length = S
entries >= 0
sum = 1
```

---

# Signal Parameters

Section:

```yaml
signal:
```

Purpose: define source signal parameters and simulation grid.

Typical block:

```yaml
signal:
  W: 10.4
  tau: 1.0
  T: 1.0
  Tt: 0.01
  N_override: null
  distribution: "uniform"
  peak_to_peak: 2.0
  normalize_dft: true
  normalization_target: 3.9
  dft_signal_periods: 1
  n_periods: 1
  threshold_harmonics: 0.001
```

---

## `signal.W`

Source signal bandwidth.

Type:

```text
float > 0
```

Used for signal filtering.

---

## `signal.tau`

Signal period / frame duration.

Preferred modern key.

---

## `signal.T`

Legacy alias for `signal.tau`.

Keep consistent with `signal.tau` if both are present.

---

## `signal.Tt`

Simulation time step.

Smaller `Tt` improves waveform resolution but increases runtime.

---

## `signal.N_override`

Optional override for number of harmonics.

If unset, the core may derive:

```text
N = floor(W * tau / 2)
```

---

## `signal.distribution`

Monte Carlo signal distribution.

Supported:

```yaml
distribution: "uniform"
distribution: "gaussian"
```

---

## `signal.peak_to_peak`

Target peak-to-peak amplitude after filtering.

---

## `signal.normalize_dft`

Whether to use Fourier-core normalization.

---

## `signal.normalization_target`

Target norm used when `normalize_dft = true`.

---

## `signal.dft_signal_periods`

Optional number of concatenated periods used by some older figure code paths.

---

## `signal.n_periods`

Number of periods used by SFC / fair-method pipelines.

---

## `signal.threshold_harmonics`

Harmonic threshold used by `PhaseCoefficientCore`.

If a Fourier coefficient is null or below threshold, the corresponding `ta` or `tb` may be omitted.

This is why SED must allow sparse absent pairs.

---

# Signal Models for Method-Comparison Experiments

Section:

```yaml
signal_model:
```

Purpose: define high-level signal classes for reviewer-response and method-comparison experiments.

Typical block:

```yaml
signal_model:
  type: "bandlimited_random"

  sparse_innovation:
    K: 3
    kernel: "gaussian"
    kernel_sigma: 0.01
    amplitude_distribution: "uniform"
    amplitude_min: -1.0
    amplitude_max: 1.0
    periodic: true
    min_separation: 0.0
```

---

## `signal_model.type`

Supported values:

```text
bandlimited_random
sparse_innovation
mixed
```

### `bandlimited_random`

Original manuscript-style random signal filtered by `signal.W`.

### `sparse_innovation`

FRI-friendly signal model:

```text
x(t) = sum_k a_k phi(t - tau_k)
```

### `mixed`

Optional future model combining bandlimited background and sparse innovations.

---

## `signal_model.sparse_innovation.K`

Number of innovations per period/sensor.

---

## `signal_model.sparse_innovation.kernel`

Supported:

```text
gaussian
sinc
triangular
nearest
```

---

## `signal_model.sparse_innovation.kernel_sigma`

Kernel width / sigma in seconds.

---

## `signal_model.sparse_innovation.amplitude_distribution`

Supported:

```text
uniform
gaussian
```

---

## `signal_model.sparse_innovation.amplitude_min`

Minimum amplitude for uniform innovation amplitudes.

---

## `signal_model.sparse_innovation.amplitude_max`

Maximum amplitude for uniform innovation amplitudes.

---

## `signal_model.sparse_innovation.periodic`

Whether innovation locations are treated periodically.

---

## `signal_model.sparse_innovation.min_separation`

Minimum separation between innovations in seconds.

---

# DC Handling

Section:

```yaml
dc:
```

Typical block:

```yaml
dc:
  enabled: false
```

If:

```yaml
enabled: false
```

then the signal mean/DC component is removed before the RbCP chain.

If:

```yaml
enabled: true
```

then the DC component is preserved.

---

# Global Quantization Policy

Section:

```yaml
quantization:
```

Typical block:

```yaml
quantization:
  force_power_of_two: false
  rounding_mode: "floor"
```

---

## `quantization.force_power_of_two`

If false:

```text
M is a free integer.
```

If true:

```text
M is constrained to a power of two.
```

---

## `quantization.rounding_mode`

Supported values:

```text
floor
ceil
round
```

Used when converting continuous resolution estimates into integer values.

---

# Benchmark Settings

Section:

```yaml
benchmark:
```

Purpose: legacy/high-level benchmark settings used by older figure pipelines.

Newer acquisition-layer configs should prefer:

```yaml
acquisition:
  nyquist:
```

Typical block:

```yaml
benchmark:
  enabled: false
  sampling_rate: 10.4
  effective_rate_factor: 1.0
  W: 10.4
  use_per_sensor_bandwidth: true
```

---

## `benchmark.enabled`

Whether benchmark branches are enabled.

---

## `benchmark.sampling_rate`

Sampling rate used by the benchmark / Nyquist branch.

---

## `benchmark.effective_rate_factor`

Multiplier applied to sampling rate when computing feasible communication resolution.

---

## `benchmark.W`

Benchmark-specific bandwidth used by some legacy Figure-2 mappings.

---

## `benchmark.use_per_sensor_bandwidth`

Descriptive switch indicating whether benchmark calculations use per-sensor bandwidth.

---

# Acquisition Methods

Section:

```yaml
acquisition:
```

Purpose: group acquisition-style baselines.

Current intended modules:

```text
sfc/core/acquisition/nyquist.py
sfc/core/acquisition/cs.py
sfc/core/acquisition/sod.py
sfc/core/acquisition/fri.py
```

---

## Nyquist Acquisition

Section:

```yaml
acquisition:
  nyquist:
```

Typical block:

```yaml
acquisition:
  nyquist:
    enabled: false
    sampling_rate: 10.4
    effective_rate_factor: 1.0
    quantize: true
    peak2peak: "sample"
    periodic_replicas: 10
```

### `acquisition.nyquist.enabled`

Whether Nyquist acquisition is enabled.

### `acquisition.nyquist.sampling_rate`

Sampling rate used by Nyquist.

### `acquisition.nyquist.effective_rate_factor`

Optional multiplier applied to sampling rate for budget computations.

### `acquisition.nyquist.quantize`

Whether sampled values are quantized.

### `acquisition.nyquist.peak2peak`

Quantizer dynamic range.

Supported:

```text
sample
numeric value
```

### `acquisition.nyquist.periodic_replicas`

Number of periodic replicas used in sinc-style reconstruction.

---

## Compressive Sensing / CS Acquisition

Section:

```yaml
acquisition:
  cs:
```

Typical block:

```yaml
acquisition:
  cs:
    enabled: false
    basis: "dct"
    measurement_matrix: "gaussian"
    measurements: null
    measurement_factor: 4.0
    sparsity: null
    reconstruction: "omp"
    max_iter: 200
    tolerance: 1.0e-6
    quantize: true
    budget_bits: null
    bit_error_rate: 0.0
    seed: null
```

### `acquisition.cs.enabled`

Whether CS branch is enabled.

### `acquisition.cs.basis`

Sparse basis / dictionary.

Typical values:

```text
dct
dft
identity
fourier
```

### `acquisition.cs.measurement_matrix`

Measurement matrix / sensing operator.

Typical values:

```text
gaussian
bernoulli
partial_fourier
random_projection
```

### `acquisition.cs.measurements`

Number of CS measurements.

If null, a pipeline may derive this from bit budget or capacity.

### `acquisition.cs.measurement_factor`

Optional factor for rules such as:

```text
measurements = ceil(measurement_factor * sparsity * log(N_grid / sparsity))
```

### `acquisition.cs.sparsity`

Assumed sparsity level.

### `acquisition.cs.reconstruction`

Reconstruction algorithm.

Typical values:

```text
omp
basis_pursuit
lasso
iht
```

### `acquisition.cs.max_iter`

Maximum iterations for iterative solvers.

### `acquisition.cs.tolerance`

Solver tolerance.

### `acquisition.cs.quantize`

Whether CS measurements or coefficients are quantized.

### `acquisition.cs.budget_bits`

Optional total bit budget per sensor/period.

### `acquisition.cs.bit_error_rate`

Digital payload bit error rate.

### `acquisition.cs.seed`

Random seed for sensing matrix generation.

---

## PPM Baseline

Section:

```yaml
acquisition:
  ppm:
```

Typical block:

```yaml
acquisition:
  ppm:
    enabled: false
    M_ppm: null
    fs_msg: null
    force_power_of_two: false
    rounding_mode: "floor"
    budget_bits: null
    bit_error_rate: 0.0
```

### `acquisition.ppm.enabled`

Whether PPM branch is enabled.

### `acquisition.ppm.M_ppm`

PPM slots / message alphabet size.

### `acquisition.ppm.fs_msg`

PPM message sampling frequency.

### `acquisition.ppm.force_power_of_two`

Whether `M_ppm` must be a power of two.

### `acquisition.ppm.rounding_mode`

Rounding mode for derived `M_ppm`.

Supported:

```text
floor
ceil
round
```

### `acquisition.ppm.budget_bits`

Optional bit budget per sensor/period.

### `acquisition.ppm.bit_error_rate`

Bit error rate for decoded PPM messages.

---

## Send-on-Delta / SoD Acquisition

Section:

```yaml
acquisition:
  sod:
```

Typical block:

```yaml
acquisition:
  sod:
    enabled: false
    delta: 0.1
    max_events: null
    polarity: "signed"
    quantize: true
    bits_time: 10
    bits_amplitude: 10
    budget_bits: null
    bit_error_rate: 0.0
```

### `acquisition.sod.enabled`

Whether SoD branch is enabled.

### `acquisition.sod.delta`

Send-on-delta threshold.

### `acquisition.sod.max_events`

Maximum number of events per period/sensor.

### `acquisition.sod.polarity`

Typical values:

```text
signed
absolute
```

### `acquisition.sod.quantize`

Whether event times and amplitudes are quantized.

### `acquisition.sod.bits_time`

Bits used for event time/location.

### `acquisition.sod.bits_amplitude`

Bits used for event amplitude.

### `acquisition.sod.budget_bits`

Optional total bit budget per sensor/period.

### `acquisition.sod.bit_error_rate`

Digital payload bit error rate.

---

## Finite Rate of Innovation / FRI-Inspired Baseline

Section:

```yaml
acquisition:
  fri:
```

Typical block:

```yaml
acquisition:
  fri:
    enabled: false
    K: 3
    min_separation: 0.0
    kernel: "gaussian"
    kernel_sigma: null
    sinc_bandwidth: null
    periodic: true
    quantize: true
    budget_bits: null
    bits_location: 10
    bits_amplitude: 10
    amplitude_range: "sample"
    peak_to_peak: null
    bit_error_rate: 0.0
    seed: null
```

### `acquisition.fri.enabled`

Whether FRI-inspired branch is enabled.

---

### `acquisition.fri.K`

Number of innovations per period/sensor.

The FRI-inspired model reconstructs:

```text
x_hat(t) = sum_k a_k phi(t - tau_k)
```

---

### `acquisition.fri.min_separation`

Minimum separation between selected innovations in seconds.

If zero, no minimum separation is enforced.

---

### `acquisition.fri.kernel`

Supported reconstruction kernels:

```text
gaussian
sinc
triangular
nearest
```

---

### `acquisition.fri.kernel_sigma`

Gaussian sigma or triangular width.

If null, `FRIAcquisition` defaults to a value based on `Tt`.

---

### `acquisition.fri.sinc_bandwidth`

Bandwidth parameter for sinc kernel.

---

### `acquisition.fri.periodic`

Whether distances are computed periodically over one signal period.

---

### `acquisition.fri.quantize`

Whether innovation locations and amplitudes are quantized.

---

### `acquisition.fri.budget_bits`

Optional total bit budget per signal/period/sensor.

If provided, `FRIAcquisition` splits this budget across `K` innovations and then between location and amplitude.

---

### `acquisition.fri.bits_location`

Bits used for each innovation location when `budget_bits` is null.

---

### `acquisition.fri.bits_amplitude`

Bits used for each innovation amplitude when `budget_bits` is null.

---

### `acquisition.fri.amplitude_range`

Supported values:

```text
sample
peak_to_peak
[min, max]
```

---

### `acquisition.fri.peak_to_peak`

Peak-to-peak range used when:

```yaml
amplitude_range: "peak_to_peak"
```

If null, the FRI module may fall back to `signal.peak_to_peak`.

---

### `acquisition.fri.bit_error_rate`

Bit error rate applied to quantized innovation indices.

---

### `acquisition.fri.seed`

Optional FRI-local random seed.

If null, the module may fall back to:

```yaml
reproducibility:
  seed: ...
```

or:

```yaml
monte_carlo:
  seed: ...
```

---

# RbCP Settings

Section:

```yaml
rbcp:
```

Typical block:

```yaml
rbcp:
  enabled: false
```

`rbcp.enabled` is a descriptive method-level flag.

Pipeline execution is usually controlled by:

```yaml
mode:
  run_rbcp: true
```

---

# RbCP_time Settings

Section:

```yaml
rbcp_time:
```

Typical block:

```yaml
rbcp_time:
  enabled: false
  use_error_free_time_model: true
```

RbCP_time generally represents:

```text
signal -> ta/tb -> events -> ta/tb -> signal
```

without physical channel errors.

---

# Channel / PHY / SFC Detection Settings

Section:

```yaml
channel:
```

Purpose: configure SFC physical channel, local detection, and global candidate selection.

Typical block:

```yaml
channel:
  type: "awgn"
  collision_mode: "sum"
  detection_mode: "threshold"
  score_threshold: 4
  threshold_factor: 0.5
  threshold: null
  sensor_x_event: null
  candidate_selection: "global_frame_fit"

  global:
    allow_empty: true
    strategy: "local_only"
    restarts: 100
    max_iter: 80
    seed: 47
    observed_mode: "abs"
    event_penalty: 0.0
    residual_threshold: 0.0
```

---

## `channel.type`

Supported values:

```text
awgn
clean
```

---

## `channel.collision_mode`

Supported values:

```text
sum
binary
```

Recommended for physical consistency:

```yaml
collision_mode: "sum"
```

---

## `channel.detection_mode`

Supported values:

```text
strict
loose
threshold
```

---

## `channel.score_threshold`

Candidate map score threshold.

If null, detector may default to:

```text
score_threshold = L
```

---

## `channel.threshold_factor`

Used to derive physical chip threshold:

```text
threshold = threshold_factor * sqrt(E_chip)
```

where:

```text
E_chip = P*tau/(2*N*L)
```

---

## `channel.threshold`

Optional explicit detector threshold.

If provided, overrides `threshold_factor`.

---

## `channel.sensor_x_event`

Optional explicit sensor-event association matrix.

Expected shape when provided:

```text
(S, 2*N*S)
```

Most pipelines build this internally.

---

## `channel.candidate_selection`

Supported values:

```text
global_frame_fit
none
local
local_only
```

Recommended:

```yaml
candidate_selection: "global_frame_fit"
```

---

## `channel.global.allow_empty`

Whether the global detector may choose the empty decision for an event ID.

The manuscript allows sparse transmission because some coefficients may be absent.

Recommended:

```yaml
allow_empty: true
```

---

## `channel.global.strategy`

Supported values:

```text
local_only
event_penalty
residual_rule
```

---

## `channel.global.restarts`

Number of random restarts for global coordinate descent.

---

## `channel.global.max_iter`

Maximum coordinate-descent iterations per restart.

---

## `channel.global.seed`

Seed for global detector random restarts.

---

## `channel.global.observed_mode`

Supported values:

```text
abs
real
binary
```

Recommended for AWGN:

```yaml
observed_mode: "abs"
```

---

## `channel.global.event_penalty`

Penalty used when:

```yaml
strategy: "event_penalty"
```

Objective:

```text
||Y_obs - Y_hat||^2 + event_penalty * number_of_active_events
```

---

## `channel.global.residual_threshold`

Residual-improvement threshold used when:

```yaml
strategy: "residual_rule"
```

---

# Semantic-Based Error Detection / SED

Section:

```yaml
sed:
```

Typical block:

```yaml
sed:
  enabled: false
  discard_invalid_periods: true
  mode: "sparse_pairs"
  event_id_ordering: "ta_block_then_tb_block"
  throughput_mode: "valid_periods_times_S_over_tau"
```

---

## `sed.enabled`

Whether SED is enabled.

Some pipelines also use:

```yaml
mode:
  semantic_error_detection: true
```

---

## `sed.discard_invalid_periods`

If true, invalid periods are discarded before reconstruction / MSE aggregation.

---

## `sed.mode`

Current supported value:

```text
sparse_pairs
```

---

## `sed.event_id_ordering`

Current supported value:

```text
ta_block_then_tb_block
```

For each sensor `s`:

```text
[ta events for N harmonics][tb events for N harmonics]
```

Therefore:

```text
ta_id = 2*s*N + n_idx
tb_id = 2*s*N + N + n_idx
```

---

## `sed.throughput_mode`

Current Figure-7-style mode:

```text
valid_periods_times_S_over_tau
```

Throughput convention:

```text
throughput = S * num_valid_periods / (num_periods * tau)
```

---

# Duplicate RX Settings

Section:

```yaml
duplicate_rx:
```

Typical block:

```yaml
duplicate_rx:
  interactions: 200
```

Used by duplicate-reception experiments.

---

# Sweep Parameters

Section:

```yaml
sweep:
```

Typical block:

```yaml
sweep:
  B:
    start: 1000
    stop: 10001
    step: 1000
    include_stop: true

  n:
    start: 1
    stop: 11
    step: 1

  n_values:
    - 2
    - 3
    - 5

  m_rbcp_values:
    - 2
    - 4
    - 8
    - 16

  m_rbcp:
    start: 2
    stop: 33
    step: 1
    include_stop: true
```

---

## `sweep.B`

Sweep over total bandwidth.

---

## `sweep.B.include_stop`

If true, the runner/pipeline may include the stop value.

---

## `sweep.n`

Sweep over harmonic count `N`.

---

## `sweep.n_values`

Explicit list of `N` values.

---

## `sweep.m_rbcp_values`

Explicit list of `M_RbCP` values.

---

## `sweep.m_rbcp`

Sweep over `M_RbCP`.

---

# Monte Carlo

Section:

```yaml
monte_carlo:
```

Typical block:

```yaml
monte_carlo:
  seed: 47
  interactions: 200
```

---

## `monte_carlo.seed`

Random seed.

---

## `monte_carlo.interactions`

Number of Monte Carlo trials.

---

# Metrics

Section:

```yaml
metrics:
```

Typical block:

```yaml
metrics:
  mse: true
  throughput: false
  event_rate: false
  acquisition_rate: false
  bit_budget: false
  resource_diagnostics: true

  aggregation:
    mean: true
    std: false
    median: false
    quantiles: []
```

---

## `metrics.mse`

Whether MSE is computed/reported.

---

## `metrics.throughput`

Whether throughput is computed/reported.

---

## `metrics.event_rate`

Useful for SoD, FRI, and event-based comparisons.

---

## `metrics.acquisition_rate`

Useful for acquisition method comparisons.

---

## `metrics.bit_budget`

Whether bit-budget diagnostics are reported.

Examples:

```text
bits_per_period
bits_per_sensor
bits_per_method
```

---

## `metrics.resource_diagnostics`

Whether method-specific resource diagnostics are reported.

Examples:

```text
M_benchmark
M_rbcp
M_time
cs_measurements
ppm_fs_msg
fri_K
sod_num_events
```

---

## `metrics.aggregation`

Controls reporting of summary statistics.

---

# Output

Section:

```yaml
output:
```

Typical block:

```yaml
output:
  base_dir: "data/results"
  figure_dir: "template_experiment"

  save_dat: true
  save_plot: true
  save_diagnostics_plot: true
  save_metadata: true
  save_config_copy: true

  formats:
    plot:
      - "png"
      - "pdf"

    figure:
      - "png"
      - "pdf"

    data:
      - "dat"
```

---

## `output.base_dir`

Base results directory.

---

## `output.figure_dir`

Experiment-specific subdirectory.

---

## Save Controls

```yaml
save_dat: true
save_plot: true
save_diagnostics_plot: true
save_metadata: true
save_config_copy: true
```

---

## Output Formats

Plot/figure formats:

```yaml
plot:
  - "png"
  - "pdf"
```

Data formats:

```yaml
data:
  - "dat"
```

Some runners also support:

```yaml
data:
  - "dat"
  - "csv"
```

---

# Data Format

Section:

```yaml
data_format:
```

Typical block:

```yaml
data_format:
  columns: []
  delimiter: "\t"
```

---

## `data_format.columns`

Experiment-specific column list.

Usually left empty in the global catalog.

---

## `data_format.delimiter`

Delimiter used when saving data tables.

Common values:

```text
\t
,
```

---

# Plot Settings

Section:

```yaml
plot:
```

Typical block:

```yaml
plot:
  x_axis: "B"
  y_axis: "MSE"
  x_scale: "linear"
  y_scale: "log"
  show_grid: true
  legend: true

  label_map:
    benchmark: "Benchmark"
    nyquist: "Nyquist"
    rbcp: "RbCP"
    rbcp_time: "RbCP_time"
    sfc: "SFC"
    sfc_sed: "SFC + SED"

    cs: "CS"
    ppm: "PPM"
    sod: "SoD"
    fri: "FRI-inspired"

  panel_layout: "vertical"
```

---

## `plot.x_axis`

Default x-axis label.

---

## `plot.y_axis`

Default y-axis label.

---

## `plot.x_scale`

Supported:

```text
linear
log
```

---

## `plot.y_scale`

Supported:

```text
linear
log
```

---

## `plot.show_grid`

Whether to show grid lines.

---

## `plot.legend`

Whether to show legend.

---

## `plot.label_map`

Dictionary of method labels used by runners.

Common labels:

```yaml
label_map:
  benchmark: "Benchmark"
  nyquist: "Nyquist"
  rbcp: "RbCP"
  rbcp_time: "RbCP_time"
  sfc: "SFC"
  sfc_sed: "SFC + SED"
  cs: "CS"
  ppm: "PPM"
  sod: "SoD"
  fri: "FRI-inspired"
```

---

## Figure-7-Style Panels

```yaml
plot:
  mse_panel:
    x_axis: "B"
    y_axis: "MSE"
    x_scale: "linear"
    y_scale: "log"
    title: "MSE versus B"
    show_grid: true
    legend: true

  throughput_panel:
    x_axis: "B"
    y_axis: "Throughput"
    x_scale: "linear"
    y_scale: "linear"
    title: "Throughput versus B"
    show_grid: true
    legend: true
```

---

# Comparison Settings

Section:

```yaml
comparison:
```

Purpose: configure method comparisons and truncation policies.

---

## Truncation Comparison

Typical block:

```yaml
comparison:
  benchmark:
    free_integer:
      force_power_of_two: false
      rounding_mode: "floor"

    power_of_two:
      force_power_of_two: true
      rounding_mode: "floor"

  rbcp:
    free_integer:
      force_power_of_two: false
      rounding_mode: "floor"

    power_of_two:
      force_power_of_two: true
      rounding_mode: "floor"
```

Used by Benchmark vs RbCP truncation-comparison experiments.

---

## General Method Comparison

Typical block:

```yaml
comparison:
  methods:
    enabled:
      - "benchmark"
      - "rbcp"
      - "rbcp_time"
      - "sfc"

    use_common_bit_budget: true
    bit_budget_source: "capacity"
    fixed_budget_bits: null
    target_ber: 0.0
    sfc_error_model: "native_physical_channel"
    signal_model: "bandlimited_random"
```

---

### `comparison.methods.enabled`

Ordered list of methods included in generic comparison pipelines.

Supported typical values:

```text
benchmark
nyquist
cs
ppm
sod
fri
rbcp
rbcp_time
sfc
sfc_sed
```

---

### `comparison.methods.use_common_bit_budget`

If true, digital/acquisition baselines should use the same capacity-derived bit budget per sensor/period when supported by the pipeline.

---

### `comparison.methods.bit_budget_source`

Supported values:

```text
capacity
fixed
method_specific
```

#### `capacity`

Derive bits from:

```text
B_s * log2(1 + SNR_s) * tau
```

#### `fixed`

Use:

```yaml
comparison:
  methods:
    fixed_budget_bits: ...
```

#### `method_specific`

Let each method derive its own native budget.

---

### `comparison.methods.fixed_budget_bits`

Used only when:

```yaml
bit_budget_source: "fixed"
```

---

### `comparison.methods.target_ber`

Equivalent post-demodulation BER for digital baselines.

Applies naturally to:

```text
CS
PPM
SoD
FRI
Benchmark/Nyquist
RbCP
```

SFC may instead use its native physical AWGN event-channel model.

---

### `comparison.methods.sfc_error_model`

Supported values:

```text
native_physical_channel
equivalent_ber
```

Recommended:

```yaml
sfc_error_model: "native_physical_channel"
```

---

### `comparison.methods.signal_model`

Supported values:

```text
bandlimited_random
sparse_innovation
mixed
```

Useful for reviewer-response experiments comparing RbCP against CS and FRI.

---

# Reproducibility

Section:

```yaml
reproducibility:
```

Typical block:

```yaml
reproducibility:
  save_git_commit: true
  save_timestamp: true
  seed: 47
```

---

## `reproducibility.save_git_commit`

Whether runners save the current git commit hash in metadata.

---

## `reproducibility.save_timestamp`

Whether runners save timestamp metadata.

---

## `reproducibility.seed`

Generic seed mirror for modules expecting:

```yaml
reproducibility:
  seed: ...
```

FRI and SFC global detection may fall back to this value when their local seed is null.

---

# Practical Examples

## Minimal RbCP MSE-vs-B Style Config

```yaml
figure:
  id: "rbcp_mse_vs_B"
  title: "RbCP MSE versus B"

mode:
  run_benchmark: true
  run_rbcp: true
  run_rbcp_time: true
  run_sfc: true

system:
  S: 8
  P: 1.0
  B: 10000.0
  R: 12
  L: 4
  SNR_dB: -16.0
  N0: 0.007
  bandwidth_allocation: null

signal:
  W: 10.4
  tau: 1.0
  Tt: 0.01
  distribution: "uniform"
  peak_to_peak: 2.0
  normalize_dft: true
  normalization_target: 3.9
  threshold_harmonics: 0.001

monte_carlo:
  seed: 47
  interactions: 100

output:
  base_dir: "data/results"
  figure_dir: "rbcp_mse_vs_B"
  save_dat: true
  save_plot: true
  save_metadata: true
  save_config_copy: true
  formats:
    plot:
      - "png"
      - "pdf"
    data:
      - "dat"
```

---

## Reviewer-Response Comparison with CS and FRI

```yaml
figure:
  id: "fri_cs_rbcp_comparison_vs_B"
  title: "RbCP comparison with CS and FRI-inspired acquisition"

mode:
  run_benchmark: true
  run_cs: true
  run_fri: true
  run_rbcp: true
  run_sfc: true

comparison:
  methods:
    enabled:
      - "benchmark"
      - "cs"
      - "fri"
      - "rbcp"
      - "sfc"

    use_common_bit_budget: true
    bit_budget_source: "capacity"
    target_ber: 1.0e-3
    sfc_error_model: "native_physical_channel"
    signal_model: "bandlimited_random"

acquisition:
  cs:
    enabled: true
    basis: "dct"
    measurement_matrix: "gaussian"
    reconstruction: "omp"
    quantize: true
    bit_error_rate: 1.0e-3

  fri:
    enabled: true
    K: 3
    kernel: "gaussian"
    kernel_sigma: 0.01
    quantize: true
    bits_location: 10
    bits_amplitude: 10
    bit_error_rate: 1.0e-3
```

---

## FRI-Friendly Sparse-Innovation Signal

```yaml
signal_model:
  type: "sparse_innovation"

  sparse_innovation:
    K: 3
    kernel: "gaussian"
    kernel_sigma: 0.01
    amplitude_distribution: "uniform"
    amplitude_min: -1.0
    amplitude_max: 1.0
    periodic: true
    min_separation: 0.02

acquisition:
  fri:
    enabled: true
    K: 3
    min_separation: 0.02
    kernel: "gaussian"
    kernel_sigma: 0.01
    quantize: true
    bits_location: 10
    bits_amplitude: 10
    amplitude_range: "sample"
```

---

# Recommended Documentation Language for FRI

When describing the FRI baseline in the paper or response letter, use:

```text
FRI-inspired sparse-innovation acquisition baseline
```

rather than:

```text
full FRI annihilating-filter implementation
```

This is more precise because the current simulation baseline estimates and quantizes sparse innovations explicitly, while classical FRI reconstruction is a model-specific method based on finite innovation structure.

---

# Final Reminder

For execution, always use a figure-specific YAML under:

```text
experiments/configs/figures/
```

For discovery and documentation, use:

```text
experiments/configs/configs.yaml
```

and this Markdown reference.
``

```text
experiments/configs/configs.yaml
```

but in Markdown form for easier reading and navigation.

> **Important**
>
> - `experiments/configs/configs.yaml` is the **master YAML catalog of keys**.
> - `experiments/configs/figures/*.yaml` are the **actual runnable configs**.
> - This document is explanatory. The YAML catalog is the authoritative key list.
> - Not every pipeline reads every key.
> - Experiment-specific YAML files should include only the keys needed by that experiment.

---

## Quick Rule

- Use **`configs.yaml`** when you want to discover what the system can configure.
- Use **`experiments/configs/figures/*.yaml`** when you want to run a specific experiment.
