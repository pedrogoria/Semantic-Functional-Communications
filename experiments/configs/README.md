# Configuration Tree Guide

This directory contains **three different kinds of configuration/reference files**:

1. **`configs.yaml`** → the **master reference catalog** of configurable keys.
2. **`CONFIG_REFERENCE.md`** → the **human-readable companion reference** for `configs.yaml`.
3. **`figures/*.yaml`** → **runnable experiment configurations**.

---

## 1) What `configs.yaml` is

`experiments/configs/configs.yaml` is **not** intended to be the minimal config for a single run.

Instead, `configs.yaml` is a **long-lived reference file** that documents:

- major configuration sections supported by the core and pipelines;
- acquisition methods and comparison baselines;
- the meaning of each key;
- expected type;
- common/allowed values;
- practical ranges or examples;
- legacy vs current conventions.

Think of it as the **memory of the system**: after a long time, it should still be easy to inspect that file and answer:

- *What can be configured?*
- *Which key controls this behavior?*
- *Is this parameter still current or only legacy?*
- *What values are typical?*
- *Which acquisition/comparison methods are currently documented?*

Current method families documented in the catalog include:

- Benchmark / Nyquist
- RbCP
- RbCP_time
- SFC
- SFC + SED
- Compressive Sensing / CS
- PPM
- Send-on-Delta / SoD
- FRI-inspired sparse-innovation acquisition

---

## 2) What `CONFIG_REFERENCE.md` is

`experiments/configs/CONFIG_REFERENCE.md` is the **Markdown explanation** of the master catalog.

Use it when you want a readable, navigable description of:

- the purpose of each section;
- current naming conventions;
- physical-model conventions;
- acquisition-method conventions;
- reviewer-response comparison settings;
- example YAML snippets.

The authoritative key list remains:

```text
experiments/configs/configs.yaml
```

The Markdown document is explanatory and should be kept synchronized with the YAML catalog.

---

## 3) What `figures/*.yaml` are

Files under:

```text
experiments/configs/figures/
```

are the **actual experiment configs** used by runners and pipelines.

Examples:

- `experiments/configs/figures/fig01.yaml`
- `experiments/configs/figures/fig02.yaml`
- `experiments/configs/figures/rbcp_mse_vs_B.yaml`
- `experiments/configs/figures/rbcp_mse_vs_B_fixed_power.yaml`
- `experiments/configs/figures/rbcp_benchmark_truncation_mse_vs_B.yaml`
- `experiments/configs/figures/rbcp_signal_representation.yaml`
- `experiments/configs/figures/sfc_duplicate_rx.yaml`
- `experiments/configs/figures/sfc_mse_throughput_vs_B.yaml`
- `experiments/configs/figures/fair_methods_comparison_vs_B.yaml`
- future reviewer-response configs such as:
  - `experiments/configs/figures/fri_cs_rbcp_comparison_vs_B.yaml`
  - `experiments/configs/figures/fri_friendly_comparison_vs_B.yaml`

These files should stay focused and only override the parameters needed by the experiment.

---

## 4) Recommended workflow

### When adding a new experiment

1. Create a new file under:

   ```text
   experiments/configs/figures/
   ```

2. Keep the file experiment-specific and as small as practical.

3. If the experiment introduces **new configurable keys**, also update:

   - `experiments/configs/configs.yaml`
   - `experiments/configs/CONFIG_REFERENCE.md`
   - `experiments/configs/README.md`, if the tree usage or conventions changed.

4. If the experiment introduces a new runner, keep the runner aligned with the project output convention:

   ```text
   output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/
   ```

5. If the experiment introduces a new method or baseline, document whether the method belongs to:

   - acquisition;
   - communication;
   - channel;
   - reconstruction;
   - pipeline-only logic.

---

### When changing the core configuration surface

If the `sfc/core` layer begins to accept a new key, a new value range, or a new mode, update the reference files even if no immediate figure uses the new option.

The configuration reference files should be updated when one of these changes happens:

- a new config key is introduced;
- a key is deprecated;
- a new valid option/value is added;
- a legacy alias is kept for compatibility;
- a default behavior changes;
- a core theoretical convention changes;
- a new acquisition module is added;
- a new physical-channel convention is added;
- a new method-comparison policy is introduced.

---

## 5) Current design principle

The tree is intentionally split into:

- **reference / documentation**
- **execution configs**

This keeps figure YAMLs cleaner while preserving a single place where all supported configuration knobs are documented.

The intended relationship is:

```text
configs.yaml
    authoritative catalog of keys

CONFIG_REFERENCE.md
    readable explanation of configs.yaml

figures/*.yaml
    runnable experiment-specific configs
```

---

## 6) Acquisition-method organization

Acquisition-style baselines should live under:

```text
sfc/core/acquisition/
```

Current or expected acquisition modules include:

```text
sfc/core/acquisition/base.py
sfc/core/acquisition/nyquist.py
sfc/core/acquisition/cs.py
sfc/core/acquisition/sod.py
sfc/core/acquisition/fri.py
```

The corresponding YAML section is:

```yaml
acquisition:
  nyquist:
    enabled: false

  cs:
    enabled: false

  ppm:
    enabled: false

  sod:
    enabled: false

  fri:
    enabled: false
```

### 6.1 Nyquist / Benchmark

Nyquist sampling is treated as an acquisition baseline.

Legacy/high-level benchmark settings may still live under:

```yaml
benchmark:
```

Newer acquisition-layer settings should prefer:

```yaml
acquisition:
  nyquist:
```

This preserves backward compatibility while moving toward a cleaner acquisition hierarchy.

---

### 6.2 CS

Compressive Sensing settings are grouped under:

```yaml
acquisition:
  cs:
```

Typical keys include:

- `basis`
- `measurement_matrix`
- `measurements`
- `measurement_factor`
- `sparsity`
- `reconstruction`
- `max_iter`
- `tolerance`
- `quantize`
- `budget_bits`
- `bit_error_rate`
- `seed`

CS should be compared carefully because its performance depends strongly on sparsity/compressibility assumptions and reconstruction algorithms.

---

### 6.3 PPM

PPM settings are grouped under:

```yaml
acquisition:
  ppm:
```

Typical keys include:

- `M_ppm`
- `fs_msg`
- `force_power_of_two`
- `rounding_mode`
- `budget_bits`
- `bit_error_rate`

PPM may be used as a communication/acquisition baseline in fair-method comparisons.

---

### 6.4 SoD

Send-on-Delta settings are grouped under:

```yaml
acquisition:
  sod:
```

Typical keys include:

- `delta`
- `max_events`
- `polarity`
- `quantize`
- `bits_time`
- `bits_amplitude`
- `budget_bits`
- `bit_error_rate`

SoD is event-driven and should usually report event-rate or acquisition-rate diagnostics.

---

### 6.5 FRI-inspired acquisition

FRI-inspired sparse-innovation acquisition settings are grouped under:

```yaml
acquisition:
  fri:
```

Typical keys include:

- `K`
- `min_separation`
- `kernel`
- `kernel_sigma`
- `sinc_bandwidth`
- `periodic`
- `quantize`
- `budget_bits`
- `bits_location`
- `bits_amplitude`
- `amplitude_range`
- `peak_to_peak`
- `bit_error_rate`
- `seed`

Use the wording:

```text
FRI-inspired sparse-innovation acquisition baseline
```

rather than:

```text
full FRI annihilating-filter implementation
```

unless the actual annihilating-filter FRI reconstruction is implemented.

---

## 7) Reviewer-response comparisons

The catalog now supports method-comparison settings intended for reviewer-response experiments involving:

- RbCP
- SFC
- Benchmark / Nyquist
- CS
- FRI-inspired acquisition
- PPM
- SoD

The relevant section is:

```yaml
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
    fixed_budget_bits: null
    target_ber: 0.0
    sfc_error_model: "native_physical_channel"
    signal_model: "bandlimited_random"
```

### 7.1 Common bit budget

When supported by a pipeline, digital/acquisition baselines should use a common capacity-derived bit budget:

```text
bits_per_sensor_per_period = floor(B_s * log2(1 + SNR_s) * tau)
```

This is intended to make comparisons more defensible under equal communication resources.

---

### 7.2 BER convention

Digital baselines may use an equivalent post-demodulation BER:

```yaml
comparison:
  methods:
    target_ber: 1.0e-3
```

This can be applied to quantized payloads for methods such as:

- Benchmark / Nyquist
- CS
- PPM
- SoD
- FRI-inspired
- RbCP

SFC may instead use its native physical event-channel model:

```yaml
comparison:
  methods:
    sfc_error_model: "native_physical_channel"
```

---

### 7.3 Signal-model fairness

Generic comparison pipelines may use:

```yaml
signal_model:
  type: "bandlimited_random"
```

or:

```yaml
signal_model:
  type: "sparse_innovation"
```

Use `bandlimited_random` for original manuscript-style random bandlimited signals.

Use `sparse_innovation` for FRI-friendly experiments where the signal is explicitly generated as:

```text
x(t) = sum_k a_k phi(t - tau_k)
```

This distinction is important because FRI is model-matched to sparse-innovation signals and is not a universal replacement for Nyquist, RbCP, or SFC.

---

## 8) Important current SFC conventions

### 8.1 Theorem-consistent SFC maps

The SFC map generator follows the theorem-consistent orthogonal-vector construction.

The `R` resources are partitioned into `L` disjoint row-resource groups:

```text
D_1, D_2, ..., D_L
```

Each row `l` can only activate a resource from `D_l`.

The number of theorem-valid maps is:

```text
|D_1| * |D_2| * ... * |D_L|
```

The number of event IDs required by the standard SFC Fourier/phase representation is:

```text
num_event_ids = 2 * N * S
```

Therefore, valid configurations should satisfy:

```text
product_l |D_l| >= 2 * N * S
```

Example:

```text
S = 12
N = 5
L = 4

required maps = 2 * 5 * 12 = 120

R = 12 -> partition [3,3,3,3] -> 81 maps  -> invalid
R = 16 -> partition [4,4,4,4] -> 256 maps -> valid
```

If this condition is not met, `SFCChannel` may raise an assertion error.

---

### 8.2 SFC physical-channel convention

For SFC physical simulation, the current convention is:

```text
y = sqrt(E_chip) * superposed + n
n ~ CN(0, N0)
```

where:

```text
E_chip = P * tau / (2 * N * L)
```

The SFC physical channel uses `N0` directly at the matched-filter / resource-output level.

It does **not** generate the SFC noise variance as:

```text
B * N0
```

or:

```text
B_s * N0
```

Those expressions are useful for total-band or per-sensor SNR diagnostics, but not for the native SFC resource-output AWGN model.

---

### 8.3 Local and global SFC detection

Local SFC detection is configured through:

```yaml
channel:
  detection_mode: "threshold"
  score_threshold: 4
  threshold_factor: 0.5
```

Global candidate selection is configured through:

```yaml
channel:
  candidate_selection: "global_frame_fit"

  global:
    allow_empty: true
    strategy: "local_only"
    restarts: 100
    max_iter: 80
    observed_mode: "abs"
```

Recommended defaults for experiments are usually:

```yaml
channel:
  collision_mode: "sum"
  detection_mode: "threshold"
  candidate_selection: "global_frame_fit"

  global:
    allow_empty: true
    observed_mode: "abs"
```

---

## 9) Important current SED conventions

Semantic-based error detection is configured through:

```yaml
mode:
  semantic_error_detection: true
```

and/or:

```yaml
sed:
  enabled: true
```

depending on the pipeline.

The current theoretical SED mode is:

```yaml
sed:
  mode: "sparse_pairs"
```

The event-ID ordering is:

```yaml
sed:
  event_id_ordering: "ta_block_then_tb_block"
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

SED marks a period invalid if:

```text
count(event_id) > 1
```

or if:

```text
count(ta_id) = 1 and count(tb_id) = 0
```

or if:

```text
count(ta_id) = 0 and count(tb_id) = 1
```

SED accepts absent sparse pairs:

```text
count(ta_id) = 0 and count(tb_id) = 0
```

This is important because Fourier coefficients below the harmonic threshold may be omitted.

---

## 10) Important current signal conventions

### 10.1 Signal bandwidth

The source signal should be filtered using:

```yaml
signal:
  W: ...
```

Do not redefine the source filter bandwidth from `N` as:

```text
W_eff = 2 * N / tau
```

---

### 10.2 Signal period

Prefer:

```yaml
signal:
  tau: ...
```

Older configs may still include:

```yaml
signal:
  T: ...
```

If both are present, keep them equal.

---

### 10.3 Harmonic threshold

The harmonic threshold is configured through:

```yaml
signal:
  threshold_harmonics: 0.001
```

When a harmonic coefficient is below threshold, the corresponding `ta` or `tb` can be absent.

This absence is valid when both members of a pair are absent.

---

## 11) Output convention

Most current runners use timestamped output directories:

```text
output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/
```

Example:

```text
data/results/sfc_duplicate_rx/20260605_150000/
```

Typical output config:

```yaml
output:
  base_dir: "data/results"
  figure_dir: "sfc_duplicate_rx"

  save_dat: true
  save_plot: true
  save_diagnostics_plot: true
  save_metadata: true
  save_config_copy: true

  formats:
    plot:
      - "png"
      - "pdf"

    data:
      - "dat"
      - "csv"
```

Runners should usually save:

- data table;
- plot(s);
- metadata JSON;
- copy of the YAML config.

---

## 12) Adding a new method

When adding a new method, prefer this process:

1. Implement the core module in the appropriate package.

   Acquisition methods should usually go under:

   ```text
   sfc/core/acquisition/
   ```

2. Add a corresponding config block under:

   ```yaml
   acquisition:
   ```

   if the method is an acquisition baseline.

3. Add a `mode.run_*` flag if pipelines need branch-level switches.

4. Add labels under:

   ```yaml
   plot:
     label_map:
   ```

5. Add method-selection support under:

   ```yaml
   comparison:
     methods:
       enabled:
   ```

6. Update:

   - `experiments/configs/configs.yaml`
   - `experiments/configs/CONFIG_REFERENCE.md`
   - `experiments/configs/README.md`

7. Add or update figure-specific YAMLs under:

   ```text
   experiments/configs/figures/
   ```

8. Add or update runners under:

   ```text
   scripts/
   ```

---

## 13) Quick validation commands

From the project root:

```bash
python -m compileall sfc scripts
```

To check YAML readability manually:

```bash
python - <<'PY'
from pathlib import Path
import yaml

for path in Path("experiments/configs").rglob("*.yaml"):
    with path.open("r", encoding="utf-8") as f:
        yaml.safe_load(f)
    print("OK:", path)
PY
```

To inspect changed config files:

```bash
git diff experiments/configs
```

---

## 14) Final reminder

Use:

```text
experiments/configs/configs.yaml
```

for the **master catalog**.

Use:

```text
experiments/configs/CONFIG_REFERENCE.md
```

for the **readable reference**.

Use:

```text
experiments/configs/figures/*.yaml
```

for **actual runs**.