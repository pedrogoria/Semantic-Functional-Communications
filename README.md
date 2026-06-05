# Semantic-Functional Communications

Simulation framework for **Semantic-Functional Communications (SFC)**, **RbCP**, **RbCP_time**, compressed sensing baselines, and benchmark communication schemes used in manuscript-style numerical experiments.

The repository provides:

- theoretical helpers for SFC/RbCP relations;
- channel, collision, detection, and semantic-error-detection modules;
- experiment pipelines for manuscript figures;
- Python runners for generating `.dat`, `.csv`, `.png`, `.pdf`, metadata, and config copies;
- YAML-based experiment configuration.

---

## Repository Structure

```text
.
├── experiments/
│   └── configs/
│       ├── configs.yaml              # master configuration catalog
│       ├── CONFIG_REFERENCE.md       # human-readable configuration reference
│       ├── README.md                 # configuration tree guide
│       └── figures/                  # runnable experiment YAMLs
│
├── scripts/                          # runner scripts for figures/experiments
│   ├── run_figure1.py
│   ├── run_figure2.py
│   ├── run_fair_methods_comparison_vs_B.py
│   ├── run_rbcp_mse_vs_B.py
│   ├── run_rbcp_mse_vs_B_fixed_power.py
│   ├── run_rbcp_benchmark_truncation_mse_vs_B.py
│   ├── run_rbcp_signal_representation.py
│   ├── run_sfc_duplicate_rx.py
│   └── run_sfc_mse_throughput_vs_B.py
│
├── sfc/
│   ├── core/
│   │   ├── theory.py                 # centralized theoretical formulas
│   │   ├── system_parameters.py      # derived system-parameter builder
│   │   ├── semantic_error_detection.py
│   │   ├── channel/
│   │   │   ├── SFCChannel.py
│   │   │   ├── collision.py
│   │   │   ├── detection.py
│   │   │   ├── mapping.py
│   │   │   └── physical_channel.py
│   │   └── ...
│   │
│   └── pipelines/                    # scientific experiment pipelines
│
├── tests/                            # debug and validation scripts/configs
├── data/results/                     # generated outputs
├── pyproject.toml
└── README.md
```

---

## Installation

Create and activate a Python environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The project is configured through `pyproject.toml` and currently targets Python `>=3.8`.

Core dependencies include:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `pyyaml`

---

## Quick Start

Most experiments can be run directly from the Python console.

### Fair methods comparison versus bandwidth

```python
from scripts.run_fair_methods_comparison_vs_B import run_fair_methods_comparison_vs_B

df = run_fair_methods_comparison_vs_B()
```

This uses the default config:

```text
experiments/configs/figures/fair_methods_comparison_vs_B.yaml
```

and writes outputs to:

```text
data/results/fair_methods_comparison_vs_B/
```

Inspect results:

```python
df.head()
df.filter(regex="B|mse").to_string(index=False)
```

---

## Running Individual Figures from Python Console

### Figure 1

```python
from scripts.run_figure1 import run_fig01

df1 = run_fig01("experiments/configs/figures/fig01.yaml")
```

### Figure 2

```python
from scripts.run_figure2 import run_fig02

df2 = run_fig02("experiments/configs/figures/fig02.yaml")
```

### RbCP MSE versus bandwidth

```python
from scripts.run_rbcp_mse_vs_B import run_rbcp_mse_vs_B

df_rbcp_B = run_rbcp_mse_vs_B(
    "experiments/configs/figures/rbcp_mse_vs_B.yaml"
)
```

### RbCP MSE versus bandwidth, fixed power

```python
from scripts.run_rbcp_mse_vs_B_fixed_power import run_rbcp_mse_vs_B_fixed_power

df_rbcp_fixed = run_rbcp_mse_vs_B_fixed_power(
    "experiments/configs/figures/rbcp_mse_vs_B_fixed_power.yaml"
)
```

### RbCP / Benchmark truncation comparison

```python
from scripts.run_rbcp_benchmark_truncation_mse_vs_B import (
    run_rbcp_benchmark_truncation_mse_vs_B,
)

df_trunc = run_rbcp_benchmark_truncation_mse_vs_B(
    "experiments/configs/figures/rbcp_benchmark_truncation_mse_vs_B.yaml"
)
```

### RbCP signal representation

```python
from scripts.run_rbcp_signal_representation import run_rbcp_signal_representation

data_repr = run_rbcp_signal_representation(
    "experiments/configs/figures/rbcp_signal_representation.yaml"
)
```

### SFC duplicate reception probability

```python
from scripts.run_sfc_duplicate_rx import run_sfc_duplicate_rx

df_dup = run_sfc_duplicate_rx(
    "experiments/configs/figures/sfc_duplicate_rx.yaml"
)
```

### SFC MSE and throughput versus bandwidth

```python
from scripts.run_sfc_mse_throughput_vs_B import run_sfc_mse_throughput_vs_B

df_sfc_thr = run_sfc_mse_throughput_vs_B(
    "experiments/configs/figures/sfc_mse_throughput_vs_B.yaml"
)
```

---

## Configuration System

The configuration tree has two different roles.

### Master reference catalog

```text
experiments/configs/configs.yaml
```

This file is not meant to be a minimal runnable config. It is a long-lived reference catalog documenting supported keys, typical values, and conventions.

### Runnable experiment configs

```text
experiments/configs/figures/*.yaml
```

These are the actual YAML files used by figure runners and pipelines.

For details, see:

```text
experiments/configs/README.md
experiments/configs/CONFIG_REFERENCE.md
```

---

## Current SFC Conventions

### Theorem-consistent map generation

The SFC map generator follows a theorem-consistent orthogonal-vector construction.

The `R` resources are partitioned into `L` disjoint row-resource groups:

```text
D_1, D_2, ..., D_L
```

Each SFC map row can only activate a resource from that row's group.

Therefore, the number of theorem-valid maps is:

```text
|D_1| * |D_2| * ... * |D_L|
```

The standard SFC Fourier/phase representation requires:

```text
num_event_ids = 2 * N * S
```

A valid SFC configuration must satisfy:

```text
product_l |D_l| >= 2 * N * S
```

Example:

```text
S = 12
N = 5
L = 4
```

requires:

```text
2 * 5 * 12 = 120
```

event IDs.

With `R = 12`, the balanced partition is `[3, 3, 3, 3]`, so the number of theorem-valid maps is:

```text
3^4 = 81
```

which is insufficient.

With `R = 16`, the balanced partition is `[4, 4, 4, 4]`, so the number of theorem-valid maps is:

```text
4^4 = 256
```

which is sufficient.

If the theorem-valid map condition is not met, `SFCChannel` raises an error instead of silently generating invalid maps.

---

## SFC Detection

The SFC detector has two stages.

### 1. Local candidate generation

The local detector threshold-binarizes the received frame and scans all possible event-start windows.

For each candidate start slot `t0` and each reference map, it computes:

```text
score = sum(y_bin[t0:t0+L, :] * reference_map)
```

Supported local modes:

```yaml
channel:
  detection_mode: "strict"     # accept if score == L
```

```yaml
channel:
  detection_mode: "loose"      # accept if score > 0
```

```yaml
channel:
  detection_mode: "threshold"  # accept if score >= score_threshold
```

Recommended:

```yaml
channel:
  detection_mode: "threshold"
  score_threshold: 4
```

where `score_threshold` is usually equal to `system.L`.

### 2. Global candidate selection

The local detector can produce false positives when multiple events start in the same symbol slot. The global detector resolves this by selecting a subset of local candidates that best reconstructs the received frame.

Default modern behavior:

```yaml
channel:
  candidate_selection: "global_frame_fit"
```

To recover the old local-only behavior:

```yaml
channel:
  candidate_selection: "none"
```

Typical global block:

```yaml
channel:
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

Supported global strategies:

- `local_only`
- `event_penalty`
- `residual_rule`

---

## Semantic-Based Error Detection

SED is applied at the end of each `tau`-second transmission cycle.

The SED implementation follows the manuscript logic.

A period is invalid if either condition occurs.

### 1. Nonunique `t_z^(s,n)`

For a fixed sensor `s`, harmonic `n`, and coefficient type `z in {a,b}`, the receiver must recover at most one value.

Error condition:

```text
count(event_id) > 1
```

### 2. Failure in the pair `t_a^(s,n)`, `t_b^(s,n)`

For each sensor `s` and harmonic `n`, the pair must be either both present or both absent.

Valid:

```text
count(ta_id) = 0 and count(tb_id) = 0
count(ta_id) = 1 and count(tb_id) = 1
```

Invalid:

```text
count(ta_id) = 1 and count(tb_id) = 0
count(ta_id) = 0 and count(tb_id) = 1
```

Important:

```text
SED does not require exactly 2*N*S events per period.
```

This is intentional and manuscript-aligned, because Fourier coefficients below the harmonic threshold may not be transmitted.

Recommended SED block:

```yaml
sed:
  enabled: true
  discard_invalid_periods: true
  mode: "sparse_pairs"
  event_id_ordering: "ta_block_then_tb_block"
  throughput_mode: "valid_periods_times_S_over_tau"
```

Many existing pipelines use the pipeline-level switch:

```yaml
mode:
  semantic_error_detection: true
```

If both `mode.semantic_error_detection` and `sed.enabled` are present, keep them consistent.

---

## Physical Energy Convention

For SFC, `P` is the average transmit power per sensor over one period `tau`.

Each sensor transmits up to `2N` semantic events per period. Each event map has `L` active chips.

The total energy per sensor per period is:

```text
E_sensor = P * tau
```

The SFC active-chip energy is:

```text
E_chip = P * tau / (2 * N * L)
```

The matched-filter/resource-output signal level is:

```text
sqrt(E_chip)
```

The physical channel uses:

```text
y = sqrt(E_chip) * superposed + noise
```

---

## Generated Outputs

Most runners create outputs under:

```text
data/results/<experiment_name>/
```

or under a timestamped subdirectory, depending on the runner.

Typical generated files:

- `.dat` result table
- `.csv` result table, when enabled
- `.yaml` config copy
- metadata file
- `.png` figure
- `.pdf` figure
- diagnostics plot, when available

---

## Development Checks

Run syntax checks after modifying core files:

```bash
python -m py_compile \
  sfc/core/channel/SFCChannel.py \
  sfc/core/channel/detection.py \
  sfc/core/channel/collision.py \
  sfc/core/channel/physical_channel.py \
  sfc/core/semantic_error_detection.py \
  sfc/core/system_parameters.py \
  sfc/core/theory.py
```

Validate YAML syntax:

```bash
python - <<'PY'
from pathlib import Path
import yaml

paths = sorted(Path("experiments/configs").rglob("*.yaml"))

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        yaml.safe_load(f)
    print(f"YAML OK: {path}")

print(f"Total YAML files OK: {len(paths)}")
PY
```

---

## Recommended Workflow

1. Update core logic in `sfc/core`.
2. Add or update tests/debug console snippets.
3. Update runnable figure configs under `experiments/configs/figures/`.
4. Update configuration documentation:
   - `experiments/configs/configs.yaml`
   - `experiments/configs/README.md`
   - `experiments/configs/CONFIG_REFERENCE.md`
5. Run the relevant figure runner from the Python console.
6. Inspect `.dat`, `.csv`, plots, and metadata.
7. Commit focused changes.

---

## Notes for Future Maintenance

Whenever the configuration surface changes, update all of the following:

```text
experiments/configs/configs.yaml
experiments/configs/README.md
experiments/configs/CONFIG_REFERENCE.md
README.md
```

Whenever SFC map-generation logic changes, verify:

```text
product_l |D_l| >= 2 * N * S
```

Whenever SFC detection logic changes, test:

- clean channel;
- AWGN channel;
- local detector only;
- global detector;
- SED enabled;
- sparse-pair cases;
- complete same-slot overlaps;
- partial overlaps.
