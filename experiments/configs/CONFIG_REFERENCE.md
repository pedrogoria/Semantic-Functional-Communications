# Configuration Reference

This document mirrors the intent of `experiments/configs/configs.yaml`, but in Markdown form for easier reading and navigation.

> **Important**
> - `experiments/configs/configs.yaml` is the **master YAML catalog of keys**.
> - `experiments/configs/figures/*.yaml` are the **actual runnable configs**.

---

## Quick Rule

- Use **`configs.yaml`** when you want to discover what the system can configure.
- Use **`figures/*.yaml`** when you want to run a specific experiment.

---

## Main Sections Covered

- Figure / experiment metadata
- Execution modes
- System parameters
- Signal parameters
- DC handling
- Quantization policy
- Benchmark settings
- RbCP settings
- RbCP_time settings
- Channel / PHY settings
- Semantic-based error detection (SED)
- Duplicate RX settings
- Sweep parameters
- Monte Carlo
- Metrics
- Output
- Data format
- Plot settings
- Truncation-comparison settings
- Reproducibility

---

## Notes on Current Conventions

### Signal period
- Prefer `signal.tau` in new configs.
- `signal.T` is still documented because some old figure pipelines accept it.

### Bandwidth
- `system.B` is the **total system bandwidth**.
- Per-sensor bandwidth slices are derived inside the core from `system.bandwidth_allocation`.

### Quantization
- Current aligned core defaults to:
  - `quantization.force_power_of_two: false`
  - `quantization.rounding_mode: "floor"`

This means that `M` and `M_RbCP` are, by default, **free integers** rather than powers of 2.

### Reference vs execution
- `configs.yaml` is intentionally **exhaustive** and **comment-heavy**.
- figure YAMLs should remain more concise and execution-focused.

---

## Maintenance Rule

Whenever the configuration surface changes, update:

1. `experiments/configs/configs.yaml`
2. `experiments/configs/README.md`
3. `experiments/configs/CONFIG_REFERENCE.md`

This keeps the configuration system understandable long-term.