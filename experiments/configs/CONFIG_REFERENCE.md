# Configuration Reference

This document mirrors the intent of `experiments/configs/configs.yaml`, but in
Markdown form for easier reading and navigation.

> **Important**
>
> - `experiments/configs/configs.yaml` is the **master YAML catalog of keys**.
> - `experiments/configs/figures/*.yaml` are the **actual runnable configs**.
> - This document is explanatory. The YAML catalog is the authoritative key list.

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
- Channel / PHY / SFC detection settings
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
- `signal.T` is still documented because some older figure pipelines accept it.
- If both are present, keep them numerically consistent.

---

### Bandwidth

- `system.B` is the **total system bandwidth**.
- Per-sensor bandwidth slices are derived inside the core from:

```yaml
system:
  bandwidth_allocation: null
