    # Configuration Tree Guide

This directory contains **two different kinds of configuration files**:

1. **`configs.yaml`** → a **master reference catalog** of configurable keys.
2. **`figures/*.yaml`** → **runnable experiment configurations**.

---

## 1) What `configs.yaml` is

`experiments/configs/configs.yaml` is **not** intended to be the minimal config for a single run.

Instead, `configs.yaml` is a **long-lived reference file** that documents:

- major configuration sections supported by the core and pipelines;
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

---

## 2) What `figures/*.yaml` are

Files under `experiments/configs/figures/` are the **actual experiment configs** used by runners and pipelines.

Examples:

- `experiments/configs/figures/figure1.yaml`
- `experiments/configs/figures/figure2.yaml`
- `experiments/configs/figures/rbcp_mse_vs_B.yaml`
- `experiments/configs/figures/rbcp_mse_vs_B_fixed_power.yaml`
- `experiments/configs/figures/sfc_mse_throughput_vs_B.yaml`

These files should stay focused and only override the parameters needed by the experiment.

---

## 3) Recommended workflow

### When adding a new experiment

1. Create a new file under `experiments/configs/figures/`.
2. Keep it experiment-specific and as small as practical.
3. If the experiment introduces **new configurable keys**, also update:
   - `experiments/configs/configs.yaml`
   - `experiments/configs/CONFIG_REFERENCE.md`

### When changing the core configuration surface

If the `sfc/core` layer begins to accept a new key or a new value range, update the reference files even if no immediate figure uses the new option.

---

## 4) Current design principle

The tree is intentionally split into:

- **reference / documentation**
- **execution configs**

This keeps figure YAMLs cleaner while preserving a single place where all supported configuration knobs are documented.

---

## 5) Suggested maintenance rule

Whenever one of these changes happens:

- a new config key is introduced;
- a key is deprecated;
- a new valid option/value is added;
- a legacy alias is kept for compatibility;

update all three of the following:

- `experiments/configs/configs.yaml`
- `experiments/configs/README.md` (if the tree usage changed)
- `experiments/configs/CONFIG_REFERENCE.md`

This keeps the configuration system understandable long-term.
