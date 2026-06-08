#!/usr/bin/env python3
"""
tests/quantization_error_distribution_mc.py

Monte Carlo runner for the RbCP quantization-error distribution.

This script is intentionally placed under:

    tests/

and its YAML configuration is expected under:

    tests/configs/

Purpose
-------
Evaluate the accuracy of the uniform quantization-error assumption used in the
RbCP analytical MSE derivations.

The script implements two Monte Carlo pipelines.

Pipeline 1: random_signal
-------------------------
    random signal
        -> sfc.core.filters.filter_periodic
        -> sfc.core.fourier.FourierCoefficientCore.calc_an_bn_dft
        -> sfc.core.phase_cof.PhaseCoefficientCore.calc_ta_tb
        -> sfc.core.quantization.quantize_ta_tb
        -> normalized quantization error
        -> empirical distribution metrics.

Pipeline 2: random_coefficients
-------------------------------
    random Fourier coefficients satisfying a_n^2 + b_n^2 <= 4
        -> sfc.core.phase_cof.PhaseCoefficientCore.calc_ta_tb
        -> sfc.core.quantization.quantize_ta_tb
        -> normalized quantization error
        -> empirical distribution metrics.

For each M_RbCP, the normalized quantization error is

    E_norm = (T - Q(T)) / (ell_n/2),

where T is either t_a^(s,n) or t_b^(s,n), and

    ell_n = 2*pi / (M_RbCP * n * omega0)

is the quantization interval for harmonic n.

The empirical distribution of E_norm is compared against Uniform[-1,1] using:

- Kolmogorov-Smirnov distance;
- Wasserstein-1 distance;
- total variation distance over histogram bins;
- Kullback-Leibler divergence D_KL(P_emp || U);
- smoothed reverse KL divergence D_KL(U || P_emp);
- Jensen-Shannon divergence.

Output convention
-----------------
The script follows the project debug-output convention:

    output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/

Typical output files are:

- quantization_error_distribution_random_signal_pdf_<timestamp>.png/pdf;
- quantization_error_distribution_random_coefficients_pdf_<timestamp>.png/pdf;
- quantization_error_distribution_metrics_<timestamp>.csv;
- quantization_error_distribution_arrays_<timestamp>.npz;
- quantization_error_distribution_metadata_<timestamp>.json;
- quantization_error_distribution_<timestamp>.yaml.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "tests"
    / "configs"
    / "quantization_error_distribution_mc.yaml"
)

BASE_NAME = "quantization_error_distribution"


# =============================================================================
# YAML / OUTPUT
# =============================================================================

def load_yaml_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_output_dir(cfg: Dict[str, Any]) -> tuple[str, str]:
    output_cfg = cfg["output"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(
        output_cfg["base_dir"],
        output_cfg["figure_dir"],
        timestamp
    )

    os.makedirs(output_dir, exist_ok=True)

    return output_dir, timestamp


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def save_config_copy(config_path: str | Path, output_dir: str, timestamp: str) -> str:
    dst = os.path.join(output_dir, f"{BASE_NAME}_{timestamp}.yaml")
    shutil.copyfile(config_path, dst)
    return dst


def save_metadata(metadata: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    path = os.path.join(output_dir, f"{BASE_NAME}_metadata_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return path


def save_npz_payload(output_dir: str, timestamp: str, **arrays) -> str:
    path = os.path.join(output_dir, f"{BASE_NAME}_arrays_{timestamp}.npz")
    np.savez(path, **arrays)
    return path


def figure_formats(cfg: Dict[str, Any]) -> list[str]:
    return list(cfg["output"].get("formats", {}).get("figure", ["png", "pdf"]))


def save_figure(fig, output_dir: str, base_name: str, formats: list[str]) -> Dict[str, str]:
    generated = {}

    for fmt in formats:
        path = os.path.join(output_dir, f"{base_name}.{fmt}")

        if fmt == "png":
            fig.savefig(path, dpi=300)
        elif fmt == "pdf":
            fig.savefig(path)
        else:
            raise ValueError(f"Unsupported figure format: {fmt}")

        generated[fmt] = path

    plt.close(fig)
    return generated


def save_metrics_csv(
    metrics_rows: List[Dict[str, Any]],
    output_dir: str,
    timestamp: str
) -> str:
    path = os.path.join(output_dir, f"{BASE_NAME}_metrics_{timestamp}.csv")
    df = pd.DataFrame(metrics_rows)
    df.to_csv(path, index=False)
    return path


def _make_json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isfinite(obj):
            return float(obj)
        return None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def get_num_trials(cfg: Dict[str, Any]) -> int:
    return int(cfg["monte_carlo"].get("num_trials", cfg["monte_carlo"].get("interactions", 1000)))


def get_num_periods(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("signal", {}).get("n_periods", cfg.get("simulation", {}).get("n_periods", 1)))


def get_num_sensors(cfg: Dict[str, Any]) -> int:
    return int(cfg["system"].get("S", cfg["signal"].get("S", 1)))


def get_num_harmonics(cfg: Dict[str, Any]) -> int:
    return int(cfg["signal"].get("N_override", cfg["signal"].get("N", cfg["signal"].get("n_harmonics", 5))))


def get_tau(cfg: Dict[str, Any]) -> float:
    return float(cfg["signal"].get("tau", cfg["system"].get("tau", 1.0)))


def get_Tt(cfg: Dict[str, Any]) -> float:
    return float(cfg["signal"].get("Tt", cfg["signal"].get("sampling_period", 1e-3)))


def get_w0(cfg: Dict[str, Any]) -> float:
    return 2.0 * np.pi / get_tau(cfg)


def get_L(cfg: Dict[str, Any]) -> int:
    return int(cfg["system"].get("L", cfg.get("sfc", {}).get("L", 4)))


def get_R(cfg: Dict[str, Any]) -> int:
    return int(cfg["system"].get("R", cfg.get("sfc", {}).get("R", 12)))


def get_B(cfg: Dict[str, Any]) -> float:
    return float(cfg["system"].get("B", 1.0))


# =============================================================================
# SIGNAL GENERATION / PREPROCESSING
# =============================================================================

def generate_random_signals(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    num_time_samples: int,
    n_periods: int,
    S: int
) -> np.ndarray:
    """
    Generate S random signals across periods.

    Output shape:
        (time, periods, sensors)
    """
    dist = cfg["signal"].get("distribution", "uniform")

    if dist == "uniform":
        return rng.uniform(-1.0, 1.0, size=(num_time_samples, n_periods, S))

    if dist in ("gaussian", "normal"):
        return rng.normal(0.0, 1.0, size=(num_time_samples, n_periods, S))

    raise ValueError("signal.distribution must be 'uniform', 'gaussian', or 'normal'.")


def filter_signals_with_core(
    x_raw: np.ndarray,
    cfg: Dict[str, Any],
    tau: float,
    Tt: float
) -> np.ndarray:
    """
    Band-limit each signal using sfc.core.filters.filter_periodic.

    IMPORTANT:
    The filtering bandwidth is signal.W. It is not redefined from N.
    """
    W_filter = float(cfg["signal"]["W"])

    if W_filter <= 0:
        raise ValueError("signal.W must be positive.")

    x_filtered = np.zeros_like(x_raw, dtype=float)

    _, n_periods, S = x_raw.shape

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw[:, p, s],
                W_filter,
                Tt,
                tau
            )

    return x_filtered


def apply_peak_to_peak_control(x: np.ndarray, peak_to_peak: float) -> np.ndarray:
    """
    Apply peak-to-peak control independently per (period, sensor).
    """
    if float(peak_to_peak) == 0.0:
        return np.array(x, copy=True)

    x_out = np.array(x, copy=True, dtype=float)
    _, n_periods, S = x.shape

    for p in range(n_periods):
        for s in range(S):
            current_p2p = float(np.max(x[:, p, s]) - np.min(x[:, p, s]))
            if not np.isclose(current_p2p, 0.0):
                x_out[:, p, s] = x[:, p, s] * (float(peak_to_peak) / current_p2p)

    return x_out


def apply_dc_handling(
    x_filtered: np.ndarray,
    tau: float,
    Tt: float,
    dc_enabled: bool
) -> np.ndarray:
    """
    Remove DC component unless dc_enabled is True.
    """
    x_zero_mean = np.zeros_like(x_filtered, dtype=float)
    _, n_periods, S = x_filtered.shape

    for p in range(n_periods):
        for s in range(S):
            if not dc_enabled:
                dc = Tt * np.sum(x_filtered[:, p, s]) / tau
                x_zero_mean[:, p, s] = x_filtered[:, p, s] - dc
            else:
                x_zero_mean[:, p, s] = x_filtered[:, p, s]

    return x_zero_mean


# =============================================================================
# CORE FOURIER / PHASE / QUANTIZATION
# =============================================================================

def compute_fourier_coefficients_with_core(
    x_zero_mean: np.ndarray,
    cfg: Dict[str, Any],
    tau: float,
    Tt: float,
    N: int,
    S: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Fourier coefficients using sfc.core.fourier.FourierCoefficientCore.
    """
    fourier_core = FourierCoefficientCore(
        T=tau,
        harmonics=N,
        sensor_nodes=S
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=bool(cfg["signal"].get("normalize_dft", True)),
        norm=float(cfg["signal"].get("normalization_target", 3.99))
    )

    return an, bn, x_used


def compute_phase_coefficients_with_core(
    an: np.ndarray,
    bn: np.ndarray,
    cfg: Dict[str, Any],
    tau: float,
    N: int,
    S: int,
    n_periods: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ta/tb using sfc.core.phase_cof.PhaseCoefficientCore.
    """
    phase_core = PhaseCoefficientCore(
        T=tau,
        harmonics=N,
        n_sub_symbol=get_L(cfg),
        resource=get_R(cfg),
        sensor_nodes=S,
        bandwidth=get_B(cfg),
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=float(cfg["signal"].get("threshold_harmonics", 0.001))
    )

    ta, tb = phase_core.calc_ta_tb(an, bn)

    ta = np.real(ta).astype(float)
    tb = np.real(tb).astype(float)

    return ta, tb


def quantize_ta_tb_with_core(
    ta: np.ndarray,
    tb: np.ndarray,
    cfg: Dict[str, Any],
    M_rbcp: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize ta/tb tensors using sfc.core.quantization.quantize_ta_tb.

    Input shape:
        (periods, N, sensors)

    Output shape:
        (periods, N, sensors)
    """
    ta = np.asarray(ta, dtype=float)
    tb = np.asarray(tb, dtype=float)

    if ta.shape != tb.shape:
        raise ValueError("ta and tb must have the same shape.")

    if ta.ndim != 3:
        raise ValueError("ta and tb must have shape (periods, N, sensors).")

    w0 = get_w0(cfg)
    n_periods, _, S = ta.shape

    ta_q = np.zeros_like(ta, dtype=float)
    tb_q = np.zeros_like(tb, dtype=float)

    for p in range(n_periods):
        for s in range(S):
            ta_q[p, :, s], tb_q[p, :, s] = quantize_ta_tb(
                ta[p, :, s],
                tb[p, :, s],
                w0,
                int(M_rbcp)
            )

    return ta_q, tb_q


# =============================================================================
# RANDOM COEFFICIENT PIPELINE
# =============================================================================

def generate_random_coefficients_inside_lemma1_disk(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    n_periods: int,
    N: int,
    S: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random Fourier coefficients satisfying a_n^2 + b_n^2 <= 4.

    Output shape:
        (periods, N, sensors)
    """
    radius_max = float(cfg["random_coefficients"].get("coefficient_radius_max", 1.95))

    if radius_max <= 0.0 or radius_max > 2.0:
        raise ValueError("random_coefficients.coefficient_radius_max must be in (0,2].")

    phases = rng.uniform(-np.pi, np.pi, size=(n_periods, N, S))
    radii = radius_max * np.sqrt(rng.uniform(0.0, 1.0, size=(n_periods, N, S)))

    an = radii * np.cos(phases)
    bn = radii * np.sin(phases)

    return an, bn


# =============================================================================
# ERROR COLLECTION
# =============================================================================

def normalized_errors_from_ta_tb(
    ta: np.ndarray,
    tb: np.ndarray,
    ta_q: np.ndarray,
    tb_q: np.ndarray,
    cfg: Dict[str, Any],
    M_rbcp: int
) -> np.ndarray:
    """
    Compute normalized quantization errors from ta/tb and quantized ta/tb.

    For harmonic n:

        ell_n = 2*pi / (M_RbCP * n * omega0)

    and the normalized errors are:

        (ta - ta_q)/(ell_n/2),
        (tb - tb_q)/(ell_n/2).
    """
    ta = np.asarray(ta, dtype=float)
    tb = np.asarray(tb, dtype=float)
    ta_q = np.asarray(ta_q, dtype=float)
    tb_q = np.asarray(tb_q, dtype=float)

    if not (ta.shape == tb.shape == ta_q.shape == tb_q.shape):
        raise ValueError("ta, tb, ta_q, and tb_q must have the same shape.")

    if ta.ndim != 3:
        raise ValueError("ta/tb tensors must have shape (periods, N, sensors).")

    w0 = get_w0(cfg)
    _, N, _ = ta.shape

    errors = []

    for idx, n in enumerate(range(1, N + 1)):
        ell_n = 2.0 * np.pi / (float(M_rbcp) * n * w0)
        half_ell = ell_n / 2.0

        errors.append((ta[:, idx, :] - ta_q[:, idx, :]).reshape(-1) / half_ell)
        errors.append((tb[:, idx, :] - tb_q[:, idx, :]).reshape(-1) / half_ell)

    errors = np.concatenate(errors)

    # Numerical guard. Values should lie in [-1,1] up to floating-point noise.
    return np.clip(errors, -1.0, 1.0)


def collect_errors_random_signal_pipeline(
    cfg: Dict[str, Any],
    M_rbcp: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Pipeline 1:

        random signal
            -> filter_periodic
            -> FourierCoefficientCore.calc_an_bn_dft
            -> PhaseCoefficientCore.calc_ta_tb
            -> quantize_ta_tb
            -> normalized quantization error.
    """
    tau = get_tau(cfg)
    Tt = get_Tt(cfg)
    N = get_num_harmonics(cfg)
    S = get_num_sensors(cfg)
    n_periods = get_num_periods(cfg)

    t = np.arange(0.0, tau, Tt)
    n_time = len(t)

    x_raw = generate_random_signals(
        cfg=cfg,
        rng=rng,
        num_time_samples=n_time,
        n_periods=n_periods,
        S=S
    )

    x_filtered = filter_signals_with_core(
        x_raw=x_raw,
        cfg=cfg,
        tau=tau,
        Tt=Tt
    )

    x_filtered = apply_peak_to_peak_control(
        x_filtered,
        peak_to_peak=float(cfg["signal"].get("peak_to_peak", 0.0))
    )

    x_zero_mean = apply_dc_handling(
        x_filtered=x_filtered,
        tau=tau,
        Tt=Tt,
        dc_enabled=bool(cfg.get("dc", {}).get("enabled", False))
    )

    an, bn, _ = compute_fourier_coefficients_with_core(
        x_zero_mean=x_zero_mean,
        cfg=cfg,
        tau=tau,
        Tt=Tt,
        N=N,
        S=S
    )

    ta, tb = compute_phase_coefficients_with_core(
        an=an,
        bn=bn,
        cfg=cfg,
        tau=tau,
        N=N,
        S=S,
        n_periods=n_periods
    )

    ta_q, tb_q = quantize_ta_tb_with_core(
        ta=ta,
        tb=tb,
        cfg=cfg,
        M_rbcp=M_rbcp
    )

    return normalized_errors_from_ta_tb(
        ta=ta,
        tb=tb,
        ta_q=ta_q,
        tb_q=tb_q,
        cfg=cfg,
        M_rbcp=M_rbcp
    )


def collect_errors_random_coefficients_pipeline(
    cfg: Dict[str, Any],
    M_rbcp: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Pipeline 2:

        random a_n,b_n
            -> PhaseCoefficientCore.calc_ta_tb
            -> quantize_ta_tb
            -> normalized quantization error.
    """
    tau = get_tau(cfg)
    N = get_num_harmonics(cfg)
    S = get_num_sensors(cfg)
    n_periods = get_num_periods(cfg)

    an, bn = generate_random_coefficients_inside_lemma1_disk(
        cfg=cfg,
        rng=rng,
        n_periods=n_periods,
        N=N,
        S=S
    )

    ta, tb = compute_phase_coefficients_with_core(
        an=an,
        bn=bn,
        cfg=cfg,
        tau=tau,
        N=N,
        S=S,
        n_periods=n_periods
    )

    ta_q, tb_q = quantize_ta_tb_with_core(
        ta=ta,
        tb=tb,
        cfg=cfg,
        M_rbcp=M_rbcp
    )

    return normalized_errors_from_ta_tb(
        ta=ta,
        tb=tb,
        ta_q=ta_q,
        tb_q=tb_q,
        cfg=cfg,
        M_rbcp=M_rbcp
    )


# =============================================================================
# DISTRIBUTION METRICS
# =============================================================================

def histogram_probabilities(
    errors_norm: np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, edges = np.histogram(
        errors_norm,
        bins=bins,
        range=(-1.0, 1.0),
        density=False,
    )

    probs = counts.astype(float) / max(np.sum(counts), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return probs, edges, centers


def empirical_ks_distance_to_uniform(errors_norm: np.ndarray) -> float:
    x = np.sort(errors_norm)
    n = x.size

    if n == 0:
        return float("nan")

    f_uniform = (x + 1.0) / 2.0
    f_uniform = np.clip(f_uniform, 0.0, 1.0)

    f_emp_upper = np.arange(1, n + 1) / n
    f_emp_lower = np.arange(0, n) / n

    d_plus = np.max(f_emp_upper - f_uniform)
    d_minus = np.max(f_uniform - f_emp_lower)

    return float(max(d_plus, d_minus))


def wasserstein1_to_uniform(
    errors_norm: np.ndarray,
    grid_size: int = 2001,
) -> float:
    x_grid = np.linspace(-1.0, 1.0, grid_size)
    x_sorted = np.sort(errors_norm)

    empirical_cdf = (
        np.searchsorted(x_sorted, x_grid, side="right")
        / max(x_sorted.size, 1)
    )
    uniform_cdf = (x_grid + 1.0) / 2.0

    return float(np.trapz(np.abs(empirical_cdf - uniform_cdf), x_grid))


def total_variation_to_uniform(probs: np.ndarray) -> float:
    q = np.ones_like(probs, dtype=float) / probs.size
    return float(0.5 * np.sum(np.abs(probs - q)))


def kl_divergence_to_uniform_bits(
    probs: np.ndarray,
    eps: float = 1e-15,
) -> float:
    p = probs.astype(float)
    p = p / max(np.sum(p), eps)

    q = np.ones_like(p, dtype=float) / p.size

    mask = p > 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def reverse_kl_uniform_to_empirical_bits(
    probs: np.ndarray,
    eps: float = 1e-15,
) -> float:
    p = probs.astype(float)
    p = p + eps
    p = p / np.sum(p)

    q = np.ones_like(p, dtype=float) / p.size

    return float(np.sum(q * np.log2(q / p)))


def jensen_shannon_divergence_bits(
    probs: np.ndarray,
) -> float:
    p = probs.astype(float)
    p = p / max(np.sum(p), 1e-15)

    q = np.ones_like(p, dtype=float) / p.size
    m = 0.5 * (p + q)

    def kl_bits(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def compute_metrics(
    pipeline_name: str,
    M_rbcp: int,
    errors_norm: np.ndarray,
    bins: int,
) -> Dict[str, Any]:
    probs, _, _ = histogram_probabilities(errors_norm, bins=bins)

    return {
        "pipeline": pipeline_name,
        "M_RbCP": int(M_rbcp),
        "num_errors": int(errors_norm.size),
        "mean": float(np.mean(errors_norm)),
        "variance": float(np.var(errors_norm)),
        "uniform_mean_reference": 0.0,
        "uniform_variance_reference": 1.0 / 3.0,
        "ks_distance": empirical_ks_distance_to_uniform(errors_norm),
        "wasserstein1_distance": wasserstein1_to_uniform(errors_norm),
        "total_variation_hist": total_variation_to_uniform(probs),
        "kl_empirical_to_uniform_bits_hist": kl_divergence_to_uniform_bits(probs),
        "kl_uniform_reference_bits": 0.0,
        "reverse_kl_uniform_to_empirical_bits_hist": (
            reverse_kl_uniform_to_empirical_bits(probs)
        ),
        "jensen_shannon_bits_hist": jensen_shannon_divergence_bits(probs),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_error_distributions_for_pipeline(
    pipeline_name: str,
    histograms: Dict[int, Tuple[np.ndarray, np.ndarray]],
    metrics_by_m: Dict[int, Dict[str, Any]],
    output_dir: str,
    timestamp: str,
    formats: list[str],
) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for m, (centers, density) in histograms.items():
        kl = metrics_by_m[m]["kl_empirical_to_uniform_bits_hist"]
        js = metrics_by_m[m]["jensen_shannon_bits_hist"]

        ax.plot(
            centers,
            density,
            linewidth=1.8,
            label=(
                rf"$M_{{\rm RbCP}}={m}$, "
                rf"KL={kl:.3g}, JS={js:.3g}"
            ),
        )

    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"Uniform $[-1,1]$",
    )

    ax.set_xlabel(r"Normalized quantization error, $E/(\ell_n/2)$")
    ax.set_ylabel("Empirical probability density")
    ax.set_title(f"RbCP quantization error distribution: {pipeline_name}")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8, loc="best", frameon=True)

    fig.tight_layout()

    safe_pipeline_name = pipeline_name.replace(" ", "_")

    return {
        f"{safe_pipeline_name}_pdf_{fmt}": path
        for fmt, path in save_figure(
            fig,
            output_dir,
            f"{BASE_NAME}_{safe_pipeline_name}_pdf_{timestamp}",
            formats
        ).items()
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_quantization_error_distribution_mc(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    show_plots: bool = False,
) -> Dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_yaml_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    seed = int(cfg["monte_carlo"]["seed"])
    num_trials = get_num_trials(cfg)
    bins = int(cfg["analysis"]["histogram_bins"])
    m_values = [int(v) for v in cfg["analysis"]["m_rbcp_values"]]
    pipelines = list(cfg["analysis"].get("pipelines", ["random_signal", "random_coefficients"]))

    print("\n[QUANTIZATION ERROR DISTRIBUTION MC]")
    print(f"[INFO] output_dir = {output_dir}")
    print(f"[INFO] timestamp = {timestamp}")
    print(f"[INFO] seed = {seed}")
    print(f"[INFO] num_trials = {num_trials}")
    print(f"[INFO] n_periods = {get_num_periods(cfg)}")
    print(f"[INFO] S = {get_num_sensors(cfg)}")
    print(f"[INFO] N = {get_num_harmonics(cfg)}")
    print(f"[INFO] tau = {get_tau(cfg):.6e}")
    print(f"[INFO] Tt = {get_Tt(cfg):.6e}")
    print(f"[INFO] M_RbCP values = {m_values}")
    print(f"[INFO] pipelines = {pipelines}")
    print(f"[INFO] histogram_bins = {bins}")

    metrics_rows: List[Dict[str, Any]] = []
    metrics_nested: Dict[str, Dict[int, Dict[str, Any]]] = {}
    generated_files: Dict[str, str] = {}

    arrays_payload: Dict[str, Any] = {
        "M_RbCP_values": np.asarray(m_values, dtype=int),
    }

    formats = figure_formats(cfg)

    for pipeline_name in pipelines:
        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] Pipeline = {pipeline_name}")

        metrics_by_m: Dict[int, Dict[str, Any]] = {}
        histograms_for_plot: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for m in m_values:
            rng_m = np.random.default_rng(seed + 100000 * _pipeline_index(pipeline_name) + m)

            all_errors = []

            for _ in range(num_trials):
                if pipeline_name == "random_signal":
                    errors_norm = collect_errors_random_signal_pipeline(
                        cfg=cfg,
                        M_rbcp=m,
                        rng=rng_m
                    )

                elif pipeline_name == "random_coefficients":
                    errors_norm = collect_errors_random_coefficients_pipeline(
                        cfg=cfg,
                        M_rbcp=m,
                        rng=rng_m
                    )

                else:
                    raise ValueError(
                        "Unknown pipeline. Use 'random_signal' or 'random_coefficients'."
                    )

                all_errors.append(errors_norm)

            errors_norm = np.concatenate(all_errors)

            metrics = compute_metrics(
                pipeline_name=pipeline_name,
                M_rbcp=m,
                errors_norm=errors_norm,
                bins=bins,
            )

            metrics_by_m[m] = metrics
            metrics_rows.append(metrics)

            probs, edges, centers = histogram_probabilities(
                errors_norm,
                bins=bins,
            )

            bin_width = edges[1] - edges[0]
            density = probs / bin_width

            histograms_for_plot[m] = (centers, density)

            key_prefix = f"{pipeline_name}_M_{m}"
            arrays_payload[f"errors_norm_{key_prefix}"] = errors_norm
            arrays_payload[f"hist_probs_{key_prefix}"] = probs
            arrays_payload[f"hist_density_{key_prefix}"] = density
            arrays_payload[f"hist_centers_{key_prefix}"] = centers
            arrays_payload[f"hist_edges_{key_prefix}"] = edges

            print(
                f"[INFO] pipeline={pipeline_name} | "
                f"M_RbCP={m:>5d} | "
                f"Nerr={metrics['num_errors']} | "
                f"mean={metrics['mean']:.6e} | "
                f"var={metrics['variance']:.6e} | "
                f"KS={metrics['ks_distance']:.6e} | "
                f"W1={metrics['wasserstein1_distance']:.6e} | "
                f"TV={metrics['total_variation_hist']:.6e} | "
                f"KL(P||U)={metrics['kl_empirical_to_uniform_bits_hist']:.6e} bits | "
                f"JS={metrics['jensen_shannon_bits_hist']:.6e} bits"
            )

        metrics_nested[pipeline_name] = metrics_by_m

        generated_files.update(
            plot_error_distributions_for_pipeline(
                pipeline_name=pipeline_name,
                histograms=histograms_for_plot,
                metrics_by_m=metrics_by_m,
                output_dir=output_dir,
                timestamp=timestamp,
                formats=formats,
            )
        )

    metrics_csv_path = save_metrics_csv(
        metrics_rows=metrics_rows,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    generated_files["metrics_csv"] = metrics_csv_path

    arrays_path = save_npz_payload(
        output_dir=output_dir,
        timestamp=timestamp,
        **arrays_payload,
    )
    generated_files["arrays_npz"] = arrays_path

    config_copy = save_config_copy(
        config_path=config_path,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    generated_files["config"] = config_copy

    metadata = {
        "created_at": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "output_dir": output_dir,
        "git_commit": get_git_commit(),
        "experiment": {
            "name": BASE_NAME,
            "purpose": (
                "Monte Carlo evaluation of the uniform quantization-error "
                "assumption for RbCP time-shift parameters using sfc.core "
                "Fourier, phase, and quantization functions."
            ),
        },
        "simulation": {
            "seed": seed,
            "num_trials": num_trials,
            "n_periods": get_num_periods(cfg),
            "S": get_num_sensors(cfg),
            "N": get_num_harmonics(cfg),
            "tau": get_tau(cfg),
            "Tt": get_Tt(cfg),
            "m_rbcp_values": m_values,
            "pipelines": pipelines,
            "histogram_bins": bins,
            "normalized_error_definition": "E_norm = (T - Q(T))/(ell_n/2)",
            "theoretical_reference": "Uniform[-1,1]",
        },
        "core_functions": {
            "filter": "sfc.core.filters.filter_periodic",
            "fourier": "sfc.core.fourier.FourierCoefficientCore.calc_an_bn_dft",
            "phase": "sfc.core.phase_cof.PhaseCoefficientCore.calc_ta_tb",
            "quantization": "sfc.core.quantization.quantize_ta_tb",
        },
        "uniform_reference": {
            "support": [-1.0, 1.0],
            "density": 0.5,
            "mean": 0.0,
            "variance": 1.0 / 3.0,
            "kl_reference_bits": 0.0,
            "histogram_probability_per_bin": 1.0 / bins,
        },
        "distance_definitions": {
            "ks_distance": "sup_x |F_empirical(x)-F_uniform(x)|",
            "wasserstein1_distance": (
                "Integral of absolute CDF difference over [-1,1]"
            ),
            "total_variation_hist": (
                "0.5 * sum_i |p_i-u_i| over histogram bins"
            ),
            "kl_empirical_to_uniform_bits_hist": (
                "sum_i p_i log2(p_i/u_i), with u_i=1/number_of_bins"
            ),
            "kl_uniform_reference_bits": (
                "Theoretical value D_KL(U||U)=D_KL(P||U)=0 when P=U"
            ),
            "reverse_kl_uniform_to_empirical_bits_hist": (
                "sum_i u_i log2(u_i/p_i), using additive smoothing"
            ),
            "jensen_shannon_bits_hist": (
                "0.5 KL(P||M)+0.5 KL(U||M), M=(P+U)/2"
            ),
        },
        "metrics_by_pipeline_and_M_RbCP": _make_json_safe(metrics_nested),
        "generated_files": generated_files,
    }

    metadata_path = save_metadata(
        metadata=metadata,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    generated_files["metadata"] = metadata_path

    print("\n[INFO] Generated files:")
    for key, path in generated_files.items():
        print(f"       {key}: {path}")

    if show_plots:
        plt.show()

    return {
        "metrics_rows": metrics_rows,
        "metrics_nested": metrics_nested,
        "output_dir": output_dir,
        "timestamp": timestamp,
        "generated_files": generated_files,
    }


def _pipeline_index(pipeline_name: str) -> int:
    if pipeline_name == "random_signal":
        return 1
    if pipeline_name == "random_coefficients":
        return 2
    return 9


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Monte Carlo sensitivity analysis for the RbCP "
            "uniform quantization-error assumption using sfc.core functions."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively."
    )

    args = parser.parse_args()

    run_quantization_error_distribution_mc(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()
