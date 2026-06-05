"""
tests/run_debug_cs_fdma_single_trial.py

Debug runner for one CS + FDMA transmission cycle.

This script is intentionally placed under:

    tests/

and its YAML configuration is expected under:

    tests/configs/

Purpose
-------
Visualize and diagnose the CS + FDMA branch with the physically clearer chain:

    x(t) -> x[n] -> y = Phi x[n] -> q = Q(y) -> y_tilde
         -> x_hat[n] -> x_hat(t)

The script shows:
- dense reference signal x(t);
- uniform samples x[n] used by the CS encoder;
- ideal sinc reconstruction from x[n];
- real CS measurements y;
- quantized measurement indices q, when enabled;
- dequantized measurements y_tilde, when enabled;
- CS reconstruction x_hat[n];
- sinc reconstruction x_hat(t);
- top-K DCT coefficients of x[n];
- accumulated DCT energy of x[n];
- physical channel-budget panel.

Output convention
-----------------
The script follows the project debug-output convention:

    output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/
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
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.fft import dct as scipy_dct
from scipy.fft import idct as scipy_idct

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sfc.core.acquisition.cs import CSAcquisitionCore
from sfc.core.filters import filter_periodic
from sfc.core.system_parameters import build_derived_system_parameters


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "tests"
    / "configs"
    / "cs_fdma_single_trial_debug.yaml"
)

BASE_NAME = "cs_fdma_debug"


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


# =============================================================================
# SAMPLING / SINC
# =============================================================================

def resolve_sampling_rate(cfg: Dict[str, Any], params) -> float:
    benchmark_cfg = cfg.get("benchmark", {})
    sampling_rate = float(benchmark_cfg.get("sampling_rate", params.W))
    effective_rate_factor = float(benchmark_cfg.get("effective_rate_factor", 2.0))

    fs = effective_rate_factor * sampling_rate

    if fs <= 0:
        raise ValueError("Resolved sampling rate must be positive.")

    return fs


def sinc_reconstruct_from_samples(
    x_samples: np.ndarray,
    t_eval: np.ndarray,
    fs: float
) -> np.ndarray:
    x_samples = np.asarray(x_samples, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    n = np.arange(len(x_samples))
    x_hat = np.zeros_like(t_eval, dtype=float)

    for i, ti in enumerate(t_eval):
        x_hat[i] = np.sum(x_samples * np.sinc(fs * ti - n))

    return x_hat


def sample_dense_signal(
    x_dense: np.ndarray,
    t_dense: np.ndarray,
    tau: float,
    fs: float
) -> tuple[np.ndarray, np.ndarray]:
    Ts = 1.0 / fs
    t_samples = np.arange(0.0, tau, Ts)
    x_samples = np.interp(t_samples, t_dense, x_dense)
    return t_samples, x_samples


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_random_filtered_signal(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    params,
    t_dense: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time = len(t_dense)
    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=n_time)
    elif dist == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=n_time)
    else:
        raise ValueError("signal.distribution must be 'uniform' or 'gaussian'.")

    tau = float(params.tau)
    Tt = float(cfg["signal"]["Tt"])
    W_filter = float(cfg["signal"].get("W", params.W))

    x_filtered = filter_periodic(x_raw, W_filter, Tt, tau)

    x_filtered = apply_peak_to_peak_control(
        x_filtered,
        peak_to_peak=float(cfg["signal"].get("peak_to_peak", 0.0))
    )

    x_zero_mean = apply_dc_handling(
        x_filtered,
        tau=tau,
        Tt=Tt,
        dc_enabled=bool(cfg.get("dc", {}).get("enabled", False))
    )

    return x_raw, x_filtered, x_zero_mean


def generate_synthetic_sparse_dct_sampled_signal(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    params,
    t_dense: np.ndarray,
    fs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate x[n] exactly sparse in DCT, then obtain x(t) by sinc.
    """

    debug_cfg = cfg.get("debug", {})
    tau = float(params.tau)

    Ts = 1.0 / fs
    t_samples = np.arange(0.0, tau, Ts)
    n_samples = len(t_samples)

    K = int(debug_cfg.get("synthetic_sparsity", cfg["cs"].get("sparsity", 5)))

    if K < 1:
        raise ValueError("synthetic_sparsity must be >= 1.")
    if K >= n_samples:
        raise ValueError("synthetic_sparsity must be smaller than number of samples.")

    allow_dc = bool(debug_cfg.get("synthetic_allow_dc", False))

    if allow_dc:
        candidates = np.arange(n_samples)
    else:
        candidates = np.arange(1, n_samples)

    support = rng.choice(candidates, size=K, replace=False)
    support = np.sort(support)

    alpha = np.zeros(n_samples, dtype=float)

    amp_mode = debug_cfg.get("synthetic_amplitude_mode", "gaussian")

    if amp_mode == "gaussian":
        alpha[support] = rng.normal(0.0, 1.0, size=K)
    elif amp_mode == "uniform":
        alpha[support] = rng.uniform(-1.0, 1.0, size=K)
    elif amp_mode == "ones":
        alpha[support] = 1.0
    else:
        raise ValueError(
            "synthetic_amplitude_mode must be 'gaussian', 'uniform', or 'ones'."
        )

    x_samples = scipy_idct(alpha, norm="ortho")

    peak_to_peak = float(cfg["signal"].get("peak_to_peak", 0.0))
    x_samples = apply_peak_to_peak_control(x_samples, peak_to_peak)

    if not bool(cfg.get("dc", {}).get("enabled", False)):
        x_samples = x_samples - np.mean(x_samples)

    x_dense = sinc_reconstruct_from_samples(x_samples, t_dense, fs)

    x_raw = np.array(x_dense, copy=True)
    x_filtered = np.array(x_dense, copy=True)
    x_zero_mean = np.array(x_dense, copy=True)

    return x_raw, x_filtered, x_zero_mean, t_samples, x_samples


def apply_peak_to_peak_control(x: np.ndarray, peak_to_peak: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if peak_to_peak == 0:
        return np.array(x, copy=True)

    current_p2p = float(np.max(x) - np.min(x))

    if np.isclose(current_p2p, 0.0):
        return np.array(x, copy=True)

    return x * (float(peak_to_peak) / current_p2p)


def apply_dc_handling(
    x: np.ndarray,
    tau: float,
    Tt: float,
    dc_enabled: bool
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if dc_enabled:
        return np.array(x, copy=True)

    dc = Tt * np.sum(x) / tau
    return x - dc


# =============================================================================
# CHANNEL BUDGET
# =============================================================================

def compute_cs_budget_for_sensor(
    params,
    sensor_index: int,
    bits_per_measurement: float
) -> Dict[str, float]:
    B_sensor = float(params.B_per_sensor[sensor_index])
    SNR_sensor = float(params.P / (B_sensor * params.N0))
    SNR_sensor_dB = float(10.0 * np.log10(SNR_sensor))
    capacity_sensor = float(B_sensor * np.log2(1.0 + SNR_sensor))
    bits_per_cycle = float(capacity_sensor * params.tau)

    n_measurements_raw = int(np.floor(bits_per_cycle / bits_per_measurement))

    return {
        "B_sensor": B_sensor,
        "SNR_sensor": SNR_sensor,
        "SNR_sensor_dB": SNR_sensor_dB,
        "capacity_sensor": capacity_sensor,
        "bits_per_cycle": bits_per_cycle,
        "n_measurements_raw": n_measurements_raw,
    }


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

def _safe_get_attr_or_key(obj: Any, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)

    if isinstance(obj, dict) and name in obj:
        return obj[name]

    return None


def extract_measurements_real(acq) -> Optional[np.ndarray]:
    value = _safe_get_attr_or_key(acq, "measurements_real")
    if value is None:
        value = _safe_get_attr_or_key(acq, "measurements")
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def extract_quantized_indices(acq) -> Optional[np.ndarray]:
    value = _safe_get_attr_or_key(acq, "measurements_quantized_indices")
    if value is None:
        return None
    return np.asarray(value).reshape(-1)


def extract_measurements_dequantized(acq) -> Optional[np.ndarray]:
    value = _safe_get_attr_or_key(acq, "measurements_dequantized")
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def safe_dct_1d(x: np.ndarray) -> np.ndarray:
    return scipy_dct(np.asarray(x, dtype=float), norm="ortho")


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


def metadata_from_quantization(acq) -> Dict[str, Any]:
    meta = _safe_get_attr_or_key(acq, "quantization_metadata")
    if meta is None:
        return {"enabled": False}
    return _make_json_safe(meta)


# =============================================================================
# PLOT HELPERS
# =============================================================================

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


def plot_signal_reconstruction(
    t_dense,
    x_dense,
    t_samples,
    x_samples,
    x_sampling_ideal_dense,
    x_hat_ideal_dense,
    x_hat_quantized_dense,
    output_dir,
    timestamp,
    formats
):
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    ax.plot(t_dense, x_dense, linewidth=2.0, label="x(t) reference")
    ax.scatter(t_samples, x_samples, s=24, label="x[n] samples", zorder=3)
    ax.plot(
        t_dense,
        x_sampling_ideal_dense,
        linestyle=":",
        linewidth=1.7,
        label="sinc from true samples"
    )
    ax.plot(
        t_dense,
        x_hat_ideal_dense,
        linestyle="--",
        linewidth=1.5,
        label="CS ideal y -> sinc"
    )

    if x_hat_quantized_dense is not None:
        ax.plot(
            t_dense,
            x_hat_quantized_dense,
            linestyle="-.",
            linewidth=1.5,
            label="CS quantized y -> sinc"
        )

    ax.set_xlabel("t")
    ax.set_ylabel("Amplitude")
    ax.set_title("CS + FDMA debug: x(t), x[n], CS recovery, sinc reconstruction")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()

    return {
        f"signal_{fmt}": path
        for fmt, path in save_figure(
            fig,
            output_dir,
            f"{BASE_NAME}_signal_{timestamp}",
            formats
        ).items()
    }


def plot_measurements(
    y_real,
    q_indices,
    y_dequantized,
    output_dir,
    timestamp,
    formats
):
    if y_real is None:
        return {}

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    k = np.arange(len(y_real))

    axes[0].plot(k, y_real, marker="o", linewidth=1.0, markersize=3)
    axes[0].set_ylabel("y")
    axes[0].set_title("CS measurements: real, quantized indices, dequantized")
    axes[0].grid(True, linestyle=":", linewidth=0.7)

    if q_indices is not None:
        axes[1].step(k, q_indices, where="mid", linewidth=1.1)
    else:
        axes[1].text(
            0.05,
            0.5,
            "Quantization disabled",
            transform=axes[1].transAxes,
            family="monospace"
        )
    axes[1].set_ylabel("q index")
    axes[1].grid(True, linestyle=":", linewidth=0.7)

    axes[2].plot(k, y_real, marker="o", linewidth=1.0, markersize=3, label="y real")

    if y_dequantized is not None:
        axes[2].plot(
            k,
            y_dequantized,
            marker="x",
            linewidth=1.0,
            markersize=4,
            label="y_tilde"
        )

    axes[2].set_xlabel("measurement index")
    axes[2].set_ylabel("value")
    axes[2].grid(True, linestyle=":", linewidth=0.7)
    axes[2].legend(loc="best", frameon=True)

    fig.tight_layout()

    return {
        f"measurements_{fmt}": path
        for fmt, path in save_figure(
            fig,
            output_dir,
            f"{BASE_NAME}_measurements_{timestamp}",
            formats
        ).items()
    }


def plot_topk_coefficients(
    coeff_true,
    coeff_ideal,
    coeff_quantized,
    top_k,
    output_dir,
    timestamp,
    formats
):
    coeff_true = np.asarray(coeff_true, dtype=float)
    coeff_ideal = np.asarray(coeff_ideal, dtype=float)

    top_k = min(int(top_k), len(coeff_true))
    idx = np.argsort(np.abs(coeff_true))[-top_k:]
    idx = np.sort(idx)

    x = np.arange(len(idx))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.0, 4.8))

    ax.bar(x - width, np.abs(coeff_true[idx]), width=width, label="true samples")
    ax.bar(x, np.abs(coeff_ideal[idx]), width=width, label="CS ideal y")

    if coeff_quantized is not None:
        coeff_quantized = np.asarray(coeff_quantized, dtype=float)
        ax.bar(
            x + width,
            np.abs(coeff_quantized[idx]),
            width=width,
            label="CS quantized y"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idx], rotation=45)
    ax.set_xlabel("DCT coefficient index of sampled vector")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Top-{top_k} DCT coefficients of x[n]")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()

    return {
        f"topk_coefficients_{fmt}": path
        for fmt, path in save_figure(
            fig,
            output_dir,
            f"{BASE_NAME}_topk_coefficients_{timestamp}",
            formats
        ).items()
    }


def plot_accumulated_energy(coeff_true, output_dir, timestamp, formats):
    coeff_true = np.asarray(coeff_true, dtype=float)
    sorted_abs = np.sort(np.abs(coeff_true))[::-1]
    denom = float(np.sum(sorted_abs ** 2))

    if np.isclose(denom, 0.0):
        energy = np.zeros_like(sorted_abs)
    else:
        energy = np.cumsum(sorted_abs ** 2) / denom

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    ax.plot(np.arange(1, len(energy) + 1), energy, linewidth=1.5)
    ax.axhline(0.90, linestyle="--", linewidth=1.0, label="90%")
    ax.axhline(0.99, linestyle="--", linewidth=1.0, label="99%")

    ax.set_xlabel("Number of largest sample-DCT coefficients")
    ax.set_ylabel("Accumulated energy fraction")
    ax.set_title("Compressibility of x[n] in DCT domain")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()

    return {
        f"energy_{fmt}": path
        for fmt, path in save_figure(
            fig,
            output_dir,
            f"{BASE_NAME}_accumulated_energy_{timestamp}",
            formats
        ).items()
    }


def plot_error(
    t_dense,
    x_dense,
    x_sampling_ideal_dense,
    x_hat_ideal_dense,
    x_hat_quantized_dense,
    output_dir,
    timestamp,
    formats
):
    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    ax.plot(
        t_dense,
        x_dense - x_sampling_ideal_dense,
        linewidth=1.3,
        label="sampling+sinc error"
    )
    ax.plot(
        t_dense,
        x_dense - x_hat_ideal_dense,
        linewidth=1.3,
        linestyle="--",
        label="CS ideal y error"
    )

    if x_hat_quantized_dense is not None:
        ax.plot(
            t_dense,
            x_dense - x_hat_quantized_dense,
            linewidth=1.3,
            linestyle="-.",
            label="CS quantized y error"
        )

    ax.set_xlabel("t")
    ax.set_ylabel("Error")
    ax.set_title("Dense-time reconstruction error")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()

    return {
        f"error_{fmt}": path
        for fmt, path in save_figure(
            fig,
            output_dir,
            f"{BASE_NAME}_error_{timestamp}",
            formats
        ).items()
    }


def plot_channel_budget_panel(
    params,
    budget,
    fs,
    n_samples,
    measurement_bits,
    n_measurements,
    sparsity_eff,
    quantize_measurements,
    output_dir,
    timestamp,
    formats
):
    text = (
        "CS + FDMA physical/channel budget\n"
        "--------------------------------\n"
        f"B_total              = {params.B:.6e}\n"
        f"P_per_sensor         = {params.P:.6e}\n"
        f"N0                   = {params.N0:.6e}\n"
        f"B_sensor             = {budget['B_sensor']:.6e}\n"
        f"SNR_sensor           = {budget['SNR_sensor']:.6e}\n"
        f"SNR_sensor_dB        = {budget['SNR_sensor_dB']:.6f}\n"
        f"C_sensor             = {budget['capacity_sensor']:.6e} bits/s\n"
        f"tau                  = {params.tau:.6e} s\n"
        f"bits_per_cycle       = {budget['bits_per_cycle']:.6e}\n"
        f"sampling_rate fs     = {fs:.6e} Hz\n"
        f"num_samples x[n]     = {n_samples}\n"
        f"measurement_bits     = {measurement_bits}\n"
        f"n_measurements M     = {n_measurements}\n"
        f"sparsity_eff K       = {sparsity_eff}\n"
        f"quantize_measurements= {quantize_measurements}\n"
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10
    )
    ax.axis("off")

    fig.tight_layout()

    return {
        f"channel_budget_{fmt}": path
        for fmt, path in save_figure(
            fig,
            output_dir,
            f"{BASE_NAME}_channel_budget_{timestamp}",
            formats
        ).items()
    }


# =============================================================================
# CS RUN
# =============================================================================

def run_cs_core_once(
    x_samples: np.ndarray,
    t_samples: np.ndarray,
    cs_cfg: Dict[str, Any],
    n_measurements: int,
    sparsity_eff: int,
    quantize_measurements: bool,
) -> tuple[Any, Any, np.ndarray]:
    cs_random_state = cs_cfg.get("random_state", None)

    measurement_bits = int(cs_cfg.get("measurement_bits", 8))

    cs_core = CSAcquisitionCore(
        n_measurements=n_measurements,
        sparsity=sparsity_eff,
        basis=cs_cfg.get("basis", "dct"),
        sensing_matrix=cs_cfg.get("sensing_matrix", "gaussian"),
        random_state=cs_random_state,
        normalize_dictionary_columns=cs_cfg.get("normalize_dictionary_columns", True),
        store_true_representation=cs_cfg.get("store_true_representation", False),
        quantize_measurements=quantize_measurements,
        measurement_bits=measurement_bits,
        quantization_mode=cs_cfg.get(
            "measurement_quantization_mode",
            "uniform_midrise"
        ),
        quantization_range=cs_cfg.get(
            "measurement_quantization_range",
            "per_signal"
        ),
        measurement_quantization_min=cs_cfg.get(
            "measurement_quantization_min",
            None
        ),
        measurement_quantization_max=cs_cfg.get(
            "measurement_quantization_max",
            None
        ),
        clip_quantization=cs_cfg.get("clip_quantization", True),
    )

    x_tensor = x_samples[:, None, None]

    acq = cs_core.acquire(x=x_tensor, t=t_samples)
    rec = cs_core.reconstruct(acq)

    x_hat_samples = np.asarray(rec.reconstructed_signal[:, 0, 0], dtype=float)

    return acq, rec, x_hat_samples


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_debug_cs_fdma_single_trial(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    show_plots: bool = False,
) -> Dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_yaml_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])
    params = build_derived_system_parameters(cfg)

    tau = float(params.tau)
    Tt = float(cfg["signal"]["Tt"])
    t_dense = np.arange(0.0, tau, Tt)

    debug_cfg = cfg.get("debug", {})
    sensor_index = int(debug_cfg.get("sensor_index", 0))
    top_k = int(debug_cfg.get("top_k_coefficients", 20))

    if sensor_index < 0 or sensor_index >= params.S:
        raise ValueError(f"debug.sensor_index must be in [0, {params.S - 1}]")

    cs_cfg = cfg["cs"]

    measurement_bits = int(cs_cfg.get("measurement_bits", 8))
    bits_per_measurement = float(
        cs_cfg.get("bits_per_measurement", measurement_bits)
    )
    sparsity_cfg = int(cs_cfg.get("sparsity", 5))
    quantize_measurements = bool(cs_cfg.get("quantize_measurements", False))

    fs = resolve_sampling_rate(cfg, params)

    signal_mode = debug_cfg.get("signal_mode", "random_filtered")

    if signal_mode == "random_filtered":
        x_raw, x_filtered, x_dense = generate_random_filtered_signal(
            cfg=cfg,
            rng=rng,
            params=params,
            t_dense=t_dense
        )

        t_samples, x_samples = sample_dense_signal(
            x_dense=x_dense,
            t_dense=t_dense,
            tau=tau,
            fs=fs
        )

    elif signal_mode == "synthetic_sparse_dct_samples":
        (
            x_raw,
            x_filtered,
            x_dense,
            t_samples,
            x_samples,
        ) = generate_synthetic_sparse_dct_sampled_signal(
            cfg=cfg,
            rng=rng,
            params=params,
            t_dense=t_dense,
            fs=fs
        )

    else:
        raise ValueError(
            "debug.signal_mode must be 'random_filtered' or "
            "'synthetic_sparse_dct_samples'."
        )

    n_samples = len(x_samples)

    budget = compute_cs_budget_for_sensor(
        params=params,
        sensor_index=sensor_index,
        bits_per_measurement=bits_per_measurement
    )

    n_measurements = int(np.floor(budget["bits_per_cycle"] / bits_per_measurement))
    n_measurements = max(1, min(n_measurements, n_samples))

    override = debug_cfg.get("n_measurements_override", None)
    if override is not None:
        n_measurements = int(override)
        n_measurements = max(1, min(n_measurements, n_samples))

    sparsity_eff = min(sparsity_cfg, n_measurements)

    print("\n[CS FDMA DEBUG]")
    print(f"[INFO] output_dir = {output_dir}")
    print(f"[INFO] timestamp = {timestamp}")
    print(f"[INFO] signal_mode = {signal_mode}")
    print(f"[INFO] tau = {tau:.6e}")
    print(f"[INFO] Tt_dense = {Tt:.6e}")
    print(f"[INFO] fs = {fs:.6e}")
    print(f"[INFO] n_samples = {n_samples}")
    print(f"[INFO] B_sensor = {budget['B_sensor']:.6e}")
    print(f"[INFO] SNR_sensor = {budget['SNR_sensor']:.6e}")
    print(f"[INFO] capacity_sensor = {budget['capacity_sensor']:.6e}")
    print(f"[INFO] bits_per_cycle = {budget['bits_per_cycle']:.6e}")
    print(f"[INFO] measurement_bits = {measurement_bits}")
    print(f"[INFO] n_measurements = {n_measurements}")
    print(f"[INFO] sparsity_eff = {sparsity_eff}")
    print(f"[INFO] quantize_measurements = {quantize_measurements}")

    x_sampling_ideal_dense = sinc_reconstruct_from_samples(
        x_samples=x_samples,
        t_eval=t_dense,
        fs=fs
    )

    acq_ideal, rec_ideal, x_hat_ideal_samples = run_cs_core_once(
        x_samples=x_samples,
        t_samples=t_samples,
        cs_cfg=cs_cfg,
        n_measurements=n_measurements,
        sparsity_eff=sparsity_eff,
        quantize_measurements=False,
    )

    x_hat_ideal_dense = sinc_reconstruct_from_samples(
        x_samples=x_hat_ideal_samples,
        t_eval=t_dense,
        fs=fs
    )

    y_real = extract_measurements_real(acq_ideal)

    acq_quantized = None
    rec_quantized = None
    x_hat_quantized_samples = None
    x_hat_quantized_dense = None
    q_indices = None
    y_dequantized = None
    quantization_metadata = {"enabled": False}

    if quantize_measurements:
        acq_quantized, rec_quantized, x_hat_quantized_samples = run_cs_core_once(
            x_samples=x_samples,
            t_samples=t_samples,
            cs_cfg=cs_cfg,
            n_measurements=n_measurements,
            sparsity_eff=sparsity_eff,
            quantize_measurements=True,
        )

        x_hat_quantized_dense = sinc_reconstruct_from_samples(
            x_samples=x_hat_quantized_samples,
            t_eval=t_dense,
            fs=fs
        )

        y_real = extract_measurements_real(acq_quantized)
        q_indices = extract_quantized_indices(acq_quantized)
        y_dequantized = extract_measurements_dequantized(acq_quantized)
        quantization_metadata = metadata_from_quantization(acq_quantized)

    coeff_true = safe_dct_1d(x_samples)
    coeff_ideal = safe_dct_1d(x_hat_ideal_samples)

    if x_hat_quantized_samples is not None:
        coeff_quantized = safe_dct_1d(x_hat_quantized_samples)
    else:
        coeff_quantized = None

    mse_sampling_ideal = float(np.mean((x_dense - x_sampling_ideal_dense) ** 2))
    mse_ideal_cs = float(np.mean((x_dense - x_hat_ideal_dense) ** 2))

    if x_hat_quantized_dense is not None:
        mse_quantized_cs = float(np.mean((x_dense - x_hat_quantized_dense) ** 2))
    else:
        mse_quantized_cs = None

    measurement_quantization_mse = None
    if y_real is not None and y_dequantized is not None:
        measurement_quantization_mse = float(np.mean((y_real - y_dequantized) ** 2))

    print(f"[INFO] mse_sampling_ideal = {mse_sampling_ideal:.8e}")
    print(f"[INFO] mse_ideal_cs = {mse_ideal_cs:.8e}")

    if mse_quantized_cs is not None:
        print(f"[INFO] mse_quantized_cs = {mse_quantized_cs:.8e}")

    if measurement_quantization_mse is not None:
        print(f"[INFO] measurement_quantization_mse = {measurement_quantization_mse:.8e}")

    formats = figure_formats(cfg)
    generated_files: Dict[str, str] = {}

    generated_files.update(
        plot_signal_reconstruction(
            t_dense=t_dense,
            x_dense=x_dense,
            t_samples=t_samples,
            x_samples=x_samples,
            x_sampling_ideal_dense=x_sampling_ideal_dense,
            x_hat_ideal_dense=x_hat_ideal_dense,
            x_hat_quantized_dense=x_hat_quantized_dense,
            output_dir=output_dir,
            timestamp=timestamp,
            formats=formats,
        )
    )

    generated_files.update(
        plot_measurements(
            y_real=y_real,
            q_indices=q_indices,
            y_dequantized=y_dequantized,
            output_dir=output_dir,
            timestamp=timestamp,
            formats=formats,
        )
    )

    generated_files.update(
        plot_topk_coefficients(
            coeff_true=coeff_true,
            coeff_ideal=coeff_ideal,
            coeff_quantized=coeff_quantized,
            top_k=top_k,
            output_dir=output_dir,
            timestamp=timestamp,
            formats=formats,
        )
    )

    generated_files.update(
        plot_accumulated_energy(
            coeff_true=coeff_true,
            output_dir=output_dir,
            timestamp=timestamp,
            formats=formats,
        )
    )

    generated_files.update(
        plot_error(
            t_dense=t_dense,
            x_dense=x_dense,
            x_sampling_ideal_dense=x_sampling_ideal_dense,
            x_hat_ideal_dense=x_hat_ideal_dense,
            x_hat_quantized_dense=x_hat_quantized_dense,
            output_dir=output_dir,
            timestamp=timestamp,
            formats=formats,
        )
    )

    generated_files.update(
        plot_channel_budget_panel(
            params=params,
            budget=budget,
            fs=fs,
            n_samples=n_samples,
            measurement_bits=measurement_bits,
            n_measurements=n_measurements,
            sparsity_eff=sparsity_eff,
            quantize_measurements=quantize_measurements,
            output_dir=output_dir,
            timestamp=timestamp,
            formats=formats,
        )
    )

    arrays_path = save_npz_payload(
        output_dir=output_dir,
        timestamp=timestamp,
        t_dense=t_dense,
        x_raw=x_raw,
        x_filtered=x_filtered,
        x_dense=x_dense,
        t_samples=t_samples,
        x_samples=x_samples,
        x_sampling_ideal_dense=x_sampling_ideal_dense,
        x_hat_ideal_samples=x_hat_ideal_samples,
        x_hat_ideal_dense=x_hat_ideal_dense,
        x_hat_quantized_samples=(
            x_hat_quantized_samples
            if x_hat_quantized_samples is not None
            else np.array([])
        ),
        x_hat_quantized_dense=(
            x_hat_quantized_dense
            if x_hat_quantized_dense is not None
            else np.array([])
        ),
        y_real=(
            y_real
            if y_real is not None
            else np.array([])
        ),
        q_indices=(
            q_indices
            if q_indices is not None
            else np.array([])
        ),
        y_dequantized=(
            y_dequantized
            if y_dequantized is not None
            else np.array([])
        ),
        coeff_true=coeff_true,
        coeff_ideal=coeff_ideal,
        coeff_quantized=(
            coeff_quantized
            if coeff_quantized is not None
            else np.array([])
        ),
    )
    generated_files["arrays_npz"] = arrays_path

    config_copy = save_config_copy(config_path, output_dir, timestamp)
    generated_files["config"] = config_copy

    metadata = {
        "created_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "output_dir": output_dir,
        "git_commit": get_git_commit(),
        "signal_mode": signal_mode,
        "physical_model": {
            "P": float(params.P),
            "N0": float(params.N0),
            "B_total": float(params.B),
            "B_sensor": float(budget["B_sensor"]),
            "SNR_sensor": float(budget["SNR_sensor"]),
            "SNR_sensor_dB": float(budget["SNR_sensor_dB"]),
            "capacity_sensor": float(budget["capacity_sensor"]),
            "bits_per_cycle": float(budget["bits_per_cycle"]),
        },
        "sampling": {
            "fs": float(fs),
            "num_samples": int(n_samples),
            "dense_Tt": float(Tt),
            "dense_num_samples": int(len(t_dense)),
        },
        "cs": {
            "measurement_bits": int(measurement_bits),
            "n_measurements": int(n_measurements),
            "sparsity_requested": int(sparsity_cfg),
            "sparsity_effective": int(sparsity_eff),
            "basis": cs_cfg.get("basis", "dct"),
            "sensing_matrix": cs_cfg.get("sensing_matrix", "gaussian"),
            "quantize_measurements": bool(quantize_measurements),
            "measurement_quantization_mse": measurement_quantization_mse,
            "quantization_metadata": quantization_metadata,
        },
        "metrics": {
            "mse_sampling_ideal": mse_sampling_ideal,
            "mse_ideal_cs": mse_ideal_cs,
            "mse_quantized_cs": mse_quantized_cs,
        },
        "generated_files": generated_files,
    }

    metadata_path = save_metadata(metadata, output_dir, timestamp)
    generated_files["metadata"] = metadata_path

    print("\n[INFO] Generated files:")
    for key, path in generated_files.items():
        print(f"       {key}: {path}")

    if show_plots:
        plt.show()

    return {
        "t_dense": t_dense,
        "x_dense": x_dense,
        "t_samples": t_samples,
        "x_samples": x_samples,
        "x_sampling_ideal_dense": x_sampling_ideal_dense,
        "x_hat_ideal_samples": x_hat_ideal_samples,
        "x_hat_ideal_dense": x_hat_ideal_dense,
        "x_hat_quantized_samples": x_hat_quantized_samples,
        "x_hat_quantized_dense": x_hat_quantized_dense,
        "y_real": y_real,
        "q_indices": q_indices,
        "y_dequantized": y_dequantized,
        "mse_sampling_ideal": mse_sampling_ideal,
        "mse_ideal_cs": mse_ideal_cs,
        "mse_quantized_cs": mse_quantized_cs,
        "budget": budget,
        "output_dir": output_dir,
        "timestamp": timestamp,
        "generated_files": generated_files,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run single-trial CS + FDMA debug."
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

    run_debug_cs_fdma_single_trial(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()
