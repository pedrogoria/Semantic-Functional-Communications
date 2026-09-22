"""
tests/debug_fair_methods_comparison.py

Interactive debug/visualization tool for the fair-methods comparison setup.

Use from PyCharm Python Console:

from tests.debug_fair_methods_comparison import run_debug_fair_methods_comparison

out = run_debug_fair_methods_comparison(
    "tests/configs/debug_fair_methods_comparison.yaml"
)

This script does not save result tables. It generates explanatory plots.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sfc.core.acquisition.cs import CSAcquisitionCore
from sfc.core.acquisition.fri import FRIAcquisition
from sfc.core.acquisition.sod import SoDAcquisitionCore
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.channel.physical_channel import apply_awgn
from sfc.core.filters import filter_periodic, sinc_reconstruct_from_samples
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.mac.fdma import FDMACore
from sfc.core.modulation.ppm import PPMCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize, quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.semantic_error_detection import detect_semantic_errors
from sfc.core.system_parameters import build_derived_system_parameters, compute_benchmark_M_per_sensor
from sfc.core.theory import compute_N0, compute_sensor_snr, compute_snr_db, compute_snr_linear

_THIS_FILE = Path(__file__).resolve()
_TESTS_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _TESTS_DIR.parent

DEFAULT_DEBUG_CONFIG = _TESTS_DIR / "configs" / "debug_fair_methods_comparison.yaml"


def _resolve_B_values_from_cfg(cfg: dict) -> np.ndarray:
    """
    Resolve B sweep values.

    Supported formats
    -----------------
    Explicit list:

        sweep:
          B:
            values: [1000.0, 2500.0, 5000.0]

    Range format:

        sweep:
          B:
            start: 1000.0
            stop: 50001.0
            step: 10000.0
            include_stop: true
    """
    b_cfg = cfg["sweep"]["B"]

    if "values" in b_cfg and b_cfg["values"] is not None:
        values = np.asarray(b_cfg["values"], dtype=float).reshape(-1)

        if values.size < 1:
            raise ValueError("sweep.B.values must contain at least one value.")

        if np.any(~np.isfinite(values)):
            raise ValueError("sweep.B.values contains non-finite values.")

        if np.any(values <= 0):
            raise ValueError("All sweep.B.values must be positive.")

        return values

    b_start = float(b_cfg["start"])
    b_stop = float(b_cfg["stop"])
    b_step = float(b_cfg["step"])
    include_stop = bool(b_cfg.get("include_stop", True))

    if b_step <= 0:
        raise ValueError("sweep.B.step must be positive.")

    if include_stop:
        return np.arange(b_start, b_stop + 0.5 * b_step, b_step)

    return np.arange(b_start, b_stop, b_step)


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path,
                Path.cwd() / path,
                _PROJECT_ROOT / path,
                _TESTS_DIR / path,
            ]
        )

    resolved_path = None

    for candidate in candidates:
        if candidate.is_file():
            resolved_path = candidate
            break

    if resolved_path is None:
        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "YAML file not found. Searched:\n"
            f"{searched}"
        )

    with resolved_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML file is empty: {resolved_path}")

    return cfg


def _resolve_B_values_from_cfg(cfg: dict) -> np.ndarray:
    """
    Resolve B sweep values from either explicit values or start/stop/step.
    """
    b_cfg = cfg["sweep"]["B"]

    if "values" in b_cfg and b_cfg["values"] is not None:
        values = np.asarray(b_cfg["values"], dtype=float).reshape(-1)

        if values.size < 1:
            raise ValueError("sweep.B.values must contain at least one value.")

        if np.any(~np.isfinite(values)):
            raise ValueError("sweep.B.values contains non-finite values.")

        if np.any(values <= 0):
            raise ValueError("All sweep.B.values must be positive.")

        return values

    b_start = float(b_cfg["start"])
    b_stop = float(b_cfg["stop"])
    b_step = float(b_cfg["step"])
    include_stop = bool(b_cfg.get("include_stop", True))

    if b_step <= 0:
        raise ValueError("sweep.B.step must be positive.")

    if include_stop:
        return np.arange(b_start, b_stop + 0.5 * b_step, b_step)

    return np.arange(b_start, b_stop, b_step)


def _resolve_b_value(base_cfg: dict, debug_cfg: dict) -> float:
    dbg = debug_cfg.get("debug", {})

    if dbg.get("B_value", None) is not None:
        return float(dbg["B_value"])

    b_values = _resolve_B_values_from_cfg(base_cfg)

    idx = int(dbg.get("B_index", 0))

    if idx < 0 or idx >= len(b_values):
        raise IndexError(
            f"debug.B_index={idx} out of range for {len(b_values)} B points. "
            f"Available B values: {b_values.tolist()}"
        )

    return float(b_values[idx])


def _apply_power_model_local(cfg: dict):
    mode = cfg.get("power_model", {}).get("mode", "fixed_P_and_N0")
    if mode == "fixed_P_and_N0":
        P = float(cfg["system"]["P"])
        B = float(cfg["system"]["B"])
        N0 = float(cfg["system"]["N0"])
        snr = compute_sensor_snr(P=P, B_sensor=B, N0=N0)
        cfg["system"]["SNR_dB"] = compute_snr_db(snr)
        return
    if mode == "fixed_P_and_SNR":
        P = float(cfg["system"]["P"])
        B = float(cfg["system"]["B"])
        SNR_dB = float(cfg["system"]["SNR_dB"])
        snr = compute_snr_linear(SNR_dB)
        cfg["system"]["N0"] = compute_N0(P=P, B=B, SNR=snr)
        return
    raise ValueError(f"Unsupported power_model.mode: {mode}")


def _prepare_cfg(debug_cfg: dict):
    exp_path = debug_cfg.get("experiment_config_path", "experiments/configs/figures/fair_methods_comparison_vs_B.yaml")
    cfg = copy.deepcopy(_load_yaml(exp_path))
    cfg["system"]["B"] = _resolve_b_value(cfg, debug_cfg)
    seed = int(debug_cfg.get("debug", {}).get("seed", cfg.get("monte_carlo", {}).get("seed", 47)))
    cfg.setdefault("monte_carlo", {})["seed"] = seed
    cfg.setdefault("reproducibility", {})["seed"] = seed
    _apply_power_model_local(cfg)
    params = build_derived_system_parameters(cfg)
    N = int(cfg["signal"].get("N_override", params.N))
    t = np.arange(0.0, float(params.tau), float(cfg["signal"]["Tt"]))
    rng = np.random.default_rng(seed)
    return cfg, debug_cfg, params, N, t, rng


def _method_cfg(cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    merged = {}
    merged.update(cfg.get(method, {}))
    merged.update(cfg.get("acquisition", {}).get(method, {}))
    return merged


def _safe_log2_positive(value) -> float:
    value = int(value)
    value = max(value, 1)
    return float(np.log2(float(value)))


def _format_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _axis_cfg(debug_cfg: dict, figure_key: str, axis_key: str) -> dict:
    plot_axes = debug_cfg.get("plot_axes", {}) or {}
    out = {}
    out.update(plot_axes.get("default", {}) or {})
    out.update((plot_axes.get(figure_key, {}) or {}).get(axis_key, {}) or {})
    return out


def _apply_axis_cfg(ax, debug_cfg: dict, figure_key: str, axis_key: str):
    cfg = _axis_cfg(debug_cfg, figure_key, axis_key)
    if cfg.get("xscale", None) is not None:
        ax.set_xscale(str(cfg["xscale"]))
    if cfg.get("yscale", None) is not None:
        ax.set_yscale(str(cfg["yscale"]))
    if cfg.get("xlim", None) is not None:
        ax.set_xlim(float(cfg["xlim"][0]), float(cfg["xlim"][1]))
    if cfg.get("ylim", None) is not None:
        ax.set_ylim(float(cfg["ylim"][0]), float(cfg["ylim"][1]))
    if cfg.get("grid", None) is not None:
        ax.grid(bool(cfg["grid"]), which="both", linestyle=":")


def _figure(title: str, nrows: int, figsize=(11, 8)):
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex="none")
    if nrows == 1:
        axes = [axes]
    fig.suptitle(title)
    return fig, axes


def _finish(fig, debug_cfg: dict):
    fig.tight_layout()
    if bool(debug_cfg.get("debug", {}).get("show", True)):
        plt.show()


def _sensor_period(debug_cfg: dict) -> Tuple[int, int]:
    dbg = debug_cfg.get("debug", {})
    return int(dbg.get("period", 0)), int(dbg.get("sensor", 0))


def _text_panel(ax, title: str, lines: List[str]):
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", family="monospace", fontsize=10)


def _psd_one_sided(x: np.ndarray, dt: float):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.array([0.0]), np.array([0.0])
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(n, d=dt)
    psd = (np.abs(X) ** 2) * dt / max(np.sum(w ** 2), np.finfo(float).eps)
    return f, psd


def _build_fdma(params):
    fdma = FDMACore(
        S=params.S,
        B_total=params.B,
        P_per_sensor=params.P,
        tau=params.tau,
        bandwidth_allocation=params.bandwidth_allocation,
        N0=params.N0,
        normalize_sensor_power=False,
        return_nonorthogonal_sum_preview=False,
    )
    capacity = np.asarray(fdma.get_capacity_per_sensor(), dtype=float)
    budget_bits = np.floor(capacity * params.tau).astype(int)
    return fdma, capacity, budget_bits


def _normalize_tensor_power(x, target_power):
    x = np.asarray(x, dtype=float)
    p = np.mean(x ** 2)
    if np.isclose(p, 0.0):
        return np.array(x, copy=True)
    return x * np.sqrt(float(target_power) / p)


def _resolve_benchmark_sampling_rate(cfg, params):
    benchmark_cfg = cfg.get("benchmark", {})
    nyq_cfg = cfg.get("acquisition", {}).get("nyquist", {})
    sampling_rate = float(nyq_cfg.get("sampling_rate", benchmark_cfg.get("sampling_rate", params.W)))
    factor = float(nyq_cfg.get("effective_rate_factor", benchmark_cfg.get("effective_rate_factor", 2.0)))
    fs = sampling_rate * factor
    if fs <= 0:
        raise ValueError("Resolved sampling rate must be positive.")
    return fs


def _sample_signal_tensor_uniform(x_ref, t_dense, tau, fs):
    t_samples = np.arange(0.0, tau, 1.0 / fs)
    _, n_periods, S = x_ref.shape
    x_samples = np.zeros((len(t_samples), n_periods, S), dtype=float)
    for p in range(n_periods):
        for s in range(S):
            x_samples[:, p, s] = np.interp(t_samples, t_dense, x_ref[:, p, s])
    return t_samples, x_samples


def generate_raw_and_processed_source(cfg, params, N, t, rng, n_periods):
    _ = N
    n_time = len(t)
    S = int(params.S)
    dist = cfg["signal"].get("distribution", "gaussian")
    if dist == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=(n_time, n_periods, S))
    elif dist == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=(n_time, n_periods, S))
    else:
        raise ValueError("signal.distribution must be uniform or gaussian.")
    W_filter = float(cfg["signal"].get("W", params.W))
    Tt = float(cfg["signal"]["Tt"])
    tau = float(params.tau)
    x_filtered = np.zeros_like(x_raw)
    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(x_raw[:, p, s], W_filter, Tt, tau)
    if cfg.get("dc", {}).get("enabled", False):
        x_processed = np.array(x_filtered, copy=True)
    else:
        x_processed = np.zeros_like(x_filtered)
        for p in range(n_periods):
            for s in range(S):
                dc = Tt * np.sum(x_filtered[:, p, s]) / tau
                x_processed[:, p, s] = x_filtered[:, p, s] - dc
    p2p_target = cfg["signal"].get("peak_to_peak", 0.0)
    if p2p_target != 0:
        p2p_target = float(p2p_target)
        for p in range(n_periods):
            for s in range(S):
                p2p = np.max(x_processed[:, p, s]) - np.min(x_processed[:, p, s])
                if p2p != 0:
                    x_processed[:, p, s] *= p2p_target / p2p
    return x_raw, x_processed


def plot_source_debug(cfg, debug_cfg, params, t, x_raw, x_processed):
    p, s = _sensor_period(debug_cfg)
    f, psd = _psd_one_sided(x_processed[:, p, s], float(cfg["signal"]["Tt"]))
    fig, axes = _figure("Source debug: raw, simulation signal, PSD", 3, figsize=(11, 8))
    axes[0].plot(t, x_raw[:, p, s])
    axes[0].set_title("Random raw signal before filtering")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("amplitude")
    _apply_axis_cfg(axes[0], debug_cfg, "source", "raw")
    axes[1].plot(t, x_processed[:, p, s])
    axes[1].set_title("Simulation signal after filtering, DC handling, and final p2p scaling")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("amplitude")
    _apply_axis_cfg(axes[1], debug_cfg, "source", "processed")
    axes[2].semilogy(f, psd + 1e-30)
    axes[2].axvline(float(cfg["signal"].get("W", params.W)), color="C3", linestyle="--", label="signal.W")
    axes[2].set_title("PSD of processed simulation/baseband signal")
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylabel("PSD")
    axes[2].legend()
    _apply_axis_cfg(axes[2], debug_cfg, "source", "psd")
    _finish(fig, debug_cfg)


def debug_benchmark_nyquist(cfg, debug_cfg, params, t, x_ref, budget_bits):
    p, s = _sensor_period(debug_cfg)
    fs = _resolve_benchmark_sampling_rate(cfg, params)
    ts, xs = _sample_signal_tensor_uniform(x_ref, t, params.tau, fs)
    M_vec = compute_benchmark_M_per_sensor(
        S=params.S,
        tau=params.tau,
        B=params.B,
        P=params.P,
        N0=params.N0,
        sampling_rate=fs,
        bandwidth_allocation=params.bandwidth_allocation,
        force_power_of_two=params.quantization_force_power_of_two,
        rounding_mode=params.quantization_rounding_mode,
    )
    M = int(M_vec[s])
    nyq_cfg = _method_cfg(cfg, "nyquist")
    peak2peak = nyq_cfg.get("peak2peak", cfg["signal"].get("peak_to_peak", "sample"))
    if peak2peak == "sample":
        vmin = float(np.min(xs[:, p, s]))
        vmax = float(np.max(xs[:, p, s]))
    else:
        p2p = float(peak2peak)
        vmin, vmax = -0.5 * p2p, 0.5 * p2p
    xsq = np.array(xs, copy=True)
    xsq[:, p, s] = quantize(xs[:, p, s], vmin, vmax, M, poss=0.5)
    x_hat = sinc_reconstruct_from_samples(xsq, ts, t, params.tau, 1.0 / fs, periodic_replicas=int(nyq_cfg.get("periodic_replicas", 10)))
    mse = float(np.mean((x_ref[:, p, s] - x_hat[:, p, s]) ** 2))
    fig, axes = _figure("Benchmark / Nyquist + FDMA debug", 4, figsize=(11, 10))
    axes[0].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[0].plot(ts, xs[:, p, s], "o", markersize=4, label="uniform samples")
    axes[0].set_title(f"Uniform sampling at fs={fs:.3g} Hz")
    axes[0].legend()
    _apply_axis_cfg(axes[0], debug_cfg, "benchmark", "samples")
    axes[1].plot(ts, xs[:, p, s], "o-", label="samples")
    axes[1].plot(ts, xsq[:, p, s], "s--", label=f"quantized samples, M={M}")
    axes[1].set_title(f"Scalar quantization range [{vmin:.3g}, {vmax:.3g}], FDMA budget bits sensor={budget_bits[s]}")
    axes[1].legend()
    _apply_axis_cfg(axes[1], debug_cfg, "benchmark", "quantization")
    axes[2].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[2].plot(t, x_hat[:, p, s], "--", label=f"Nyquist reconstructed, MSE={mse:.3e}")
    axes[2].set_title("Reconstruction from quantized samples")
    axes[2].legend()
    _apply_axis_cfg(axes[2], debug_cfg, "benchmark", "reconstruction")
    _text_panel(axes[3], "Benchmark/Nyquist FDMA bit-budget diagnostics", [
        f"B_s                        = {float(params.B_per_sensor[s]):.6e} Hz",
        f"FDMA budget                = {float(budget_bits[s]):.6e} bits/period/sensor",
        f"sampling rate fs           = {fs:.6e} Hz",
        f"samples per period         = {len(ts)}",
        f"quantizer levels M         = {M}",
        f"bits per sample log2(M)    = {_safe_log2_positive(M):.6e}",
        f"quantization range         = [{vmin:.6e}, {vmax:.6e}]",
        f"reconstruction MSE         = {mse:.6e}",
    ])
    _finish(fig, debug_cfg)


def debug_cs(cfg, debug_cfg, params, t, x_ref, capacity_per_sensor):
    p, s = _sensor_period(debug_cfg)
    cs_cfg = _method_cfg(cfg, "cs")
    fs = _resolve_benchmark_sampling_rate(cfg, params)
    ts, xs = _sample_signal_tensor_uniform(x_ref, t, params.tau, fs)
    n_samples = xs.shape[0]
    bits_per_cycle = float(capacity_per_sensor[s] * params.tau)
    full_bits = int(np.floor(bits_per_cycle / n_samples))
    if full_bits >= 1:
        n_measurements, measurement_bits = n_samples, full_bits
    else:
        measurement_bits = 1
        n_measurements = max(1, min(int(np.floor(bits_per_cycle)), n_samples))
    sparsity = min(int(cs_cfg.get("sparsity", n_measurements)), n_measurements, n_samples)
    cs = CSAcquisitionCore(
        n_measurements=n_measurements,
        sparsity=sparsity,
        basis=cs_cfg.get("basis", "dct"),
        sensing_matrix=cs_cfg.get("sensing_matrix", cs_cfg.get("measurement_matrix", "gaussian")),
        random_state=cs_cfg.get("random_state", cs_cfg.get("seed", 12345)),
        normalize_dictionary_columns=cs_cfg.get("normalize_dictionary_columns", True),
        store_true_representation=cs_cfg.get("store_true_representation", False),
        quantize_measurements=bool(cs_cfg.get("quantize_measurements", cs_cfg.get("quantize", True))),
        measurement_bits=measurement_bits,
        quantization_mode=cs_cfg.get("measurement_quantization_mode", "uniform_midrise"),
        quantization_range=cs_cfg.get("measurement_quantization_range", "per_signal"),
        measurement_quantization_min=cs_cfg.get("measurement_quantization_min", None),
        measurement_quantization_max=cs_cfg.get("measurement_quantization_max", None),
        clip_quantization=cs_cfg.get("clip_quantization", True),
    )
    acq = cs.acquire(x=xs[:, p:p + 1, s:s + 1], t=ts)
    rec = cs.reconstruct(acq)
    x_hat_samples = np.zeros_like(xs)
    x_hat_samples[:, p:p + 1, s:s + 1] = rec.reconstructed_signal
    x_hat = sinc_reconstruct_from_samples(x_hat_samples, ts, t, params.tau, 1.0 / fs, periodic_replicas=10)
    y = getattr(acq, "measurements", None)
    if y is None and isinstance(acq, dict):
        y = acq.get("measurements", None)
    y_vec = np.asarray(y).reshape(-1) if y is not None else np.array([])
    mse = float(np.mean((x_ref[:, p, s] - x_hat[:, p, s]) ** 2))
    fig, axes = _figure("CS + FDMA debug", 4, figsize=(11, 10))
    axes[0].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[0].plot(ts, xs[:, p, s], "o", markersize=4, label="uniform samples x[n]")
    axes[0].set_title(f"CS input samples: n={n_samples}, fs={fs:.3g} Hz")
    axes[0].legend()
    _apply_axis_cfg(axes[0], debug_cfg, "cs", "samples")
    axes[1].stem(np.arange(len(y_vec)), y_vec, basefmt=" ")
    axes[1].set_title(f"CS measurement vector y = Phi x | M={n_measurements}, bits/measurement={measurement_bits}, sparsity={sparsity}")
    _apply_axis_cfg(axes[1], debug_cfg, "cs", "measurements")
    axes[2].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[2].plot(t, x_hat[:, p, s], "--", label=f"CS reconstructed, MSE={mse:.3e}")
    axes[2].set_title("CS reconstruction after sample-domain recovery")
    axes[2].legend()
    _apply_axis_cfg(axes[2], debug_cfg, "cs", "reconstruction")
    _text_panel(axes[3], "CS FDMA bit-budget diagnostics", [
        f"B_s                        = {float(params.B_per_sensor[s]):.6e} Hz",
        f"FDMA capacity C_s          = {float(capacity_per_sensor[s]):.6e} bits/s",
        f"FDMA budget C_s*tau        = {bits_per_cycle:.6e} bits/period/sensor",
        f"uniform sampling rate fs   = {fs:.6e} Hz",
        f"input samples per period   = {n_samples}",
        f"CS measurements            = {n_measurements}",
        f"measurement bits           = {measurement_bits}",
        f"effective sparsity         = {sparsity}",
        f"quantize measurements      = {_format_bool(cs.quantize_measurements)}",
        f"reconstruction MSE         = {mse:.6e}",
    ])
    _finish(fig, debug_cfg)


# PPM helpers and SoD helpers shared with pipeline kept minimal below.
# To keep this debug file stable, only local implementations are used.

def _resolve_ppm_timing_from_B_sensor(ppm_cfg, params, B_sensor, Tt=None):
    pulse_type = str(ppm_cfg.get("pulse_type", "raised_cosine")).lower()
    rolloff = float(ppm_cfg.get("rolloff", 0.99))
    requested_fs_msg = ppm_cfg.get("fs_msg", params.W)
    if requested_fs_msg is None or (isinstance(requested_fs_msg, str) and requested_fs_msg.lower() == "auto"):
        requested_fs_msg = params.W
    requested_fs_msg = float(requested_fs_msg)
    if pulse_type in {"raised_cosine", "root_raised_cosine", "rrc", "rc"}:
        max_fs_msg_from_B_sensor = 2.0 * B_sensor / (1.0 + rolloff)
    else:
        max_fs_msg_from_B_sensor = B_sensor
    if not bool(ppm_cfg.get("enforce_bandwidth_from_B_sensor", True)):
        max_fs_msg_from_B_sensor = np.inf
    min_samples = int(ppm_cfg.get("min_samples_per_symbol", 8))
    max_fs_grid = 1.0 / (min_samples * float(Tt)) if Tt is not None else np.inf
    max_frac = float(ppm_cfg.get("max_pulse_width_fraction_of_symbol", 0.9))
    if ppm_cfg.get("pulse_width", None) is not None:
        pulse_width = float(ppm_cfg["pulse_width"])
        max_fs_pw = max_frac / pulse_width
    else:
        max_fs_pw = np.inf
        pulse_width = None
    fs_msg = min(requested_fs_msg, max_fs_msg_from_B_sensor, max_fs_grid, max_fs_pw)
    if pulse_width is None:
        pulse_width = float(ppm_cfg.get("pulse_width_fraction", 0.1)) / fs_msg
    pw_factor = float(ppm_cfg.get("pulse_width_bandwidth_factor", 1.0))
    if bool(ppm_cfg.get("enforce_pulse_width_from_B_sensor", True)):
        min_pw = pw_factor / float(B_sensor)
        if pulse_width < min_pw:
            if ppm_cfg.get("pulse_width", None) is not None:
                raise ValueError(f"ppm.pulse_width={pulse_width:.6e} too small for B_sensor={B_sensor:.6e}; minimum={min_pw:.6e}")
            pulse_width = min_pw
            fs_msg = min(fs_msg, max_frac / pulse_width)
    if pulse_width >= 1.0 / fs_msg:
        raise ValueError("Resolved PPM pulse_width does not fit inside symbol period.")
    return {"fs_msg": float(fs_msg), "pulse_width": float(pulse_width), "B_sensor": float(B_sensor)}


def _make_ppm_core(ppm_cfg, fs_msg, pulse_width, B_sensor):
    kwargs = dict(
        fc=fs_msg,
        pulse_width=pulse_width,
        rec_pulse=ppm_cfg.get("rec_pulse", 0.0),
        pulse_type=ppm_cfg.get("pulse_type", "raised_cosine"),
        rolloff=ppm_cfg.get("rolloff", 0.99),
        span=ppm_cfg.get("span", 12),
        eps_margin=ppm_cfg.get("eps_margin", 1e-3),
        interp_mode=ppm_cfg.get("interp_mode", "sinc"),
        periodic_replicas=ppm_cfg.get("periodic_replicas", 10),
        clip_recovered_to_unit_interval=ppm_cfg.get("clip_recovered_to_unit_interval", True),
    )
    try:
        return PPMCore(**kwargs, B_sensor=B_sensor, enforce_bandwidth_warning=ppm_cfg.get("enforce_bandwidth_warning", True),
                       pulse_width_bandwidth_factor=ppm_cfg.get("pulse_width_bandwidth_factor", 1.0))
    except TypeError:
        return PPMCore(**kwargs)


def debug_ppm(cfg, debug_cfg, params, t, x_ref, rng):
    p, s = _sensor_period(debug_cfg)
    ppm_cfg = _method_cfg(cfg, "ppm")
    B_sensor = float(params.B_per_sensor[s])
    timing = _resolve_ppm_timing_from_B_sensor(ppm_cfg, params, B_sensor, float(cfg["signal"]["Tt"]))
    fs_msg = timing["fs_msg"]
    pulse_width = timing["pulse_width"]
    x_sensor = x_ref[:, p:p + 1, s:s + 1]
    t_msg = np.arange(0.0, params.tau, 1.0 / fs_msg)
    x_msg = np.interp(t_msg, t, x_sensor[:, 0, 0])
    ppm = _make_ppm_core(ppm_cfg, fs_msg, pulse_width, B_sensor)
    mod = ppm.modulate(x=x_sensor, t=t)
    tx = np.array(mod.tx_waveform, dtype=float, copy=True)
    if ppm_cfg.get("normalize_sensor_power", True):
        tx = _normalize_tensor_power(tx, params.P)
    add_awgn = bool(debug_cfg.get("debug", {}).get("add_awgn", True))
    rx = apply_awgn(tx, params.N0, complex_noise=np.iscomplexobj(tx), rng=rng) if add_awgn else tx.copy()
    demod = ppm.demodulate(y=rx, t=t, modulation_result=mod, reconstruct_continuous=True)
    x_hat = demod.recovered_continuous
    mse = float(np.mean((x_sensor - x_hat) ** 2))
    ftx, psd_tx = _psd_one_sided(tx[:, 0, 0], float(cfg["signal"]["Tt"]))
    frx, psd_rx = _psd_one_sided(rx[:, 0, 0], float(cfg["signal"]["Tt"]))
    fig, axes = _figure("PPM + FDMA debug | AWGN on" if add_awgn else "PPM + FDMA debug | AWGN off", 4, figsize=(11, 10))
    axes[0].plot(t, x_sensor[:, 0, 0], label="simulation signal")
    axes[0].plot(t_msg, x_msg, "o", markersize=4, label="PPM message samples")
    axes[0].set_title(f"PPM sampling: fs_msg={fs_msg:.3g} Hz, pulse_width={pulse_width:.3g} s, B_sensor={B_sensor:.3g} Hz")
    axes[0].legend()
    _apply_axis_cfg(axes[0], debug_cfg, "ppm", "samples")
    axes[1].plot(t, tx[:, 0, 0], label="TX PPM waveform")
    if add_awgn:
        axes[1].plot(t, rx[:, 0, 0], "--", alpha=0.8, label="RX PPM waveform + AWGN")
    symbol_period = 1.0 / fs_msg
    boundaries = np.arange(0.0, float(params.tau) + 0.5 * symbol_period, symbol_period)
    for b in boundaries:
        axes[1].axvline(b, color="0.65", linestyle=":", linewidth=0.8, alpha=0.8)
    axes[1].plot(t_msg, np.zeros_like(t_msg), "|", color="C3", markersize=10, label="PPM symbol starts")
    axes[1].set_title("PPM transmitted waveform with symbol-frame boundaries")
    axes[1].legend()
    _apply_axis_cfg(axes[1], debug_cfg, "ppm", "waveform")
    axes[2].plot(t, x_sensor[:, 0, 0], label="simulation signal")
    axes[2].plot(t, x_hat[:, 0, 0], "--", label=f"PPM reconstructed, MSE={mse:.3e}")
    axes[2].set_title("PPM demodulated reconstruction")
    axes[2].legend()
    _apply_axis_cfg(axes[2], debug_cfg, "ppm", "reconstruction")
    axes[3].semilogy(ftx, psd_tx + 1e-30, label="TX PSD")
    if add_awgn:
        axes[3].semilogy(frx, psd_rx + 1e-30, "--", label="RX PSD")
    axes[3].axvline(B_sensor, color="C3", linestyle=":", label="B_sensor")
    axes[3].set_title("PSD of transmitted/received PPM waveform")
    axes[3].set_xlabel("frequency [Hz]")
    axes[3].set_ylabel("PSD")
    axes[3].legend()
    _apply_axis_cfg(axes[3], debug_cfg, "ppm", "psd")
    _finish(fig, debug_cfg)


def _resolve_sod_fdma_payload_allocation(K_candidate, budget_bits, n_time, transmit_event_times, min_amplitude_bits=1, min_time_bits=1,
                                         time_bits_policy="balanced"):
    K_candidate = int(max(1, K_candidate))
    budget_bits = int(max(0, budget_bits))
    n_time = int(max(1, n_time))
    max_time_bits = int(np.ceil(np.log2(float(max(n_time, 1)))))
    if budget_bits <= 0:
        amp_bits = max(1, int(min_amplitude_bits))
        return 1, amp_bits, 0, 2 ** amp_bits, 1, amp_bits
    for K_tx in range(K_candidate, 0, -1):
        bits_per_event = budget_bits // K_tx
        if bits_per_event <= 0:
            continue
        amp_bits, time_bits = _split_sod_bits_per_event(bits_per_event, transmit_event_times, min_amplitude_bits, min_time_bits, max_time_bits,
                                                        time_bits_policy, K_tx)
        if amp_bits < 0 or time_bits < 0 or amp_bits + time_bits <= 0:
            continue
        payload = K_tx * (amp_bits + time_bits)
        if payload <= budget_bits:
            return K_tx, amp_bits, time_bits, 2 ** amp_bits if amp_bits > 0 else 1, 2 ** time_bits if time_bits > 0 else 1, payload
    return 1, 1, 0, 2, 1, 1


def _split_sod_bits_per_event(bits_per_event, transmit_event_times, min_amplitude_bits, min_time_bits, max_time_bits, time_bits_policy, K_tx):
    bits_per_event = int(bits_per_event)
    if bits_per_event <= 0:
        return -1, -1
    min_amplitude_bits = int(max(0, min_amplitude_bits))
    min_time_bits = int(max(0, min_time_bits))
    max_time_bits = int(max(0, max_time_bits))
    if not transmit_event_times:
        return (bits_per_event, 0) if bits_per_event >= min_amplitude_bits else (-1, -1)
    effective_min_time_bits = min_time_bits if K_tx > 1 else 0
    if bits_per_event < min_amplitude_bits + effective_min_time_bits:
        return -1, -1
    if time_bits_policy == "time_first":
        time_bits = min(max_time_bits, bits_per_event - min_amplitude_bits)
        time_bits = max(time_bits, effective_min_time_bits)
        amp_bits = bits_per_event - time_bits
    elif time_bits_policy == "amplitude_first":
        time_bits = effective_min_time_bits
        amp_bits = bits_per_event - time_bits
    else:
        time_bits = min(max_time_bits, bits_per_event // 2)
        time_bits = max(time_bits, effective_min_time_bits)
        amp_bits = bits_per_event - time_bits
        if amp_bits < min_amplitude_bits:
            amp_bits = min_amplitude_bits
            time_bits = bits_per_event - amp_bits
    if amp_bits < min_amplitude_bits or time_bits < effective_min_time_bits or time_bits > max_time_bits:
        return -1, -1
    return int(amp_bits), int(time_bits)


def _select_sod_event_indices(K_candidate, K_tx):
    K_candidate = int(max(1, K_candidate))
    K_tx = int(max(1, min(K_tx, K_candidate)))
    if K_tx == 1:
        return np.asarray([0], dtype=int)
    idx = np.linspace(0, K_candidate - 1, K_tx)
    idx = np.unique(np.round(idx).astype(int))
    if len(idx) < K_tx:
        used = set(int(v) for v in idx)
        extra = [i for i in range(K_candidate) if i not in used]
        idx = np.concatenate([idx, np.asarray(extra[:K_tx - len(idx)], dtype=int)])
        idx = np.sort(idx)
    return idx[:K_tx].astype(int)


def _resolve_sod_amplitude_range(x_ref, values, mode, fixed_min, fixed_max):
    mode = str(mode).lower()
    if mode == "fixed":
        return float(fixed_min), float(fixed_max)
    if mode in {"global", "per_signal"}:
        return float(np.min(x_ref)), float(np.max(x_ref))
    if mode == "per_events":
        return float(np.min(values)), float(np.max(values))
    raise ValueError("Unsupported SoD amplitude_quantization_range")


def _resolve_sod_time_range(t, mode, fixed_min, fixed_max):
    mode = str(mode).lower()
    t = np.asarray(t, dtype=float)
    if mode == "fixed":
        return float(fixed_min), float(fixed_max)
    if mode == "period":
        dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
        return float(t[0]), float(t[-1] + dt)
    raise ValueError("Unsupported SoD time_quantization_range")


def _quantize_sod_values(values, value_min, value_max, bins, poss, clip):
    values = np.asarray(values, dtype=float).reshape(-1)
    bins = int(max(1, bins))
    if np.isclose(value_max, value_min):
        return np.full(values.shape, float(value_min), dtype=float)
    if clip:
        values = np.clip(values, value_min, value_max)
    return np.asarray(quantize(values, value_min, value_max, bins, poss=poss), dtype=float).reshape(-1)


def debug_sod(cfg, debug_cfg, params, t, x_ref, budget_bits):
    p, s = _sensor_period(debug_cfg)
    sod_cfg = _method_cfg(cfg, "sod")
    threshold = float(sod_cfg.get("threshold", sod_cfg.get("delta", 0.1)))
    reconstruction_mode = sod_cfg.get("reconstruction_mode", "zero_order_hold")
    initial_event = bool(sod_cfg.get("initial_event", True))
    transmit_event_times = bool(sod_cfg.get("transmit_event_times", True))
    detector = SoDAcquisitionCore(threshold=threshold, initial_event=initial_event, reconstruction_mode=reconstruction_mode,
                                  transmit_event_times=transmit_event_times, quantize_amplitudes=False, amplitude_bins=None, quantize_times=False,
                                  time_bins=None, clip_quantization=bool(sod_cfg.get("clip_quantization", True)))
    acq_raw = detector.acquire(x=x_ref, t=t)
    key = f"period_{p}_sensor_{s}"
    event_times = np.asarray(acq_raw.event_times[key], dtype=float).reshape(-1)
    event_values = np.asarray(acq_raw.event_values[key], dtype=float).reshape(-1)
    if len(event_times) < 1:
        event_times = np.asarray([float(t[0])], dtype=float)
        event_values = np.asarray([float(x_ref[0, p, s])], dtype=float)
    K_candidate = len(event_times)
    budget_s = int(max(0, np.floor(budget_bits[s])))
    K_tx, amp_bits, time_bits, amp_bins, time_bins, payload_bits = _resolve_sod_fdma_payload_allocation(K_candidate, budget_s, len(t),
                                                                                                        transmit_event_times,
                                                                                                        int(sod_cfg.get("min_amplitude_bits", 1)),
                                                                                                        int(sod_cfg.get("min_time_bits", 1)),
                                                                                                        str(sod_cfg.get("time_bits_policy",
                                                                                                                        "balanced")).lower())
    selected_idx = _select_sod_event_indices(K_candidate, K_tx)
    tx_times = event_times[selected_idx]
    tx_values = event_values[selected_idx]
    amp_min, amp_max = _resolve_sod_amplitude_range(x_ref[:, p, s], tx_values, sod_cfg.get("amplitude_quantization_range", "per_signal"),
                                                    sod_cfg.get("amplitude_quantization_min", None), sod_cfg.get("amplitude_quantization_max", None))
    tx_values_q = _quantize_sod_values(tx_values, amp_min, amp_max, amp_bins, float(sod_cfg.get("quantization_poss", 0.5)),
                                       bool(sod_cfg.get("clip_quantization", True)))
    if transmit_event_times:
        tmin, tmax = _resolve_sod_time_range(t, sod_cfg.get("time_quantization_range", "period"), sod_cfg.get("time_quantization_min", None),
                                             sod_cfg.get("time_quantization_max", None))
        tx_times_q = _quantize_sod_values(tx_times, tmin, tmax, time_bins, float(sod_cfg.get("quantization_poss", 0.5)),
                                          bool(sod_cfg.get("clip_quantization", True)))
    else:
        tx_times_q = tx_times
    x_hat_1d = SoDAcquisitionCore._reconstruct_1d(tx_times_q, tx_values_q, t, reconstruction_mode)
    mse = float(np.mean((x_ref[:, p, s] - x_hat_1d) ** 2))
    fig, axes = _figure("SoD + FDMA debug", 4, figsize=(11, 10))
    axes[0].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[0].plot(event_times, event_values, "o", markersize=4, label="candidate SoD events")
    axes[0].plot(tx_times, tx_values, "s", markersize=6, label="selected transmitted events")
    axes[0].set_title(f"SoD candidate/transmitted events: K_candidate={K_candidate}, K_tx={K_tx}, threshold={threshold}")
    axes[0].legend()
    _apply_axis_cfg(axes[0], debug_cfg, "sod", "events")
    axes[1].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[1].plot(t, x_hat_1d, "--", label=f"SoD reconstructed, MSE={mse:.3e}")
    axes[1].set_title("SoD reconstruction from budget-adapted transmitted events")
    axes[1].legend()
    _apply_axis_cfg(axes[1], debug_cfg, "sod", "reconstruction")
    axes[2].stem(np.arange(K_tx), tx_values_q, basefmt=" ")
    axes[2].set_title(f"Quantized transmitted event amplitudes (amplitude_bits={amp_bits}, bins={amp_bins})")
    axes[2].set_xlabel("transmitted event index")
    axes[2].set_ylabel("quantized amplitude")
    _apply_axis_cfg(axes[2], debug_cfg, "sod", "amplitudes")
    _text_panel(axes[3], "SoD FDMA payload diagnostics", [
        f"B_s                        = {float(params.B_per_sensor[s]):.6e} Hz",
        f"threshold                  = {threshold:.6e}",
        f"candidate events           = {K_candidate}",
        f"transmitted events K_tx    = {K_tx}",
        f"FDMA budget                = {float(budget_s):.6e} bits/period/sensor",
        f"payload used               = {float(payload_bits):.6e} bits/period/sensor",
        f"amplitude bits/event       = {amp_bits}",
        f"time bits/event            = {time_bits}",
        f"amplitude bins             = {amp_bins}",
        f"time bins                  = {time_bins}",
        f"transmit event times       = {_format_bool(transmit_event_times)}",
        f"reconstruction mode        = {reconstruction_mode}",
        f"reconstruction MSE         = {mse:.6e}",
    ])
    _finish(fig, debug_cfg)


def _cap_fri_budget_bits_for_numeric_safety(budget_bits, n_time, fri_cfg):
    budget_bits = int(max(0, budget_bits))
    n_time = int(max(1, n_time))
    K = int(fri_cfg.get("K", fri_cfg.get("num_innovations", 3)))
    K = max(1, K)
    max_location_bits = int(np.ceil(np.log2(float(max(n_time, 2)))))
    max_amplitude_bits = int(fri_cfg.get("max_amplitude_bits_safe", 24))
    return int(min(budget_bits, K * (max_location_bits + max_amplitude_bits)))


def debug_fri(cfg, debug_cfg, params, t, x_ref, budget_bits):
    p, s = _sensor_period(debug_cfg)
    fri_cfg = _method_cfg(cfg, "fri")
    raw_budget_bits = int(max(0, np.floor(budget_bits[s])))
    safe_budget_bits = _cap_fri_budget_bits_for_numeric_safety(raw_budget_bits, len(t), fri_cfg)
    cfg_s = copy.deepcopy(cfg)
    cfg_s.setdefault("acquisition", {}).setdefault("fri", {})["budget_bits"] = safe_budget_bits
    fri = FRIAcquisition(cfg_s)
    x_sensor = x_ref[:, p:p + 1, s:s + 1]
    out = fri.run(x_sensor, t=t, tau=params.tau, Tt=float(cfg["signal"]["Tt"]), budget_bits=safe_budget_bits,
                  bit_error_rate=float(fri_cfg.get("bit_error_rate", cfg.get("comparison", {}).get("methods", {}).get("target_ber", 0.0))),
                  return_result=True)
    x_hat = out.x_hat
    locs = out.innovations["locations"][:, 0, 0]
    amps = out.innovations["amplitudes"][:, 0, 0]
    mse = float(np.mean((x_sensor - x_hat) ** 2))
    fig, axes = _figure("FRI-inspired + FDMA debug", 4, figsize=(11, 10))
    axes[0].plot(t, x_sensor[:, 0, 0], label="simulation signal")
    axes[0].plot(locs, amps, "o", markersize=6, label="estimated innovations")
    axes[0].set_title(f"FRI innovation estimation: K={len(locs)}, budget_bits_used={safe_budget_bits}")
    axes[0].legend()
    _apply_axis_cfg(axes[0], debug_cfg, "fri", "innovations")
    axes[1].plot(t, x_sensor[:, 0, 0], label="simulation signal")
    axes[1].plot(t, x_hat[:, 0, 0], "--", label=f"FRI reconstructed, MSE={mse:.3e}")
    axes[1].set_title("FRI sparse-innovation reconstruction")
    axes[1].legend()
    _apply_axis_cfg(axes[1], debug_cfg, "fri", "reconstruction")
    axes[2].stem(np.arange(len(amps)), amps, basefmt=" ")
    axes[2].set_title("Innovation amplitudes versus innovation index")
    axes[2].set_xlabel("innovation index")
    axes[2].set_ylabel("innovation amplitude")
    _apply_axis_cfg(axes[2], debug_cfg, "fri", "amplitudes")
    _text_panel(axes[3], "FRI-inspired FDMA bit-budget diagnostics", [
        f"B_s                        = {float(params.B_per_sensor[s]):.6e} Hz",
        f"FDMA budget raw            = {float(raw_budget_bits):.6e} bits/period/sensor",
        f"FRI budget used safely     = {float(safe_budget_bits):.6e} bits/period/sensor",
        f"K innovations              = {len(locs)}",
        f"bits_location              = {out.diagnostics.get('bits_location')}",
        f"bits_amplitude             = {out.diagnostics.get('bits_amplitude')}",
        f"reconstruction MSE         = {mse:.6e}",
    ])
    _finish(fig, debug_cfg)


def _phase_objects(cfg, params, N, x_ref):
    Tt = float(cfg["signal"]["Tt"])
    fourier = FourierCoefficientCore(T=params.tau, harmonics=N, sensor_nodes=params.S)
    an, bn, _ = fourier.calc_an_bn_dft(x_ref, Tt, normalize=cfg["signal"].get("normalize_dft", True),
                                       norm=cfg["signal"].get("normalization_target", 3.99))
    phase = PhaseCoefficientCore(T=params.tau, harmonics=N, n_sub_symbol=params.L, resource=params.R, sensor_nodes=params.S, bandwidth=params.B,
                                 detect_errors=False, periods=x_ref.shape[1], threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001))
    ta, tb = phase.calc_ta_tb(an, bn)
    return np.real(ta), np.real(tb), phase


def _reconstruct_ta_tb(ta, tb, t, params):
    w0 = 2.0 * np.pi / params.tau
    n_periods, _, S = ta.shape
    x_hat = np.zeros((len(t), n_periods, S), dtype=float)
    for p in range(n_periods):
        for s in range(S):
            x_hat[:, p, s] = recover_signal(ta[p, :, s], tb[p, :, s], t, w0)
    return x_hat


def debug_rbcp(cfg, debug_cfg, params, N, t, x_ref, use_time=False):
    p, s = _sensor_period(debug_cfg)
    M = int(params.M_time if use_time else params.M_rbcp)
    label = "RbCP_time" if use_time else "RbCP"
    figure_key = "rbcp_time" if use_time else "rbcp"
    ta, tb, _ = _phase_objects(cfg, params, N, x_ref)
    w0 = 2.0 * np.pi / params.tau
    ta_q = np.zeros_like(ta)
    tb_q = np.zeros_like(tb)
    for pp in range(ta.shape[0]):
        for ss in range(ta.shape[2]):
            ta_q[pp, :, ss], tb_q[pp, :, ss] = quantize_ta_tb(ta[pp, :, ss], tb[pp, :, ss], w0, M)
    x_hat = _reconstruct_ta_tb(ta_q, tb_q, t, params)
    mse = float(np.mean((x_ref[:, p, s] - x_hat[:, p, s]) ** 2))
    fig, axes = _figure(f"{label} debug", 4, figsize=(11, 10))
    axes[0].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[0].plot(t, x_hat[:, p, s], "--", label=f"{label} reconstructed, MSE={mse:.3e}")
    axes[0].set_title(f"{label} reconstruction with M={M}")
    axes[0].legend()
    _apply_axis_cfg(axes[0], debug_cfg, figure_key, "reconstruction")
    axes[1].plot(np.arange(1, N + 1), ta[p, :, s], "o-", label="ta")
    axes[1].plot(np.arange(1, N + 1), ta_q[p, :, s], "s--", label="ta quantized")
    axes[1].set_title("ta coefficients before/after quantization")
    axes[1].legend()
    _apply_axis_cfg(axes[1], debug_cfg, figure_key, "ta")
    axes[2].plot(np.arange(1, N + 1), tb[p, :, s], "o-", label="tb")
    axes[2].plot(np.arange(1, N + 1), tb_q[p, :, s], "s--", label="tb quantized")
    axes[2].set_title("tb coefficients before/after quantization")
    axes[2].legend()
    _apply_axis_cfg(axes[2], debug_cfg, figure_key, "tb")
    _text_panel(axes[3], f"{label} phase-quantization diagnostics", [
        f"B_total                    = {float(params.B):.6e} Hz",
        f"B_s                        = {float(params.B_per_sensor[s]):.6e} Hz",
        f"N harmonics                = {N}",
        f"M quantization bins        = {M}",
        f"bits per phase log2(M)     = {_safe_log2_positive(M):.6e}",
        f"normalization target       = {cfg['signal'].get('normalization_target', None)}",
        f"threshold_harmonics        = {cfg['signal'].get('threshold_harmonics', None)}",
        f"reconstruction MSE         = {mse:.6e}",
    ])
    _finish(fig, debug_cfg)


def _build_sensor_x_event_local(S, N):
    out = np.zeros((S, 2 * N * S), dtype=float)
    for s in range(S):
        out[s, 2 * s * N:2 * (s + 1) * N] = 1.0
    return out


def _build_sfc_channel_debug(cfg, debug_cfg, N, S):
    cfg_sfc = copy.deepcopy(cfg)
    cfg_sfc.setdefault("channel", {})
    if not bool(debug_cfg.get("debug", {}).get("add_awgn", True)):
        cfg_sfc["system"]["N0"] = 0.00001
    cfg_sfc["channel"]["sensor_x_event"] = _build_sensor_x_event_local(S, N)
    if cfg_sfc["channel"].get("threshold", None) is None:
        cfg_sfc["channel"].pop("threshold", None)
        cfg_sfc["channel"]["threshold_factor"] = cfg_sfc["channel"].get("threshold_factor", 0.5)
    cfg_sfc.setdefault("reproducibility", {})
    cfg_sfc["reproducibility"].setdefault("seed", cfg.get("monte_carlo", {}).get("seed", 47))
    return SFCChannel(cfg_sfc)


def debug_sfc(cfg, debug_cfg, params, N, t, x_ref, sed=False):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    p, s = _sensor_period(debug_cfg)

    # -------------------------------------------------------------------------
    # Build SFC events from the reference signal.
    # -------------------------------------------------------------------------
    ta, tb, phase = _phase_objects(cfg, params, N, x_ref)
    events = phase.ta_tb_to_events(ta, tb)

    add_awgn = bool(debug_cfg.get("debug", {}).get("add_awgn", True))

    # -------------------------------------------------------------------------
    # NOISY / NOMINAL SFC CHANNEL.
    # -------------------------------------------------------------------------
    channel = _build_sfc_channel_debug(cfg, debug_cfg, N, params.S)

    out = channel(events, return_intermediates=True)

    if not isinstance(out, dict):
        raise TypeError(
            "SFC debug expected SFCChannel(..., return_intermediates=True) "
            "to return a dictionary."
        )

    maps_tx = out.get("maps_tx", None)
    superposed = out.get("superposed", None)
    y = out.get("y", None)
    maps_est = out.get("maps_est", None)
    events_est = out.get("events_est", None)
    diagnostics = out.get("diagnostics", {}) or {}

    if events_est is None:
        raise ValueError("SFCChannel output dictionary does not contain 'events_est'.")

    if maps_tx is None:
        raise ValueError("SFCChannel output dictionary does not contain 'maps_tx'.")

    if maps_est is None:
        raise ValueError("SFCChannel output dictionary does not contain 'maps_est'.")

    if y is None:
        raise ValueError("SFCChannel output dictionary does not contain 'y'.")

    # -------------------------------------------------------------------------
    # CLEAN SFC CHANNEL.
    #
    # Minimal clean-channel definition:
    #
    #   channel.type = "clean"
    #
    # This clean channel is used only to obtain y_clean. Then:
    #
    #   noise = y_noisy - y_clean
    #
    # No N0 modification is done here.
    # -------------------------------------------------------------------------
    cfg_clean = copy.deepcopy(cfg)
    cfg_clean.setdefault("channel", {})
    cfg_clean["channel"]["type"] = "clean"

    cfg_clean["channel"]["sensor_x_event"] = _build_sensor_x_event_local(
        params.S,
        N,
    )

    # MapDetector cannot consume threshold=None.
    if cfg_clean["channel"].get("threshold", None) is None:
        cfg_clean["channel"].pop("threshold", None)
        cfg_clean["channel"]["threshold_factor"] = cfg_clean["channel"].get(
            "threshold_factor",
            0.5,
        )

    cfg_clean.setdefault("reproducibility", {})
    cfg_clean["reproducibility"].setdefault(
        "seed",
        cfg.get("monte_carlo", {}).get("seed", 47),
    )

    channel_clean = SFCChannel(cfg_clean)
    out_clean = channel_clean(events, return_intermediates=True)

    if not isinstance(out_clean, dict):
        raise TypeError(
            "Clean SFC debug expected SFCChannel(..., return_intermediates=True) "
            "to return a dictionary."
        )

    y_clean = out_clean.get("y", None)

    if y_clean is None:
        raise ValueError("Clean SFCChannel output dictionary does not contain 'y'.")

    # -------------------------------------------------------------------------
    # SED, if requested.
    #
    # Reconstruction uses events_est or corrected events_est.
    # The decision map compares raw maps_est against maps_tx, because this
    # diagnostic is about the detector decision before semantic correction.
    # -------------------------------------------------------------------------
    events_for_rec = events_est
    valid_mask = None

    title = ("SFC + SED debug" if sed else "SFC debug")
    title = title + (" | AWGN on" if add_awgn else " | AWGN off")

    figure_key = "sfc_sed" if sed else "sfc"

    if sed:
        n_periods = x_ref.shape[1]

        if events_est.shape[0] % n_periods != 0:
            raise ValueError("SFC event slots are not divisible by n_periods.")

        period_slots = events_est.shape[0] // n_periods

        sed_out = detect_semantic_errors(
            events_est=events_est,
            period_slots=period_slots,
            N=N,
            sensor_x_event=_build_sensor_x_event_local(params.S, N),
            discard_invalid_periods=cfg.get("sed", {}).get(
                "discard_invalid_periods",
                True,
            ),
        )

        if isinstance(sed_out, dict):
            events_for_rec = sed_out["corrected_events_est"]
            valid_mask = np.asarray(sed_out["period_valid_mask"], dtype=bool)
        else:
            events_for_rec = sed_out.corrected_events_est
            valid_mask = np.asarray(sed_out.period_valid_mask, dtype=bool)

    # -------------------------------------------------------------------------
    # Reconstruction from detected/corrected events.
    # -------------------------------------------------------------------------
    ta_rec, tb_rec = phase.event_to_ta_tb(events_for_rec)
    x_hat = _reconstruct_ta_tb(np.real(ta_rec), np.real(tb_rec), t, params)

    mse = float(np.mean((x_ref[:, p, s] - x_hat[:, p, s]) ** 2))

    # -------------------------------------------------------------------------
    # Ideal representation/reconstruction diagnostics.
    #
    # This isolates the intrinsic Fourier/SFC representation error from channel
    # or detection errors.
    # -------------------------------------------------------------------------
    x_hat_ideal = _reconstruct_ta_tb(np.real(ta), np.real(tb), t, params)
    mse_ideal = float(np.mean((x_ref[:, p, s] - x_hat_ideal[:, p, s]) ** 2))

    # Raw reconstruction from raw events_est, before SED correction.
    ta_rec_raw, tb_rec_raw = phase.event_to_ta_tb(events_est)

    ta_mse_raw = float(np.mean((np.real(ta_rec_raw) - np.real(ta)) ** 2))
    tb_mse_raw = float(np.mean((np.real(tb_rec_raw) - np.real(tb)) ** 2))

    ta_mse_used = float(np.mean((np.real(ta_rec) - np.real(ta)) ** 2))
    tb_mse_used = float(np.mean((np.real(tb_rec) - np.real(tb)) ** 2))

    events_bin = (np.asarray(events) > 0).astype(int)
    events_est_bin = (np.asarray(events_est) > 0).astype(int)

    if events_bin.shape != events_est_bin.shape:
        raise ValueError(
            "events and events_est have different shapes: "
            f"{events_bin.shape} vs {events_est_bin.shape}."
        )

    event_fp = int(np.sum((events_bin == 0) & (events_est_bin == 1)))
    event_fn = int(np.sum((events_bin == 1) & (events_est_bin == 0)))
    event_tp = int(np.sum((events_bin == 1) & (events_est_bin == 1)))
    event_tn = int(np.sum((events_bin == 0) & (events_est_bin == 0)))
    event_abs_error = float(np.sum(np.abs(events_est_bin - events_bin)))
    event_error_rate = event_abs_error / max(events_bin.size, 1)

    # -------------------------------------------------------------------------
    # Period-selection helpers.
    #
    # Observed shape:
    #
    #   maps_tx.shape = (event_slots_total, num_event_ids, L, R)
    #
    # For map-domain plotting, collapse num_event_ids:
    #
    #   (slots, event_ids, L, R) -> max over event_ids -> (slots, L, R)
    #
    # Then flatten:
    #
    #   (slots, L, R) -> (slots * L, R)
    #
    # so the plot has:
    #
    #   x = resource index R
    #   y = flattened time-row index mapped to [0, tau]
    # -------------------------------------------------------------------------
    n_periods = int(x_ref.shape[1])
    event_slots_total = int(events.shape[0])

    if event_slots_total % n_periods != 0:
        raise ValueError(
            "events.shape[0] must be divisible by n_periods for SFC debug. "
            f"events.shape[0]={event_slots_total}, n_periods={n_periods}."
        )

    event_slots_per_period = event_slots_total // n_periods

    def _select_period_along_first_axis(arr, name):
        arr = np.asarray(arr)

        if arr.shape[0] == event_slots_total:
            start = p * event_slots_per_period
            stop = (p + 1) * event_slots_per_period
            return arr[start:stop]

        return arr

    def _collapse_to_time_resource_matrix(
            x,
            name,
            binary=False,
            collapse_event_axis="max",
            complex_mode="magnitude",
    ):
        """
        Convert SFC intermediate tensors into a 2-D resource-time matrix.

        Supported shapes
        ----------------
        4-D:
            (event_slots, event_ids, L, R)

        3-D:
            (event_slots, L, R)

        2-D:
            (time_rows, R)

        complex_mode
        ------------
        magnitude:
            Complex values are converted to abs(.) for plotting.

        preserve:
            Complex values are preserved for power/noise calculations.
        """
        if x is None:
            return None

        arr = np.asarray(x)

        if arr.ndim not in (2, 3, 4):
            raise ValueError(
                f"{name} must be 2-D, 3-D, or 4-D for plotting, "
                f"got shape={arr.shape}."
            )

        arr = _select_period_along_first_axis(arr, name)

        if np.iscomplexobj(arr):
            if complex_mode == "magnitude":
                arr = np.abs(arr)
            elif complex_mode == "preserve":
                pass
            else:
                raise ValueError(
                    "complex_mode must be 'magnitude' or 'preserve', "
                    f"got {complex_mode}."
                )

        # -------------------------------------------------------------
        # 4-D case:
        #   (slots, event_ids, L, R)
        # -------------------------------------------------------------
        if arr.ndim == 4:
            if collapse_event_axis == "max":
                arr = np.max(arr, axis=1)
            elif collapse_event_axis == "sum":
                arr = np.sum(arr, axis=1)
            else:
                raise ValueError(
                    "collapse_event_axis must be 'max' or 'sum', "
                    f"got {collapse_event_axis}."
                )

        # -------------------------------------------------------------
        # 3-D case:
        #   (slots, L, R)
        # -------------------------------------------------------------
        if arr.ndim == 3:
            slots, L_local, R_local = arr.shape
            arr2 = arr.reshape(slots * L_local, R_local)

        # -------------------------------------------------------------
        # 2-D case:
        #   already (time_rows, R)
        # -------------------------------------------------------------
        elif arr.ndim == 2:
            arr2 = arr

        else:
            raise RuntimeError(
                f"Unexpected dimensionality after collapse for {name}: "
                f"{arr.ndim}"
            )

        if complex_mode == "preserve":
            arr2 = np.asarray(arr2)
        else:
            arr2 = np.asarray(arr2, dtype=float)

        if binary:
            arr2 = (arr2 > 0).astype(int)

        return arr2

    # -------------------------------------------------------------------------
    # Convert intermediates to resource-time matrices.
    # -------------------------------------------------------------------------
    maps_tx_plot = _collapse_to_time_resource_matrix(
        maps_tx,
        "maps_tx",
        binary=True,
        collapse_event_axis="max",
        complex_mode="magnitude",
    )

    maps_est_plot = _collapse_to_time_resource_matrix(
        maps_est,
        "maps_est",
        binary=True,
        collapse_event_axis="max",
        complex_mode="magnitude",
    )

    superposed_plot = _collapse_to_time_resource_matrix(
        superposed,
        "superposed",
        binary=False,
        collapse_event_axis="sum",
        complex_mode="magnitude",
    )

    y_plot = _collapse_to_time_resource_matrix(
        y,
        "y",
        binary=False,
        collapse_event_axis="sum",
        complex_mode="magnitude",
    )

    # Preserve complex values for empirical power/noise calculation.
    y_noisy_power = _collapse_to_time_resource_matrix(
        y,
        "y",
        binary=False,
        collapse_event_axis="sum",
        complex_mode="preserve",
    )

    y_clean_power = _collapse_to_time_resource_matrix(
        y_clean,
        "y_clean",
        binary=False,
        collapse_event_axis="sum",
        complex_mode="preserve",
    )

    if maps_tx_plot.shape != maps_est_plot.shape:
        raise ValueError(
            "maps_tx and maps_est have different plotting shapes: "
            f"{maps_tx_plot.shape} vs {maps_est_plot.shape}."
        )

    if y_noisy_power.shape != y_clean_power.shape:
        raise ValueError(
            "Noisy and clean SFC y matrices have different plotting shapes: "
            f"{y_noisy_power.shape} vs {y_clean_power.shape}."
        )

    # -------------------------------------------------------------------------
    # Resource axis.
    # -------------------------------------------------------------------------
    R_system = int(getattr(params, "R", maps_tx_plot.shape[1]))
    R_plot = int(maps_tx_plot.shape[1])

    if R_plot != R_system:
        print(
            "[WARNING] SFC debug: plotted map width differs from params.R. "
            f"maps width={R_plot}, params.R={R_system}. "
            "Plotting the actual map width returned by SFCChannel."
        )

    # -------------------------------------------------------------------------
    # Classification map comparing maps_est against maps_tx.
    #
    # classification = 2*TX + RX:
    #   0 -> TN
    #   1 -> FP
    #   2 -> FN
    #   3 -> TP
    # -------------------------------------------------------------------------
    tx_occ = (maps_tx_plot > 0).astype(int)
    rx_occ = (maps_est_plot > 0).astype(int)

    classification = 2 * tx_occ + rx_occ

    tp = int(np.sum((tx_occ == 1) & (rx_occ == 1)))
    tn = int(np.sum((tx_occ == 0) & (rx_occ == 0)))
    fp = int(np.sum((tx_occ == 0) & (rx_occ == 1)))
    fn = int(np.sum((tx_occ == 1) & (rx_occ == 0)))

    fp_rate = fp / max(fp + tn, 1)
    fn_rate = fn / max(fn + tp, 1)
    total_error_rate = (fp + fn) / max(tx_occ.size, 1)

    # -------------------------------------------------------------------------
    # Superposed / y statistics and histogram partitions.
    # -------------------------------------------------------------------------
    if superposed_plot is not None:
        superposed_min = float(np.nanmin(superposed_plot))
        superposed_max = float(np.nanmax(superposed_plot))
        superposed_mean = float(np.nanmean(superposed_plot))
        superposed_nonzero_fraction = float(
            np.mean(np.asarray(superposed_plot) != 0)
        )
    else:
        superposed_min = np.nan
        superposed_max = np.nan
        superposed_mean = np.nan
        superposed_nonzero_fraction = np.nan

    if y_plot is not None:
        y_abs = np.abs(y_plot)

        if superposed_plot is not None and superposed_plot.shape == y_plot.shape:
            ind_where_zero = np.where(superposed_plot == 0)
            ind_where_active = np.where(superposed_plot > 0)

            y_idle_abs = y_abs[ind_where_zero]
            y_active_abs = y_abs[ind_where_active]
        else:
            ind_where_zero = None
            ind_where_active = None

            y_idle_abs = np.asarray([], dtype=float)
            y_active_abs = np.asarray([], dtype=float)

        y_abs_min = float(np.nanmin(y_abs))
        y_abs_max = float(np.nanmax(y_abs))
        y_abs_mean = float(np.nanmean(y_abs))

        if y_idle_abs.size > 0:
            y_idle_abs_mean = float(np.nanmean(y_idle_abs))
            y_idle_abs_var = float(np.nanvar(y_idle_abs))
        else:
            y_idle_abs_mean = np.nan
            y_idle_abs_var = np.nan

        if y_active_abs.size > 0:
            y_active_abs_mean = float(np.nanmean(y_active_abs))
            y_active_abs_var = float(np.nanvar(y_active_abs))
        else:
            y_active_abs_mean = np.nan
            y_active_abs_var = np.nan

    else:
        y_abs = np.asarray([], dtype=float)
        y_idle_abs = np.asarray([], dtype=float)
        y_active_abs = np.asarray([], dtype=float)

        y_abs_min = np.nan
        y_abs_max = np.nan
        y_abs_mean = np.nan

        y_idle_abs_mean = np.nan
        y_idle_abs_var = np.nan

        y_active_abs_mean = np.nan
        y_active_abs_var = np.nan

    # -------------------------------------------------------------------------
    # Empirical power/noise statistics.
    #
    # Clean y:
    #   y_clean_power is the clean matched-filter/resource-level TX signal.
    #
    # Noisy y:
    #   y_noisy_power is the noisy matched-filter/resource-level signal.
    #
    # Noise estimate:
    #   n = y_noisy_power - y_clean_power
    #
    # Two different quantities:
    #
    #   P_tx_mean:
    #       period-average equivalent power:
    #       sum(|y_clean|^2) / tau
    #
    #   P_N0_mean:
    #       empirical noise variance per matched-filter/resource output:
    #       mean(|noise|^2)
    #
    # P_N0_mean is the quantity directly comparable to params.N0.
    # -------------------------------------------------------------------------
    noise_power = y_noisy_power - y_clean_power

    P_tx_mean = float(
        np.sum(np.abs(y_clean_power) ** 2) / float(params.tau)
    )

    P_N0_mean = float(
        np.mean(np.abs(noise_power) ** 2)
    )

    N0_target = float(params.N0)

    # -------------------------------------------------------------------------
    # Matrix axes.
    # -------------------------------------------------------------------------
    extent = [-0.5, R_plot - 0.5, params.tau, 0.0]
    resource_ticks = np.arange(R_plot)

    # -------------------------------------------------------------------------
    # Figure layout:
    #
    # 1. reconstruction
    # 2. text statistics
    # 3. superposed
    # 4. decision classification map comparing maps_est with maps_tx
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex="none")
    fig.suptitle(title)

    # -------------------------------------------------------------------------
    # 1) Recovered signal: keep as before.
    # -------------------------------------------------------------------------
    axes[0].plot(t, x_ref[:, p, s], label="simulation signal")
    axes[0].plot(
        t,
        x_hat[:, p, s],
        "--",
        label=f"reconstructed, MSE={mse:.3e}",
    )
    axes[0].set_title(
        "SFC reconstruction from detected events"
        + (" after SED" if sed else "")
    )
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("amplitude")
    axes[0].legend()
    _apply_axis_cfg(axes[0], debug_cfg, figure_key, "reconstruction")

    # -------------------------------------------------------------------------
    # 2) Text statistics.
    # -------------------------------------------------------------------------
    stats_lines = [
        f"reconstruction MSE           = {mse:.6e}",
        f"ideal representation MSE     = {mse_ideal:.6e}",
        f"event TP/TN/FP/FN            = "
        f"{event_tp} / {event_tn} / {event_fp} / {event_fn}",
        f"event abs error / error rate = "
        f"{event_abs_error:.6e} / {event_error_rate:.6e}",
        f"ta MSE raw / used            = {ta_mse_raw:.6e} / {ta_mse_used:.6e}",
        f"tb MSE raw / used            = {tb_mse_raw:.6e} / {tb_mse_used:.6e}",
        f"map TP/TN/FP/FN              = "
        f"{tp} / {tn} / {fp} / {fn}",
        f"map total decision error     = {total_error_rate:.6e}",
        f"superposed min/mean/max      = "
        f"{superposed_min:.6e} / {superposed_mean:.6e} / {superposed_max:.6e}",
        f"E_chip / idle |y| mean/var  = "
        f"{channel.channel.E_chip:.6e} / {y_idle_abs_mean:.6e} / {y_idle_abs_var:.6e}",
        f"active |y| mean/var         = "
        f"{y_active_abs_mean:.6e} / {y_active_abs_var:.6e}",
    ]

    if sed and valid_mask is not None:
        stats_lines.extend(
            [
                "",
                f"SED valid fraction          = {float(np.mean(valid_mask)):.6e}",
            ]
        )

    axes[1].axis("off")
    axes[1].set_title("SFC statistics")
    axes[1].text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    # -------------------------------------------------------------------------
    # 3) Superposed resource-time frame.
    # -------------------------------------------------------------------------
    if superposed_plot is not None:
        im_superposed = axes[2].imshow(
            superposed_plot,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            origin="upper",
        )
        fig.colorbar(
            im_superposed,
            ax=axes[2],
            fraction=0.025,
            pad=0.01,
            label="superposed value",
        )
        axes[2].set_title("SFC superposed resource-time frame")
    else:
        axes[2].axis("off")
        axes[2].set_title("SFC superposed resource-time frame")
        axes[2].text(
            0.02,
            0.95,
            "SFCChannel did not return 'superposed'.",
            transform=axes[2].transAxes,
            va="top",
            ha="left",
            family="monospace",
        )

    axes[2].set_xlabel("resource index R")
    axes[2].set_ylabel("time [s]")
    _apply_axis_cfg(axes[2], debug_cfg, figure_key, "superposed")

    # -------------------------------------------------------------------------
    # 4) Classification map: maps_est compared with maps_tx.
    # -------------------------------------------------------------------------
    cmap = ListedColormap(
        [
            "black",  # 0: TN, idle -> idle
            "red",  # 1: FP, idle -> used
            "orange",  # 2: FN, used -> idle
            "green",  # 3: TP, used -> used
        ]
    )

    axes[3].imshow(
        classification,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=3,
        extent=extent,
        origin="upper",
    )

    axes[3].set_title(
        "Decision map: maps_est vs maps_tx | "
        f"FP={fp_rate:.3f}, FN={fn_rate:.3f}, Pe={total_error_rate:.3f} "
        f"| TP={tp}, TN={tn}, FP={fp}, FN={fn}"
    )
    axes[3].set_xlabel("resource index R")
    axes[3].set_ylabel("time [s]")

    legend_handles = [
        Patch(facecolor="black", label="TN: idle → idle"),
        Patch(facecolor="red", label="FP: idle → used"),
        Patch(facecolor="orange", label="FN: used → idle"),
        Patch(facecolor="green", label="TP: used → used"),
    ]

    axes[3].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    _apply_axis_cfg(axes[3], debug_cfg, figure_key, "maps_classification")

    # -------------------------------------------------------------------------
    # Force resource-axis ticks AFTER YAML axis configuration.
    # -------------------------------------------------------------------------
    for ax in axes[2:4]:
        ax.set_xlim(-0.5, R_plot - 0.5)
        ax.set_xticks(resource_ticks)
        ax.set_xticklabels([str(i) for i in resource_ticks])
        ax.set_xlabel("resource index R")

    # -------------------------------------------------------------------------
    # Optional SFC |y| histogram.
    #
    # Enable by adding "sfc_y_hist" to:
    #
    #   debug:
    #     methods:
    #       - sfc
    #       - sfc_y_hist
    #
    # This plot shows one histogram per subplot:
    #
    #   1) all |y|
    #   2) idle |y|, where superposed == 0
    #   3) active |y|, where superposed > 0
    # -------------------------------------------------------------------------
    methods_enabled = set(debug_cfg.get("debug", {}).get("methods", []))

    plot_sfc_y_hist = (
            "sfc_y_hist" in methods_enabled
            or "sfc_hist" in methods_enabled
    )

    if plot_sfc_y_hist:
        hist_bins = int(debug_cfg.get("debug", {}).get("hist_bins", 80))

        y_hist_all = np.asarray(y_abs, dtype=float).reshape(-1)
        y_hist_all = y_hist_all[np.isfinite(y_hist_all)]

        y_hist_idle = np.asarray(y_idle_abs, dtype=float).reshape(-1)
        y_hist_idle = y_hist_idle[np.isfinite(y_hist_idle)]

        y_hist_active = np.asarray(y_active_abs, dtype=float).reshape(-1)
        y_hist_active = y_hist_active[np.isfinite(y_hist_active)]

        fig_hist, axes_hist = plt.subplots(
            3,
            1,
            figsize=(10, 9),
            sharex="none",
        )

        fig_hist.suptitle("SFC matched-filter/resource-output histograms")

        # ---------------------------------------------------------------------
        # Histogram 1: all |y|.
        # ---------------------------------------------------------------------
        if y_hist_all.size > 0:
            axes_hist[0].hist(
                y_hist_all,
                bins=hist_bins,
                alpha=0.75,
                density=True,
                label="all |y|",
            )

        if np.isfinite(y_abs_mean):
            axes_hist[0].axvline(
                y_abs_mean,
                color="C0",
                linestyle="--",
                linewidth=1.2,
                label=f"mean all |y| = {y_abs_mean:.3e}",
            )

        axes_hist[0].set_title("All resource-time cells")
        axes_hist[0].set_xlabel("|y|")
        axes_hist[0].set_ylabel("density")
        axes_hist[0].legend()
        axes_hist[0].grid(True, linestyle=":")

        # ---------------------------------------------------------------------
        # Histogram 2: idle |y|, superposed == 0.
        # ---------------------------------------------------------------------
        if y_hist_idle.size > 0:
            axes_hist[1].hist(
                y_hist_idle,
                bins=hist_bins,
                alpha=0.75,
                density=True,
                label="idle |y|, superposed = 0",
                color="C1",
            )

        if np.isfinite(y_idle_abs_mean):
            axes_hist[1].axvline(
                y_idle_abs_mean,
                color="C1",
                linestyle="--",
                linewidth=1.2,
                label=f"mean idle |y| = {y_idle_abs_mean:.3e}",
            )

        axes_hist[1].set_title(
            f"Idle resource-time cells | var={y_idle_abs_var:.3e}"
        )
        axes_hist[1].set_xlabel("|y|")
        axes_hist[1].set_ylabel("density")
        axes_hist[1].legend()
        axes_hist[1].grid(True, linestyle=":")

        # ---------------------------------------------------------------------
        # Histogram 3: active |y|, superposed > 0.
        # ---------------------------------------------------------------------
        if y_hist_active.size > 0:
            axes_hist[2].hist(
                y_hist_active,
                bins=hist_bins,
                alpha=0.75,
                density=True,
                label="active |y|, superposed > 0",
                color="C2",
            )

        if np.isfinite(y_active_abs_mean):
            axes_hist[2].axvline(
                y_active_abs_mean,
                color="C2",
                linestyle="--",
                linewidth=1.2,
                label=f"mean active |y| = {y_active_abs_mean:.3e}",
            )

        axes_hist[2].set_title(
            f"Active resource-time cells | var={y_active_abs_var:.3e}"
        )
        axes_hist[2].set_xlabel("|y|")
        axes_hist[2].set_ylabel("density")
        axes_hist[2].legend()
        axes_hist[2].grid(True, linestyle=":")

        _finish(fig_hist, debug_cfg)

    _finish(fig, debug_cfg)


def run_debug_fair_methods_comparison(debug_config_path: str | Path = DEFAULT_DEBUG_CONFIG):
    debug_cfg = _load_yaml(debug_config_path)
    cfg, debug_cfg, params, N, t, rng = _prepare_cfg(debug_cfg)
    n_periods = int(cfg.get("signal", {}).get("n_periods", 1))
    p, s = _sensor_period(debug_cfg)
    if p >= n_periods:
        raise IndexError(f"debug.period={p} but n_periods={n_periods}")
    if s >= int(params.S):
        raise IndexError(f"debug.sensor={s} but S={params.S}")
    x_raw, x_ref = generate_raw_and_processed_source(cfg, params, N, t, rng, n_periods)
    _, capacity_per_sensor, budget_bits_per_sensor = _build_fdma(params)
    add_awgn = bool(debug_cfg.get("debug", {}).get("add_awgn", True))
    print("\n[DEBUG FAIR METHODS]")
    print(f"B_total = {float(params.B):.6e}")
    print(f"S = {params.S}, N = {N}, tau = {params.tau}, Tt = {cfg['signal']['Tt']}")
    print(f"period = {p}, sensor = {s}")
    print(f"B_sensor = {float(params.B_per_sensor[s]):.6e}")
    print(f"FDMA capacity sensor = {float(capacity_per_sensor[s]):.6e}")
    print(f"FDMA budget bits sensor = {budget_bits_per_sensor[s]}")
    print(f"add_awgn = {_format_bool(add_awgn)}")
    print(f"P = {float(params.P):.6e}")
    print(f"N0 = {float(params.N0):.6e}")
    print(f"SNR_dB = {float(params.SNR_dB):.6e}")
    print(f"SNR_per_sensor_dB = {float(np.mean(params.SNR_per_sensor_dB)):.6e}")
    methods = set(
        debug_cfg.get("debug", {}).get("methods", ["source", "benchmark", "cs", "ppm", "sod", "fri", "rbcp", "rbcp_time", "sfc", "sfc_sed"]))
    if "source" in methods:
        plot_source_debug(cfg, debug_cfg, params, t, x_raw, x_ref)
    if "benchmark" in methods or "nyquist" in methods:
        debug_benchmark_nyquist(cfg, debug_cfg, params, t, x_ref, budget_bits_per_sensor)
    if "cs" in methods:
        debug_cs(cfg, debug_cfg, params, t, x_ref, capacity_per_sensor)
    if "ppm" in methods:
        debug_ppm(cfg, debug_cfg, params, t, x_ref, rng)
    if "sod" in methods:
        debug_sod(cfg, debug_cfg, params, t, x_ref, budget_bits_per_sensor)
    if "fri" in methods:
        debug_fri(cfg, debug_cfg, params, t, x_ref, budget_bits_per_sensor)
    if "rbcp" in methods:
        debug_rbcp(cfg, debug_cfg, params, N, t, x_ref, use_time=False)
    if "rbcp_time" in methods:
        debug_rbcp(cfg, debug_cfg, params, N, t, x_ref, use_time=True)
    if "sfc" in methods:
        debug_sfc(cfg, debug_cfg, params, N, t, x_ref, sed=False)
    if "sfc_sed" in methods:
        debug_sfc(cfg, debug_cfg, params, N, t, x_ref, sed=True)
    return {"cfg": cfg, "debug_cfg": debug_cfg, "params": params, "N": N, "t": t, "x_raw": x_raw, "x_ref": x_ref,
            "capacity_per_sensor": capacity_per_sensor, "budget_bits_per_sensor": budget_bits_per_sensor}


if __name__ == "__main__":
    run_debug_fair_methods_comparison(DEFAULT_DEBUG_CONFIG)
