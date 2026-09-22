"""
sfc/pipelines/fair_methods_comparison_vs_B.py

Fair comparison pipeline versus total bandwidth B.

Compared methods
----------------
- Benchmark / Nyquist + FDMA
- CS + FDMA
- PPM + FDMA
- SoD + FDMA
- FRI-inspired + FDMA
- RbCP
- RbCP_time
- SFC
- SFC + SED

Core conventions
----------------
1. Source processing order:
       random source -> filter_periodic(signal.W) -> DC handling
       -> final peak-to-peak scaling

2. FDMA capacity-derived budget:
       B_s = alpha_s B
       C_s = B_s log2(1 + P/(B_s N0))
       budget_bits_s = floor(C_s tau)

3. Correct SoD + FDMA logic:
       SoD detects candidate events, then adapts K_tx and quantization bits for
       event time/amplitude to the available FDMA channel budget. SoD always
       transmits at least one event.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
from sfc.core.system_parameters import (
    build_derived_system_parameters,
    compute_benchmark_M_per_sensor,
)
from sfc.core.theory import (
    compute_N0,
    compute_sensor_snr,
    compute_snr_db,
    compute_snr_linear,
)

PERIODIC_REPLICAS = 10


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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_fair_methods_comparison_vs_B_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["monte_carlo"]["seed"]))

    print("[INFO] Starting fair_methods_comparison_vs_B data generation")
    print(f"[INFO] Monte Carlo seed = {cfg['monte_carlo']['seed']}")
    print(f"[INFO] Trials per B = {cfg['monte_carlo']['interactions']}")

    b_values = _resolve_B_values_from_cfg(cfg)

    print(f"[INFO] Number of B points = {len(b_values)}")

    rows: List[Dict[str, Any]] = []

    for b_idx, B in enumerate(b_values, start=1):
        cfg_B = copy.deepcopy(cfg)
        cfg_B["system"]["B"] = float(B)
        _apply_power_model(cfg_B)

        params = build_derived_system_parameters(cfg_B)
        N = int(cfg_B["signal"].get("N_override", params.N))
        n_trials = int(cfg_B["monte_carlo"]["interactions"])

        mse_benchmark_fdma, M_benchmark_vec = _run_benchmark_fdma_branch(cfg_B, params)
        M_benchmark_arr = np.asarray(M_benchmark_vec, dtype=object)
        if M_benchmark_arr.size > 0:
            M_benchmark_min = _safe_min_int(M_benchmark_arr)
            M_benchmark_mean = _safe_mean_float(M_benchmark_arr)
            M_benchmark_max = _safe_max_int(M_benchmark_arr)
        else:
            M_benchmark_min = np.nan
            M_benchmark_mean = np.nan
            M_benchmark_max = np.nan

        fdma_core = _build_fdma_core(params)
        capacity_per_sensor = np.asarray(fdma_core.get_capacity_per_sensor(), dtype=float)
        budget_bits_per_sensor = np.floor(capacity_per_sensor * params.tau).astype(int)

        sfc_channel = None
        run_sfc = _mode_enabled(cfg_B, "run_sfc", True)
        run_sfc_sed = _mode_enabled(cfg_B, "run_sfc_sed", True)
        if run_sfc or run_sfc_sed:
            sfc_channel = _build_sfc_channel_for_B(cfg_B, N=N, S=params.S)

        accum = _new_accumulator()
        for _ in range(n_trials):
            trial = _run_one_trial(
                cfg=cfg_B,
                rng=rng,
                params=params,
                N=N,
                sfc_channel=sfc_channel,
                capacity_per_sensor=capacity_per_sensor,
                budget_bits_per_sensor=budget_bits_per_sensor,
            )
            _accumulate_trial(accum, trial)

        row = _build_result_row(
            B=float(B),
            params=params,
            cfg=cfg_B,
            n_trials=n_trials,
            mse_benchmark_fdma=mse_benchmark_fdma,
            M_benchmark_min=M_benchmark_min,
            M_benchmark_mean=M_benchmark_mean,
            M_benchmark_max=M_benchmark_max,
            capacity_per_sensor=capacity_per_sensor,
            budget_bits_per_sensor=budget_bits_per_sensor,
            accum=accum,
        )
        print(f"\n[INFO] Completed B point {b_idx}/{len(b_values)}")
        _print_bandwidth_point_summary(row)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# POWER / FDMA
# =============================================================================

def _apply_power_model(cfg: Dict[str, Any]):
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


def _build_fdma_core(params) -> FDMACore:
    return FDMACore(
        S=params.S,
        B_total=params.B,
        P_per_sensor=params.P,
        tau=params.tau,
        bandwidth_allocation=params.bandwidth_allocation,
        N0=params.N0,
        normalize_sensor_power=False,
        return_nonorthogonal_sum_preview=False,
    )


# =============================================================================
# ONE MONTE CARLO TRIAL
# =============================================================================

def _run_one_trial(
        cfg: Dict[str, Any],
        rng: np.random.Generator,
        params,
        N: int,
        sfc_channel: Optional[SFCChannel],
        capacity_per_sensor: np.ndarray,
        budget_bits_per_sensor: np.ndarray,
) -> Dict[str, Any]:
    n_periods = int(cfg.get("signal", {}).get("n_periods", 1))
    Tt = float(cfg["signal"]["Tt"])
    t = np.arange(0.0, params.tau, Tt)

    x_ref = _generate_common_source(cfg, rng, params, N, t, n_periods)

    trial = {
        "mse_cs_fdma": np.nan,
        "mse_ppm_fdma": np.nan,
        "mse_sod_fdma": np.nan,
        "mse_fri_fdma": np.nan,
        "mse_rbcp": np.nan,
        "mse_rbcp_time": np.nan,
        "mse_sfc": np.nan,
        "mse_sfc_sed": np.nan,
        "sfc_sed_valid_fraction": np.nan,
        "sod_fdma_feasible_fraction": np.nan,
        "fri_fdma_feasible_fraction": np.nan,
        "cs_measurements": [],
        "cs_sampling_rate": [],
        "cs_num_samples": [],
        "cs_measurement_bits": [],
        "cs_sparsity_eff": [],
        "cs_quantized_fraction": [],
        "ppm_fs_msg": [],
        "sod_num_events": [],
        "sod_candidate_events": [],
        "sod_payload_bits": [],
        "sod_payload_budget_bits": [],
        "sod_feasible_flags": [],
        "sod_amplitude_bits": [],
        "sod_time_bits": [],
        "fri_K": [],
        "fri_bits_location": [],
        "fri_bits_amplitude": [],
        "fri_budget_bits": [],
        "fri_raw_budget_bits": [],
        "fri_feasible_flags": [],
    }

    if _mode_enabled(cfg, "run_cs_fdma", _mode_enabled(cfg, "run_cs", True)):
        (
            trial["mse_cs_fdma"],
            trial["cs_measurements"],
            trial["cs_sampling_rate"],
            trial["cs_num_samples"],
            trial["cs_measurement_bits"],
            trial["cs_sparsity_eff"],
            trial["cs_quantized_fraction"],
        ) = _run_cs_fdma_branch(x_ref, t, cfg, params, rng, capacity_per_sensor)

    if _mode_enabled(cfg, "run_ppm_fdma", _mode_enabled(cfg, "run_ppm", True)):
        trial["mse_ppm_fdma"], trial["ppm_fs_msg"] = _run_ppm_fdma_branch(
            x_ref, t, cfg, params, rng
        )

    if _mode_enabled(cfg, "run_sod", False):
        (
            trial["mse_sod_fdma"],
            trial["sod_num_events"],
            trial["sod_payload_bits"],
            trial["sod_payload_budget_bits"],
            trial["sod_feasible_flags"],
            trial["sod_fdma_feasible_fraction"],
            trial["sod_candidate_events"],
            trial["sod_amplitude_bits"],
            trial["sod_time_bits"],
        ) = _run_sod_fdma_branch(
            x_ref=x_ref,
            t=t,
            cfg=cfg,
            params=params,
            budget_bits_per_sensor=budget_bits_per_sensor,
        )

    if _mode_enabled(cfg, "run_fri", False):
        (
            trial["mse_fri_fdma"],
            trial["fri_K"],
            trial["fri_bits_location"],
            trial["fri_bits_amplitude"],
            trial["fri_budget_bits"],
            trial["fri_raw_budget_bits"],
            trial["fri_feasible_flags"],
            trial["fri_fdma_feasible_fraction"],
        ) = _run_fri_fdma_branch(x_ref, t, cfg, params, budget_bits_per_sensor)

    if _mode_enabled(cfg, "run_rbcp", True):
        trial["mse_rbcp"] = _run_rbcp_branch(x_ref, t, cfg, params, N, params.M_rbcp)

    if _mode_enabled(cfg, "run_rbcp_time", True):
        trial["mse_rbcp_time"] = _run_rbcp_branch(x_ref, t, cfg, params, N, params.M_time)

    run_sfc = _mode_enabled(cfg, "run_sfc", True)
    run_sfc_sed = _mode_enabled(cfg, "run_sfc_sed", True)
    if run_sfc or run_sfc_sed:
        trial["mse_sfc"], trial["mse_sfc_sed"], trial["sfc_sed_valid_fraction"] = (
            _run_sfc_family_branch(x_ref, t, cfg, params, N, sfc_channel, run_sfc, run_sfc_sed)
        )

    return trial


# =============================================================================
# SOURCE GENERATION
# =============================================================================

def _generate_common_source(
        cfg: Dict[str, Any],
        rng: np.random.Generator,
        params,
        N: int,
        t: np.ndarray,
        n_periods: int,
) -> np.ndarray:
    _ = N
    model = cfg.get("signal_model", {}).get("type", "bandlimited_random")

    if model == "sparse_innovation":
        return _generate_sparse_innovation_source(cfg, rng, params, t, n_periods)

    if model != "bandlimited_random":
        raise ValueError("signal_model.type must be 'bandlimited_random' or 'sparse_innovation'.")

    n_time = len(t)
    S = int(params.S)
    dist = cfg["signal"].get("distribution", "gaussian")

    if dist == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=(n_time, n_periods, S))
    elif dist == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=(n_time, n_periods, S))
    else:
        raise ValueError("signal.distribution must be 'uniform' or 'gaussian'.")

    W_filter = float(cfg["signal"].get("W", params.W))
    Tt = float(cfg["signal"]["Tt"])
    x_filtered = np.zeros_like(x_raw)

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(x_raw[:, p, s], W_filter, Tt, params.tau)

    return _postprocess_source_signal(x_filtered, cfg, params)


def _generate_sparse_innovation_source(cfg, rng, params, t, n_periods):
    sm_cfg = cfg.get("signal_model", {}).get("sparse_innovation", {})
    K = int(sm_cfg.get("K", 3))
    kernel = str(sm_cfg.get("kernel", "gaussian")).lower()
    sigma = float(sm_cfg.get("kernel_sigma", 0.01))
    amp_dist = str(sm_cfg.get("amplitude_distribution", "uniform")).lower()
    amp_min = float(sm_cfg.get("amplitude_min", -1.0))
    amp_max = float(sm_cfg.get("amplitude_max", 1.0))
    periodic = bool(sm_cfg.get("periodic", True))
    min_sep = float(sm_cfg.get("min_separation", 0.0))

    S = int(params.S)
    tau = float(params.tau)
    x = np.zeros((len(t), n_periods, S), dtype=float)

    for p in range(n_periods):
        for s in range(S):
            locs = _draw_innovation_locations(rng, K, tau, min_sep)
            if amp_dist == "uniform":
                amps = rng.uniform(amp_min, amp_max, size=K)
            elif amp_dist == "gaussian":
                amps = rng.normal(0.0, 1.0, size=K)
            else:
                raise ValueError("Unsupported sparse innovation amplitude distribution.")

            sig = np.zeros(len(t), dtype=float)
            for loc, amp in zip(locs, amps):
                dt = t - loc
                if periodic:
                    dt = (dt + 0.5 * tau) % tau - 0.5 * tau
                if kernel == "gaussian":
                    phi = np.exp(-0.5 * (dt / sigma) ** 2)
                elif kernel == "triangular":
                    phi = np.maximum(1.0 - np.abs(dt) / sigma, 0.0)
                elif kernel == "sinc":
                    phi = np.sinc(dt / sigma)
                elif kernel == "nearest":
                    phi = np.zeros_like(t)
                    phi[np.argmin(np.abs(dt))] = 1.0
                else:
                    raise ValueError("Unsupported sparse innovation kernel.")
                sig += amp * phi
            x[:, p, s] = sig

    return _postprocess_source_signal(x, cfg, params)


def _draw_innovation_locations(rng, K, tau, min_separation):
    if min_separation <= 0:
        return np.sort(rng.uniform(0.0, tau, size=K))
    locs = []
    for _ in range(10000):
        candidate = float(rng.uniform(0.0, tau))
        if all(abs(candidate - loc) >= min_separation for loc in locs):
            locs.append(candidate)
            if len(locs) == K:
                break
    while len(locs) < K:
        locs.append(float(rng.uniform(0.0, tau)))
    return np.sort(np.asarray(locs[:K], dtype=float))


def _postprocess_source_signal(x_filtered: np.ndarray, cfg: Dict[str, Any], params) -> np.ndarray:
    Tt = float(cfg["signal"]["Tt"])
    tau = float(params.tau)
    x_work = np.array(x_filtered, copy=True, dtype=float)
    _, n_periods, S = x_work.shape

    if not bool(cfg.get("dc", {}).get("enabled", False)):
        for p in range(n_periods):
            for s in range(S):
                dc = Tt * np.sum(x_work[:, p, s]) / tau
                x_work[:, p, s] -= dc

    peak_to_peak = cfg["signal"].get("peak_to_peak", 0.0)
    if peak_to_peak != 0:
        peak_to_peak = float(peak_to_peak)
        for p in range(n_periods):
            for s in range(S):
                current_p2p = np.max(x_work[:, p, s]) - np.min(x_work[:, p, s])
                if current_p2p != 0:
                    x_work[:, p, s] *= peak_to_peak / current_p2p

    return x_work


# =============================================================================
# SHARED NUMERICAL HELPERS
# =============================================================================

def _safe_log2_positive(value) -> float:
    value = int(value)
    value = max(value, 1)
    return float(np.log2(float(value)))


def _resolve_benchmark_sampling_rate(cfg, params) -> float:
    benchmark_cfg = cfg.get("benchmark", {})
    nyq_cfg = cfg.get("acquisition", {}).get("nyquist", {})
    sampling_rate = float(nyq_cfg.get("sampling_rate", benchmark_cfg.get("sampling_rate", params.W)))
    factor = float(nyq_cfg.get("effective_rate_factor", benchmark_cfg.get("effective_rate_factor", 2.0)))
    fs = factor * sampling_rate
    if fs <= 0:
        raise ValueError("Resolved sampling rate must be positive.")
    return fs


def _sample_signal_tensor_uniform(x_ref, t_dense, tau, fs):
    Ts = 1.0 / fs
    t_samples = np.arange(0.0, tau, Ts)
    _, n_periods, S = x_ref.shape
    x_samples = np.zeros((len(t_samples), n_periods, S), dtype=float)
    for p in range(n_periods):
        for s in range(S):
            x_samples[:, p, s] = np.interp(t_samples, t_dense, x_ref[:, p, s])
    return t_samples, x_samples


def _normalize_tensor_power(x, target_power):
    if target_power <= 0:
        raise ValueError("target_power must be positive.")
    x = np.asarray(x, dtype=float)
    p = np.mean(x ** 2)
    if np.isclose(p, 0.0):
        return np.array(x, copy=True)
    return x * np.sqrt(target_power / p)


def _compute_phase_coefficients_from_reference(x_ref, Tt, cfg, params, N):
    fourier_core = FourierCoefficientCore(T=params.tau, harmonics=N, sensor_nodes=params.S)
    an, bn, _ = fourier_core.calc_an_bn_dft(
        x_ref,
        Tt,
        normalize=cfg["signal"].get("normalize_dft", True),
        norm=cfg["signal"].get("normalization_target", 3.99),
    )
    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=x_ref.shape[1],
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001),
    )
    ta, tb = phase_core.calc_ta_tb(an, bn)
    return np.real(ta), np.real(tb), phase_core


def _reconstruct_from_ta_tb(ta, tb, t, params):
    n_periods, _, S = ta.shape
    x_hat = np.zeros((len(t), n_periods, S), dtype=float)
    w0 = 2.0 * np.pi / params.tau
    for p in range(n_periods):
        for s in range(S):
            x_hat[:, p, s] = recover_signal(ta[p, :, s], tb[p, :, s], t, w0)
    return x_hat


def _quantize_ta_tb_tensor(ta, tb, w0, M):
    ta_q = np.zeros_like(ta, dtype=float)
    tb_q = np.zeros_like(tb, dtype=float)
    n_periods, _, S = ta.shape
    for p in range(n_periods):
        for s in range(S):
            ta_q[p, :, s], tb_q[p, :, s] = quantize_ta_tb(ta[p, :, s], tb[p, :, s], w0, int(M))
    return ta_q, tb_q


# =============================================================================
# BENCHMARK / CS / PPM
# =============================================================================

def _run_benchmark_fdma_branch(cfg, params):
    if not _mode_enabled(cfg, "run_benchmark_fdma", _mode_enabled(cfg, "run_benchmark", True)):
        return np.nan, np.array([], dtype=object)
    fs = _resolve_benchmark_sampling_rate(cfg, params)
    peak_to_peak = float(cfg["signal"].get("peak_to_peak", 1.0))
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
    vals = []
    for M in M_vec:
        M_int = int(M)
        if M_int >= 2:
            vals.append((peak_to_peak ** 2) / (12.0 * float(M_int) ** 2))
    return (float(np.mean(vals)) if vals else np.nan), np.asarray(M_vec, dtype=object)


def _run_cs_fdma_branch(x_ref, t, cfg, params, rng, capacity_per_sensor):
    cs_cfg = _method_cfg(cfg, "cs")
    fs = _resolve_benchmark_sampling_rate(cfg, params)
    t_samples, x_samples = _sample_signal_tensor_uniform(x_ref, t, params.tau, fs)
    n_samples, _, S = x_samples.shape
    x_hat_samples = np.zeros_like(x_samples)

    measurements = []
    sampling_rates = []
    num_samples = []
    measurement_bits_all = []
    sparsity_eff_all = []
    quantized_fraction = []

    for s in range(S):
        bits_per_cycle = float(capacity_per_sensor[s] * params.tau)
        configured_measurements = cs_cfg.get("measurements", None)
        if configured_measurements is not None:
            n_measurements = max(1, min(int(configured_measurements), n_samples))
            measurement_bits = max(1, int(np.floor(bits_per_cycle / n_measurements)))
        else:
            full_bits = int(np.floor(bits_per_cycle / n_samples))
            if full_bits >= 1:
                n_measurements = n_samples
                measurement_bits = full_bits
            else:
                measurement_bits = 1
                n_measurements = int(np.floor(bits_per_cycle))
                if n_measurements < 1:
                    return np.nan, measurements, sampling_rates, num_samples, measurement_bits_all, sparsity_eff_all, quantized_fraction
                n_measurements = min(n_measurements, n_samples)

        configured_sparsity = cs_cfg.get("sparsity", None)
        if configured_sparsity is None:
            sparsity_eff = min(n_measurements, n_samples)
        else:
            sparsity_eff = min(int(configured_sparsity), n_measurements, n_samples)

        quantize_measurements = bool(cs_cfg.get("quantize_measurements", cs_cfg.get("quantize", False)))
        random_state = cs_cfg.get("random_state", cs_cfg.get("seed", None))
        if random_state is None:
            random_state = int(rng.integers(0, 2 ** 32 - 1))

        core = CSAcquisitionCore(
            n_measurements=n_measurements,
            sparsity=sparsity_eff,
            basis=cs_cfg.get("basis", "dct"),
            sensing_matrix=cs_cfg.get("sensing_matrix", cs_cfg.get("measurement_matrix", "gaussian")),
            random_state=random_state,
            normalize_dictionary_columns=cs_cfg.get("normalize_dictionary_columns", True),
            store_true_representation=cs_cfg.get("store_true_representation", False),
            quantize_measurements=quantize_measurements,
            measurement_bits=measurement_bits,
            quantization_mode=cs_cfg.get("measurement_quantization_mode", "uniform_midrise"),
            quantization_range=cs_cfg.get("measurement_quantization_range", "per_signal"),
            measurement_quantization_min=cs_cfg.get("measurement_quantization_min", None),
            measurement_quantization_max=cs_cfg.get("measurement_quantization_max", None),
            clip_quantization=cs_cfg.get("clip_quantization", True),
        )
        acq = core.acquire(x=x_samples[:, :, s:s + 1], t=t_samples)
        rec = core.reconstruct(acq)
        x_hat_samples[:, :, s:s + 1] = rec.reconstructed_signal

        measurements.append(int(n_measurements))
        sampling_rates.append(float(fs))
        num_samples.append(int(n_samples))
        measurement_bits_all.append(int(measurement_bits))
        sparsity_eff_all.append(int(sparsity_eff))
        quantized_fraction.append(1.0 if quantize_measurements else 0.0)

    x_hat_dense = sinc_reconstruct_from_samples(
        x_samples=x_hat_samples,
        t_samples=t_samples,
        t_eval=t,
        tau=params.tau,
        sample_period=1.0 / fs,
        periodic_replicas=PERIODIC_REPLICAS,
    )
    mse = float(np.mean((x_ref - x_hat_dense) ** 2))
    return mse, measurements, sampling_rates, num_samples, measurement_bits_all, sparsity_eff_all, quantized_fraction


def _run_ppm_fdma_branch(x_ref, t, cfg, params, rng):
    ppm_cfg = _method_cfg(cfg, "ppm")
    _, _, S = x_ref.shape
    Tt = float(cfg["signal"]["Tt"])
    x_hat = np.zeros_like(x_ref)
    fs_msg_list = []
    for s in range(S):
        B_sensor = float(params.B_per_sensor[s])
        timing = _resolve_ppm_timing_from_B_sensor(ppm_cfg, params, B_sensor, Tt)
        fs_msg = float(timing["fs_msg"])
        pulse_width = float(timing["pulse_width"])
        fs_msg_list.append(fs_msg)
        ppm_core = _make_ppm_core(ppm_cfg, fs_msg, pulse_width, B_sensor)
        mod_result = ppm_core.modulate(x=x_ref[:, :, s:s + 1], t=t)
        tx = np.array(mod_result.tx_waveform, dtype=float, copy=True)
        if ppm_cfg.get("normalize_sensor_power", True):
            tx = _normalize_tensor_power(tx, params.P)
        rx = apply_awgn(signal=tx, N0=params.N0, complex_noise=np.iscomplexobj(tx), rng=rng)
        demod_result = ppm_core.demodulate(y=rx, t=t, modulation_result=mod_result, reconstruct_continuous=True)
        x_hat[:, :, s:s + 1] = demod_result.recovered_continuous
    return float(np.mean((x_ref - x_hat) ** 2)), fs_msg_list


def _resolve_ppm_timing_from_B_sensor(ppm_cfg, params, B_sensor, Tt=None):
    if B_sensor <= 0:
        raise ValueError("B_sensor must be positive.")
    pulse_type = str(ppm_cfg.get("pulse_type", "raised_cosine")).lower()
    rolloff = float(ppm_cfg.get("rolloff", 0.99))
    requested_fs_msg = ppm_cfg.get("fs_msg", params.W)
    if requested_fs_msg is None or (isinstance(requested_fs_msg, str) and requested_fs_msg.lower() == "auto"):
        requested_fs_msg = params.W
    requested_fs_msg = float(requested_fs_msg)
    if requested_fs_msg <= 0:
        raise ValueError("ppm.fs_msg must be positive.")

    enforce_bandwidth = bool(ppm_cfg.get("enforce_bandwidth_from_B_sensor", True))
    if pulse_type in {"raised_cosine", "root_raised_cosine", "rrc", "rc"}:
        max_fs_msg_from_B_sensor = 2.0 * B_sensor / (1.0 + rolloff)
    else:
        max_fs_msg_from_B_sensor = B_sensor
    if not enforce_bandwidth:
        max_fs_msg_from_B_sensor = np.inf

    min_samples_per_symbol = int(ppm_cfg.get("min_samples_per_symbol", 8))
    if Tt is not None:
        max_fs_msg_from_time_grid = 1.0 / (min_samples_per_symbol * float(Tt))
    else:
        max_fs_msg_from_time_grid = np.inf

    pulse_width_explicit = "pulse_width" in ppm_cfg and ppm_cfg["pulse_width"] is not None
    pulse_width_fraction = float(ppm_cfg.get("pulse_width_fraction", 0.1))
    max_pulse_width_fraction = float(ppm_cfg.get("max_pulse_width_fraction_of_symbol", 0.9))
    if pulse_width_explicit:
        pulse_width = float(ppm_cfg["pulse_width"])
        max_fs_msg_from_pulse_width = max_pulse_width_fraction / pulse_width
    else:
        pulse_width = None
        max_fs_msg_from_pulse_width = np.inf

    enforce_pw_bw = bool(ppm_cfg.get("enforce_pulse_width_from_B_sensor", True))
    pw_factor = float(ppm_cfg.get("pulse_width_bandwidth_factor", 1.0))
    min_pw_from_B = np.nan
    if pulse_width_explicit and enforce_pw_bw:
        min_pw_from_B = pw_factor / B_sensor
        if pulse_width < min_pw_from_B:
            raise ValueError(
                f"Configured ppm.pulse_width={pulse_width:.6e} is too small for B_sensor={B_sensor:.6e}. "
                f"Minimum={min_pw_from_B:.6e}."
            )

    fs_msg = min(requested_fs_msg, max_fs_msg_from_B_sensor, max_fs_msg_from_time_grid, max_fs_msg_from_pulse_width)
    if fs_msg <= 0 or not np.isfinite(fs_msg):
        raise ValueError("Resolved PPM fs_msg must be positive and finite.")

    if not pulse_width_explicit:
        pulse_width = pulse_width_fraction / fs_msg
        if enforce_pw_bw:
            min_pw_from_B = pw_factor / B_sensor
            if pulse_width < min_pw_from_B:
                pulse_width = min_pw_from_B
                fs_msg = min(fs_msg, max_pulse_width_fraction / pulse_width)

    symbol_period = 1.0 / fs_msg
    if pulse_width >= symbol_period:
        raise ValueError("Resolved PPM pulse_width does not fit inside symbol period.")

    required_symbol_bw = (1.0 + rolloff) * fs_msg / 2.0 if pulse_type in {"raised_cosine", "root_raised_cosine", "rrc", "rc"} else fs_msg
    return {
        "requested_fs_msg": requested_fs_msg,
        "max_fs_msg_from_B_sensor": max_fs_msg_from_B_sensor,
        "max_fs_msg_from_time_grid": max_fs_msg_from_time_grid,
        "max_fs_msg_from_pulse_width": max_fs_msg_from_pulse_width,
        "fs_msg": fs_msg,
        "was_fs_msg_clipped": bool(fs_msg < requested_fs_msg),
        "pulse_width": float(pulse_width),
        "pulse_width_explicit": bool(pulse_width_explicit),
        "pulse_width_fraction": pulse_width_fraction,
        "max_pulse_width_fraction_of_symbol": max_pulse_width_fraction,
        "symbol_period": symbol_period,
        "min_samples_per_symbol": min_samples_per_symbol,
        "enforce_bandwidth_from_B_sensor": enforce_bandwidth,
        "enforce_pulse_width_from_B_sensor": enforce_pw_bw,
        "pulse_width_bandwidth_factor": pw_factor,
        "min_pulse_width_from_B_sensor": min_pw_from_B,
        "required_bandwidth_symbol": required_symbol_bw,
        "required_bandwidth_pulse": pw_factor / float(pulse_width),
        "B_sensor": B_sensor,
    }


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
        periodic_replicas=ppm_cfg.get("periodic_replicas", PERIODIC_REPLICAS),
        clip_recovered_to_unit_interval=ppm_cfg.get("clip_recovered_to_unit_interval", True),
    )
    try:
        return PPMCore(
            **kwargs,
            B_sensor=B_sensor,
            enforce_bandwidth_warning=ppm_cfg.get("enforce_bandwidth_warning", True),
            pulse_width_bandwidth_factor=ppm_cfg.get("pulse_width_bandwidth_factor", 1.0),
        )
    except TypeError:
        return PPMCore(**kwargs)


# =============================================================================
# SoD + FDMA
# =============================================================================

def _run_sod_fdma_branch(x_ref, t, cfg, params, budget_bits_per_sensor):
    sod_cfg = _method_cfg(cfg, "sod")
    threshold = float(sod_cfg.get("threshold", sod_cfg.get("delta", 0.1)))
    initial_event = bool(sod_cfg.get("initial_event", True))
    reconstruction_mode = sod_cfg.get("reconstruction_mode", "zero_order_hold")
    transmit_event_times = bool(sod_cfg.get("transmit_event_times", True))
    amplitude_range = sod_cfg.get("amplitude_quantization_range", "per_signal")
    amplitude_min = sod_cfg.get("amplitude_quantization_min", None)
    amplitude_max = sod_cfg.get("amplitude_quantization_max", None)
    time_range = sod_cfg.get("time_quantization_range", "period")
    time_min = sod_cfg.get("time_quantization_min", None)
    time_max = sod_cfg.get("time_quantization_max", None)
    poss = float(sod_cfg.get("quantization_poss", 0.5))
    clip = bool(sod_cfg.get("clip_quantization", True))
    min_amp_bits = int(sod_cfg.get("min_amplitude_bits", 1))
    min_time_bits = int(sod_cfg.get("min_time_bits", 1))
    time_bits_policy = str(sod_cfg.get("time_bits_policy", "balanced")).lower()

    detector = SoDAcquisitionCore(
        threshold=threshold,
        initial_event=initial_event,
        reconstruction_mode=reconstruction_mode,
        transmit_event_times=transmit_event_times,
        quantize_amplitudes=False,
        amplitude_bins=None,
        quantize_times=False,
        time_bins=None,
        clip_quantization=clip,
    )
    acq_raw = detector.acquire(x=x_ref, t=t)

    n_time, n_periods, S = x_ref.shape
    x_hat = np.zeros_like(x_ref)
    transmitted_count_all = []
    candidate_count_all = []
    payload_bits_all = []
    payload_budget_all = []
    feasible_flags = []
    amplitude_bits_all = []
    time_bits_all = []

    for p in range(n_periods):
        for s in range(S):
            key = f"period_{p}_sensor_{s}"
            event_times = np.asarray(acq_raw.event_times[key], dtype=float).reshape(-1)
            event_values = np.asarray(acq_raw.event_values[key], dtype=float).reshape(-1)
            K_candidate = int(len(event_times))
            if K_candidate < 1:
                event_times = np.asarray([float(t[0])], dtype=float)
                event_values = np.asarray([float(x_ref[0, p, s])], dtype=float)
                K_candidate = 1

            budget_bits = int(max(0, np.floor(budget_bits_per_sensor[s])))
            K_tx, amp_bits, t_bits, amp_bins, t_bins, payload_bits = _resolve_sod_fdma_payload_allocation(
                K_candidate=K_candidate,
                budget_bits=budget_bits,
                n_time=n_time,
                transmit_event_times=transmit_event_times,
                min_amplitude_bits=min_amp_bits,
                min_time_bits=min_time_bits,
                time_bits_policy=time_bits_policy,
            )
            selected_idx = _select_sod_event_indices(K_candidate, K_tx)
            tx_times = event_times[selected_idx]
            tx_values = event_values[selected_idx]

            amp_q_min, amp_q_max = _resolve_sod_amplitude_range(x_ref[:, p, s], tx_values, amplitude_range, amplitude_min, amplitude_max)
            tx_values_q = _quantize_sod_values(tx_values, amp_q_min, amp_q_max, amp_bins, poss, clip)

            if transmit_event_times:
                time_q_min, time_q_max = _resolve_sod_time_range(t, time_range, time_min, time_max)
                tx_times_q = _quantize_sod_values(tx_times, time_q_min, time_q_max, t_bins, poss, clip)
            else:
                tx_times_q = tx_times

            x_hat[:, p, s] = SoDAcquisitionCore._reconstruct_1d(tx_times_q, tx_values_q, t, reconstruction_mode)

            transmitted_count_all.append(int(K_tx))
            candidate_count_all.append(int(K_candidate))
            payload_bits_all.append(float(payload_bits))
            payload_budget_all.append(float(budget_bits))
            feasible_flags.append(1)
            amplitude_bits_all.append(int(amp_bits))
            time_bits_all.append(int(t_bits))

    mse = float(np.mean((x_ref - x_hat) ** 2))
    return (
        mse,
        transmitted_count_all,
        payload_bits_all,
        payload_budget_all,
        feasible_flags,
        1.0,
        candidate_count_all,
        amplitude_bits_all,
        time_bits_all,
    )


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
            amp_bins = 2 ** amp_bits if amp_bits > 0 else 1
            time_bins = 2 ** time_bits if time_bits > 0 else 1
            return int(K_tx), int(amp_bits), int(time_bits), int(amp_bins), int(time_bins), int(payload)

    amp_bits, time_bits = _split_sod_bits_per_event(budget_bits, transmit_event_times, 0, 0, max_time_bits, time_bits_policy, 1)
    if amp_bits + time_bits <= 0:
        amp_bits, time_bits = 1, 0
    amp_bins = 2 ** amp_bits if amp_bits > 0 else 1
    time_bins = 2 ** time_bits if time_bits > 0 else 1
    payload = amp_bits + time_bits
    return 1, int(amp_bits), int(time_bits), int(amp_bins), int(time_bins), int(payload)


def _split_sod_bits_per_event(bits_per_event, transmit_event_times, min_amplitude_bits, min_time_bits, max_time_bits, time_bits_policy, K_tx):
    bits_per_event = int(bits_per_event)
    if bits_per_event <= 0:
        return -1, -1
    min_amplitude_bits = int(max(0, min_amplitude_bits))
    min_time_bits = int(max(0, min_time_bits))
    max_time_bits = int(max(0, max_time_bits))

    if not transmit_event_times:
        if bits_per_event < min_amplitude_bits:
            return -1, -1
        return bits_per_event, 0

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
        missing = K_tx - len(idx)
        used = set(int(v) for v in idx)
        extra = [i for i in range(K_candidate) if i not in used]
        idx = np.concatenate([idx, np.asarray(extra[:missing], dtype=int)])
        idx = np.sort(idx)
    if len(idx) > K_tx:
        idx = idx[:K_tx]
    return idx.astype(int)


def _resolve_sod_amplitude_range(x_ref, values, mode, fixed_min, fixed_max):
    mode = str(mode).lower()
    if mode == "fixed":
        if fixed_min is None or fixed_max is None:
            raise ValueError("Fixed SoD amplitude range requires min and max.")
        vmin, vmax = float(fixed_min), float(fixed_max)
    elif mode in {"global", "per_signal"}:
        vmin, vmax = float(np.min(x_ref)), float(np.max(x_ref))
    elif mode == "per_events":
        vmin, vmax = float(np.min(values)), float(np.max(values))
    else:
        raise ValueError("Unsupported SoD amplitude_quantization_range.")
    if vmax < vmin:
        raise ValueError("Invalid SoD amplitude range.")
    return vmin, vmax


def _resolve_sod_time_range(t, mode, fixed_min, fixed_max):
    mode = str(mode).lower()
    t = np.asarray(t, dtype=float).reshape(-1)
    if mode == "fixed":
        if fixed_min is None or fixed_max is None:
            raise ValueError("Fixed SoD time range requires min and max.")
        vmin, vmax = float(fixed_min), float(fixed_max)
    elif mode == "period":
        dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
        vmin, vmax = float(t[0]), float(t[-1] + dt)
    else:
        raise ValueError("Unsupported SoD time_quantization_range.")
    if vmax < vmin:
        raise ValueError("Invalid SoD time range.")
    return vmin, vmax


def _quantize_sod_values(values, value_min, value_max, bins, poss, clip):
    values = np.asarray(values, dtype=float).reshape(-1)
    bins = int(max(1, bins))
    if np.isclose(value_max, value_min):
        return np.full(values.shape, float(value_min), dtype=float)
    if clip:
        values = np.clip(values, value_min, value_max)
    else:
        if np.any(values < value_min) or np.any(values > value_max):
            raise ValueError("SoD values outside quantization range.")
    return np.asarray(quantize(values, value_min, value_max, bins, poss=poss), dtype=float).reshape(-1)


# =============================================================================
# FRI + FDMA
# =============================================================================

def _cap_fri_budget_bits_for_numeric_safety(budget_bits: int, n_time: int, fri_cfg: Dict[str, Any]) -> int:
    budget_bits = int(max(0, budget_bits))
    n_time = int(max(1, n_time))
    K = int(fri_cfg.get("K", fri_cfg.get("num_innovations", 3)))
    K = max(1, K)
    max_location_bits = int(np.ceil(np.log2(float(max(n_time, 2)))))
    max_amplitude_bits = int(fri_cfg.get("max_amplitude_bits_safe", 24))
    if max_amplitude_bits < 1:
        raise ValueError("fri.max_amplitude_bits_safe must be >= 1.")
    max_total_budget = K * (max_location_bits + max_amplitude_bits)
    return int(min(budget_bits, max_total_budget))


def _run_fri_fdma_branch(x_ref, t, cfg, params, budget_bits_per_sensor):
    fri_cfg = _method_cfg(cfg, "fri")
    _, n_periods, S = x_ref.shape
    x_hat = np.zeros_like(x_ref)
    K_all = []
    bits_location_all = []
    bits_amplitude_all = []
    budget_bits_all = []
    raw_budget_bits_all = []
    feasible_flags = []
    target_ber = float(fri_cfg.get("bit_error_rate", cfg.get("comparison", {}).get("methods", {}).get("target_ber", 0.0)))

    for s in range(S):
        raw_budget_bits_s = int(max(0, np.floor(budget_bits_per_sensor[s])))
        budget_bits_s = _cap_fri_budget_bits_for_numeric_safety(raw_budget_bits_s, x_ref.shape[0], fri_cfg)
        cfg_s = copy.deepcopy(cfg)
        cfg_s.setdefault("acquisition", {}).setdefault("fri", {})["budget_bits"] = budget_bits_s
        cfg_s["acquisition"]["fri"]["bit_error_rate"] = target_ber
        fri_core = FRIAcquisition(cfg_s)
        out = fri_core.run(
            x_ref[:, :, s:s + 1],
            t=t,
            tau=params.tau,
            Tt=float(cfg["signal"]["Tt"]),
            budget_bits=budget_bits_s,
            bit_error_rate=target_ber,
            return_result=True,
        )
        x_hat[:, :, s:s + 1] = out.x_hat
        diag = out.diagnostics
        K = int(diag.get("K", fri_cfg.get("K", 3)))
        bits_location = diag.get("bits_location", np.nan)
        bits_amplitude = diag.get("bits_amplitude", np.nan)
        feasible = budget_bits_s > 0 or bool(fri_cfg.get("allow_zero_budget", False))
        for _ in range(n_periods):
            K_all.append(K)
            bits_location_all.append(float(bits_location) if bits_location is not None else np.nan)
            bits_amplitude_all.append(float(bits_amplitude) if bits_amplitude is not None else np.nan)
            budget_bits_all.append(float(budget_bits_s))
            raw_budget_bits_all.append(float(raw_budget_bits_s))
            feasible_flags.append(1 if feasible else 0)

    feasible_fraction = float(np.mean(feasible_flags)) if feasible_flags else np.nan
    if bool(fri_cfg.get("require_budget_feasible", True)):
        vals = []
        idx = 0
        for s in range(S):
            for p in range(n_periods):
                if feasible_flags[idx] == 1:
                    vals.append(np.mean((x_ref[:, p, s] - x_hat[:, p, s]) ** 2))
                idx += 1
        mse = float(np.mean(vals)) if vals else np.nan
    else:
        mse = float(np.mean((x_ref - x_hat) ** 2))
    return mse, K_all, bits_location_all, bits_amplitude_all, budget_bits_all, raw_budget_bits_all, feasible_flags, feasible_fraction


# =============================================================================
# RbCP / SFC
# =============================================================================

def _run_rbcp_branch(x_ref, t, cfg, params, N, M):
    if M is None:
        return np.nan
    M = int(M)
    if M < 2:
        return np.nan
    ta, tb, _ = _compute_phase_coefficients_from_reference(x_ref, float(cfg["signal"]["Tt"]), cfg, params, N)
    w0 = 2.0 * np.pi / params.tau
    ta_q, tb_q = _quantize_ta_tb_tensor(ta, tb, w0, M)
    x_hat = _reconstruct_from_ta_tb(ta_q, tb_q, t, params)
    return float(np.mean((x_ref - x_hat) ** 2))


def _run_sfc_family_branch(x_ref, t, cfg, params, N, sfc_channel, run_native_sfc, run_sfc_sed):
    if sfc_channel is None:
        return np.nan, np.nan, np.nan
    ta, tb, phase_core = _compute_phase_coefficients_from_reference(x_ref, float(cfg["signal"]["Tt"]), cfg, params, N)
    events = phase_core.ta_tb_to_events(ta, tb)
    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out
    mse_sfc = np.nan
    mse_sfc_sed = np.nan
    valid_fraction = np.nan
    if run_native_sfc:
        mse_sfc = _mse_from_sfc_events(events_est, x_ref, t, params, phase_core, None)
    if run_sfc_sed:
        n_periods = x_ref.shape[1]
        if events_est.shape[0] % n_periods != 0:
            raise ValueError("SFC event slots are not divisible by n_periods.")
        period_slots = events_est.shape[0] // n_periods
        sed_result = detect_semantic_errors(
            events_est=events_est,
            period_slots=period_slots,
            N=N,
            sensor_x_event=_build_sensor_x_event(params.S, N),
            discard_invalid_periods=cfg.get("sed", {}).get("discard_invalid_periods", True),
        )
        corrected, valid_mask = _extract_sed_outputs(sed_result)
        valid_fraction = float(np.mean(valid_mask))
        mse_sfc_sed = _mse_from_sfc_events(corrected, x_ref, t, params, phase_core, valid_mask) if np.any(valid_mask) else np.nan
    return mse_sfc, mse_sfc_sed, valid_fraction


def _mse_from_sfc_events(events_for_rec, x_ref, t, params, phase_core, period_valid_mask=None):
    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_for_rec)
    x_hat = _reconstruct_from_ta_tb(np.real(ta_rec), np.real(tb_rec), t, params)
    if period_valid_mask is None:
        return float(np.mean((x_ref - x_hat) ** 2))
    valid = np.asarray(period_valid_mask, dtype=bool)
    if not np.any(valid):
        return np.nan
    return float(np.nanmean((x_ref[:, valid, :] - x_hat[:, valid, :]) ** 2))


def _build_sfc_channel_for_B(cfg_B, N, S):
    cfg_sfc = copy.deepcopy(cfg_B)
    cfg_sfc.setdefault("channel", {})
    ch = cfg_sfc["channel"]
    ch["sensor_x_event"] = _build_sensor_x_event(S, N)
    ch["collision_mode"] = ch.get("collision_mode", "sum")
    ch["type"] = ch.get("type", "awgn")
    ch["detection_mode"] = ch.get("detection_mode", "threshold")
    ch["score_threshold"] = ch.get("score_threshold", cfg_sfc["system"]["L"])
    if ch.get("threshold", None) is None:
        ch.pop("threshold", None)
        ch["threshold_factor"] = ch.get("threshold_factor", 0.5)
    else:
        ch["threshold"] = float(ch["threshold"])
    ch["candidate_selection"] = ch.get("candidate_selection", "global_frame_fit")
    ch.setdefault("global", {})
    ch["global"].setdefault("allow_empty", True)
    ch["global"].setdefault("strategy", "local_only")
    ch["global"].setdefault("restarts", 100)
    ch["global"].setdefault("max_iter", 80)
    ch["global"].setdefault("observed_mode", "abs")
    ch["global"].setdefault("event_penalty", 0.0)
    ch["global"].setdefault("residual_threshold", 0.0)
    cfg_sfc.setdefault("reproducibility", {})
    cfg_sfc["reproducibility"].setdefault("seed", cfg_B.get("monte_carlo", {}).get("seed", 12345))
    return SFCChannel(cfg_sfc)


def _build_sensor_x_event(S, N):
    num_event_ids = 2 * N * S
    sensor_x_event = np.zeros((S, num_event_ids), dtype=float)
    for s in range(S):
        sensor_x_event[s, 2 * s * N:2 * (s + 1) * N] = 1.0
    return sensor_x_event


def _extract_sed_outputs(sed_result):
    if isinstance(sed_result, dict):
        return sed_result["corrected_events_est"], np.asarray(sed_result["period_valid_mask"], dtype=bool)
    return sed_result.corrected_events_est, np.asarray(sed_result.period_valid_mask, dtype=bool)


# =============================================================================
# ACCUMULATION / ROWS
# =============================================================================

def _new_accumulator():
    return {
        "mse_cs_fdma": [0.0, 0],
        "mse_ppm_fdma": [0.0, 0],
        "mse_sod_fdma": [0.0, 0],
        "mse_fri_fdma": [0.0, 0],
        "mse_rbcp": [0.0, 0],
        "mse_rbcp_time": [0.0, 0],
        "mse_sfc": [0.0, 0],
        "mse_sfc_sed": [0.0, 0],
        "sfc_sed_valid_fraction": [0.0, 0],
        "sod_fdma_feasible_fraction": [0.0, 0],
        "fri_fdma_feasible_fraction": [0.0, 0],
        "cs_measurements": [],
        "cs_sampling_rate": [],
        "cs_num_samples": [],
        "cs_measurement_bits": [],
        "cs_sparsity_eff": [],
        "cs_quantized_fraction": [],
        "ppm_fs_msg": [],
        "sod_num_events": [],
        "sod_candidate_events": [],
        "sod_payload_bits": [],
        "sod_payload_budget_bits": [],
        "sod_feasible_flags": [],
        "sod_amplitude_bits": [],
        "sod_time_bits": [],
        "fri_K": [],
        "fri_bits_location": [],
        "fri_bits_amplitude": [],
        "fri_budget_bits": [],
        "fri_raw_budget_bits": [],
        "fri_feasible_flags": [],
    }


def _accumulate_trial(accum, trial):
    scalar_keys = [
        "mse_cs_fdma",
        "mse_ppm_fdma",
        "mse_sod_fdma",
        "mse_fri_fdma",
        "mse_rbcp",
        "mse_rbcp_time",
        "mse_sfc",
        "mse_sfc_sed",
        "sfc_sed_valid_fraction",
        "sod_fdma_feasible_fraction",
        "fri_fdma_feasible_fraction",
    ]
    list_keys = [k for k in accum.keys() if k not in scalar_keys]
    for key in scalar_keys:
        value = trial.get(key, np.nan)
        if np.isfinite(value):
            accum[key][0] += float(value)
            accum[key][1] += 1
    for key in list_keys:
        accum[key].extend(trial.get(key, []))


def _mean_from_accum(accum, key):
    total, count = accum[key]
    return total / count if count > 0 else np.nan


def _array_stat(values, stat):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if stat == "min":
        return float(np.min(arr))
    if stat == "max":
        return float(np.max(arr))
    if stat == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unsupported stat: {stat}")


def _build_result_row(B, params, cfg, n_trials, mse_benchmark_fdma, M_benchmark_min, M_benchmark_mean, M_benchmark_max, capacity_per_sensor,
                      budget_bits_per_sensor, accum):
    return {
        "B": float(B),
        "N0": float(params.N0),
        "SNR_total": float(params.SNR),
        "SNR_total_dB": float(params.SNR_dB),
        "SNR_sensor_min": float(np.min(params.SNR_per_sensor)),
        "SNR_sensor_max": float(np.max(params.SNR_per_sensor)),
        "fdma_capacity_sensor_min": float(np.min(capacity_per_sensor)),
        "fdma_capacity_sensor_max": float(np.max(capacity_per_sensor)),
        "fdma_budget_bits_sensor_min": float(np.min(budget_bits_per_sensor)),
        "fdma_budget_bits_sensor_max": float(np.max(budget_bits_per_sensor)),
        "M_benchmark_min": _safe_table_value(M_benchmark_min),
        "M_benchmark_mean": M_benchmark_mean,
        "M_benchmark_max": _safe_table_value(M_benchmark_max),
        "M_rbcp": _safe_table_value(params.M_rbcp),
        "M_time": int(params.M_time),
        "mse_benchmark_fdma": mse_benchmark_fdma,
        "mse_cs_fdma": _mean_from_accum(accum, "mse_cs_fdma"),
        "mse_ppm_fdma": _mean_from_accum(accum, "mse_ppm_fdma"),
        "mse_sod_fdma": _mean_from_accum(accum, "mse_sod_fdma"),
        "mse_fri_fdma": _mean_from_accum(accum, "mse_fri_fdma"),
        "mse_rbcp": _mean_from_accum(accum, "mse_rbcp"),
        "mse_rbcp_time": _mean_from_accum(accum, "mse_rbcp_time"),
        "mse_sfc": _mean_from_accum(accum, "mse_sfc"),
        "mse_sfc_sed": _mean_from_accum(accum, "mse_sfc_sed"),
        "sfc_sed_valid_fraction": _mean_from_accum(accum, "sfc_sed_valid_fraction"),
        "cs_measurements_min": _array_stat(accum["cs_measurements"], "min"),
        "cs_measurements_max": _array_stat(accum["cs_measurements"], "max"),
        "cs_sampling_rate": _array_stat(accum["cs_sampling_rate"], "mean"),
        "cs_num_samples": _array_stat(accum["cs_num_samples"], "mean"),
        "cs_measurement_bits_min": _array_stat(accum["cs_measurement_bits"], "min"),
        "cs_measurement_bits_max": _array_stat(accum["cs_measurement_bits"], "max"),
        "cs_sparsity_eff_min": _array_stat(accum["cs_sparsity_eff"], "min"),
        "cs_sparsity_eff_max": _array_stat(accum["cs_sparsity_eff"], "max"),
        "cs_quantize_measurements": bool(_method_cfg(cfg, "cs").get("quantize_measurements", _method_cfg(cfg, "cs").get("quantize", False))),
        "cs_quantized_fraction": _array_stat(accum["cs_quantized_fraction"], "mean"),
        "ppm_fs_msg_min": _array_stat(accum["ppm_fs_msg"], "min"),
        "ppm_fs_msg_max": _array_stat(accum["ppm_fs_msg"], "max"),
        "sod_candidate_events_min": _array_stat(accum["sod_candidate_events"], "min"),
        "sod_candidate_events_mean": _array_stat(accum["sod_candidate_events"], "mean"),
        "sod_candidate_events_max": _array_stat(accum["sod_candidate_events"], "max"),
        "sod_num_events_min": _array_stat(accum["sod_num_events"], "min"),
        "sod_num_events_mean": _array_stat(accum["sod_num_events"], "mean"),
        "sod_num_events_max": _array_stat(accum["sod_num_events"], "max"),
        "sod_payload_bits_mean": _array_stat(accum["sod_payload_bits"], "mean"),
        "sod_payload_budget_bits_mean": _array_stat(accum["sod_payload_budget_bits"], "mean"),
        "sod_amplitude_bits_mean": _array_stat(accum["sod_amplitude_bits"], "mean"),
        "sod_time_bits_mean": _array_stat(accum["sod_time_bits"], "mean"),
        "sod_fdma_feasible_fraction": _mean_from_accum(accum, "sod_fdma_feasible_fraction"),
        "fri_K_mean": _array_stat(accum["fri_K"], "mean"),
        "fri_bits_location_mean": _array_stat(accum["fri_bits_location"], "mean"),
        "fri_bits_amplitude_mean": _array_stat(accum["fri_bits_amplitude"], "mean"),
        "fri_budget_bits_mean": _array_stat(accum["fri_budget_bits"], "mean"),
        "fri_raw_budget_bits_mean": _array_stat(accum["fri_raw_budget_bits"], "mean"),
        "fri_fdma_feasible_fraction": _mean_from_accum(accum, "fri_fdma_feasible_fraction"),
        "num_trials": int(n_trials),
    }


# =============================================================================
# PRINTING / GENERAL HELPERS
# =============================================================================

def _print_bandwidth_point_summary(row):
    print("[INFO] ================= METHOD PARAMETER SUMMARY =================")
    print(f"[INFO] B_total = {row.get('B', np.nan):.6e}")
    print("[INFO] FDMA budget:")
    print(
        f"[INFO]   capacity_per_sensor min/max = {row.get('fdma_capacity_sensor_min', np.nan):.6e} / {row.get('fdma_capacity_sensor_max', np.nan):.6e}")
    print(
        f"[INFO]   budget_bits_per_sensor min/max = {row.get('fdma_budget_bits_sensor_min', np.nan):.6e} / {row.get('fdma_budget_bits_sensor_max', np.nan):.6e}")
    print("[INFO] Benchmark / Nyquist + FDMA:")
    print(
        f"[INFO]   M_benchmark min/mean/max = {row.get('M_benchmark_min', np.nan)} / {row.get('M_benchmark_mean', np.nan)} / {row.get('M_benchmark_max', np.nan)}")
    print(f"[INFO]   mse_benchmark_fdma = {row.get('mse_benchmark_fdma', np.nan):.8e}")
    print("[INFO] CS + FDMA:")
    print(f"[INFO]   measurements min/max = {row.get('cs_measurements_min', np.nan):.6e} / {row.get('cs_measurements_max', np.nan):.6e}")
    print(f"[INFO]   measurement_bits min/max = {row.get('cs_measurement_bits_min', np.nan):.6e} / {row.get('cs_measurement_bits_max', np.nan):.6e}")
    print(f"[INFO]   sparsity_eff min/max = {row.get('cs_sparsity_eff_min', np.nan):.6e} / {row.get('cs_sparsity_eff_max', np.nan):.6e}")
    print(f"[INFO]   sampling_rate mean = {row.get('cs_sampling_rate', np.nan):.6e}")
    print(f"[INFO]   num_samples mean = {row.get('cs_num_samples', np.nan):.6e}")
    print(f"[INFO]   quantized_fraction = {row.get('cs_quantized_fraction', np.nan):.6e}")
    print(f"[INFO]   mse_cs_fdma = {row.get('mse_cs_fdma', np.nan):.8e}")
    print("[INFO] PPM + FDMA:")
    print(f"[INFO]   fs_msg min/max = {row.get('ppm_fs_msg_min', np.nan):.6e} / {row.get('ppm_fs_msg_max', np.nan):.6e}")
    print(f"[INFO]   mse_ppm_fdma = {row.get('mse_ppm_fdma', np.nan):.8e}")
    print("[INFO] SoD + FDMA:")
    print(
        f"[INFO]   candidate_events min/mean/max = {row.get('sod_candidate_events_min', np.nan):.6e} / {row.get('sod_candidate_events_mean', np.nan):.6e} / {row.get('sod_candidate_events_max', np.nan):.6e}")
    print(
        f"[INFO]   transmitted_events min/mean/max = {row.get('sod_num_events_min', np.nan):.6e} / {row.get('sod_num_events_mean', np.nan):.6e} / {row.get('sod_num_events_max', np.nan):.6e}")
    print(f"[INFO]   amplitude_bits/time_bits mean = {row.get('sod_amplitude_bits_mean', np.nan):.6e} / {row.get('sod_time_bits_mean', np.nan):.6e}")
    print(f"[INFO]   payload_bits mean = {row.get('sod_payload_bits_mean', np.nan):.6e}")
    print(f"[INFO]   payload_budget_bits mean = {row.get('sod_payload_budget_bits_mean', np.nan):.6e}")
    print(f"[INFO]   feasible_fraction = {row.get('sod_fdma_feasible_fraction', np.nan):.6e}")
    print(f"[INFO]   mse_sod_fdma = {row.get('mse_sod_fdma', np.nan):.8e}")
    print("[INFO] FRI-inspired + FDMA:")
    print(f"[INFO]   K mean = {row.get('fri_K_mean', np.nan):.6e}")
    print(f"[INFO]   bits_location mean = {row.get('fri_bits_location_mean', np.nan):.6e}")
    print(f"[INFO]   bits_amplitude mean = {row.get('fri_bits_amplitude_mean', np.nan):.6e}")
    print(f"[INFO]   budget_bits used/raw mean = {row.get('fri_budget_bits_mean', np.nan):.6e} / {row.get('fri_raw_budget_bits_mean', np.nan):.6e}")
    print(f"[INFO]   feasible_fraction = {row.get('fri_fdma_feasible_fraction', np.nan):.6e}")
    print(f"[INFO]   mse_fri_fdma = {row.get('mse_fri_fdma', np.nan):.8e}")
    print("[INFO] RbCP / RbCP_time:")
    print(f"[INFO]   M_RbCP = {row.get('M_rbcp', np.nan)}")
    print(f"[INFO]   M_time = {row.get('M_time', np.nan)}")
    print(f"[INFO]   mse_rbcp = {row.get('mse_rbcp', np.nan):.8e}")
    print(f"[INFO]   mse_rbcp_time = {row.get('mse_rbcp_time', np.nan):.8e}")
    print("[INFO] SFC / SFC + SED:")
    print(f"[INFO]   mse_sfc = {row.get('mse_sfc', np.nan):.8e}")
    print(f"[INFO]   mse_sfc_sed = {row.get('mse_sfc_sed', np.nan):.8e}")
    print(f"[INFO]   sfc_sed_valid_fraction = {row.get('sfc_sed_valid_fraction', np.nan):.6e}")
    print("[INFO] ===========================================================")


def _safe_min_int(values):
    return int(min(int(v) for v in list(values)))


def _safe_max_int(values):
    return int(max(int(v) for v in list(values)))


def _safe_mean_float(values):
    vals = [int(v) for v in list(values)]
    return float(sum(vals) / len(vals))


def _fits_int64(value):
    try:
        v = int(value)
    except Exception:
        return False
    return -(2 ** 63) <= v <= 2 ** 63 - 1


def _safe_table_value(value):
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    if _fits_int64(value):
        return int(value)
    return str(value)


def _mode_enabled(cfg, key, default):
    return bool(cfg.get("mode", {}).get(key, default))


def _method_cfg(cfg, method):
    merged = {}
    merged.update(cfg.get(method, {}))
    merged.update(cfg.get("acquisition", {}).get(method, {}))
    return merged


# =============================================================================
# SAVE UTILITY
# =============================================================================

def save_dat_file(df: pd.DataFrame, path: str, delimiter: str = "\t"):
    df.to_csv(path, sep=delimiter, index=False, float_format="%.8e")


__all__ = ["generate_fair_methods_comparison_vs_B_data", "save_dat_file"]
