"""
sfc/pipelines/fair_methods_comparison_vs_B.py

Fair comparison pipeline versus total bandwidth B.

Compared methods
----------------
1. Benchmark + FDMA
2. CS + FDMA
3. PPM + FDMA
4. RbCP
5. RbCP_time
6. SFC
7. SFC + SED

Physical convention
-------------------
The project-wide convention is:

    P  = average transmit power per sensor
    B  = total system bandwidth
    N0 = universal noise spectral-density / noise parameter

Derived quantities:

    B_s     = alpha_s * B
    SNR_s   = P / (B_s * N0)
    C_s     = B_s * log2(1 + SNR_s)

Method-specific interpretation
------------------------------
Benchmark + FDMA:
    - each sensor gets bandwidth B_s
    - each sensor has SNR_s
    - feasible quantization bins M_s are derived from the per-sensor budget
    - analytical absolute MSE:
          MSE_s = peak_to_peak^2 / (12 M_s^2)

CS + FDMA:
    - each sensor gets bandwidth B_s
    - each sensor has SNR_s
    - available bits per cycle are C_s * tau
    - number of CS measurements is derived from this bit budget
    - IMPORTANT: this branch models CS measurements as a capacity-limited
      digital payload. Transmission noise is accounted for through the
      admissible number of measurements, not by adding AWGN directly to
      the CS measurement vector.

PPM + FDMA:
    - each sensor gets bandwidth B_s
    - PPM message rate fs_msg is constrained by B_s
    - PPM message rate fs_msg is also constrained by the numerical time grid
    - AWGN uses SNR_s
    - PPM waveform is normalized to average power P per sensor

RbCP:
    - uses M_RbCP derived from the per-sensor bandwidth/sensor-capacity budget
    - common M_RbCP is min_s M_RbCP,s
    - reconstructs from quantized phase coefficients

RbCP_time:
    - uses M_time derived from the SFC event-time grid using total B
    - reconstructs from phase coefficients quantized using M_time

SFC:
    - SFC uses total bandwidth B
    - SFC does not use B_s
    - SFC channel receives the full system config

SFC + SED:
    - uses the same SFC detected events as native SFC
    - applies semantic error detection/correction before reconstruction
    - reports valid-period fraction

IMPORTANT SOURCE-BANDWIDTH CONVENTION
-------------------------------------
The source signal is filtered using the configured source bandwidth:

    signal.W

The pipeline must NOT redefine the filtering bandwidth from N using:

    W_eff = 2 * N / tau

The role of N is to define the number of representation harmonics.
The role of W is to define the source-signal bandwidth used by the signal
filter.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from sfc.core.acquisition.cs import CSAcquisitionCore
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.modulation.ppm import PPMCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.semantic_error_detection import detect_semantic_errors
from sfc.core.system_parameters import (
    build_derived_system_parameters,
    compute_benchmark_M_per_sensor,
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_fair_methods_comparison_vs_B_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate fair-methods comparison data versus total bandwidth B.
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    print("[INFO] Starting fair_methods_comparison_vs_B data generation")
    print(f"[INFO] Monte Carlo seed = {cfg['monte_carlo']['seed']}")
    print(f"[INFO] Trials per B = {cfg['monte_carlo']['interactions']}")

    b_cfg = cfg["sweep"]["B"]
    b_start = float(b_cfg["start"])
    b_stop = float(b_cfg["stop"])
    b_step = float(b_cfg["step"])
    include_stop = bool(b_cfg.get("include_stop", True))

    if include_stop:
        b_values = np.arange(b_start, b_stop + 0.5 * b_step, b_step)
    else:
        b_values = np.arange(b_start, b_stop, b_step)

    results = []

    for B in b_values:
        cfg_B = copy.deepcopy(cfg)
        cfg_B["system"]["B"] = float(B)

        params = build_derived_system_parameters(cfg_B)

        N = int(cfg_B["signal"].get("N_override", params.N))
        n_trials = int(cfg_B["monte_carlo"]["interactions"])

        # ---------------------------------------------------------------------
        # Benchmark + FDMA is analytical, so compute once per B.
        # ---------------------------------------------------------------------
        mse_benchmark_fdma, M_benchmark_per_sensor = _run_benchmark_fdma_branch(
            cfg=cfg_B,
            params=params
        )

        M_benchmark_arr = np.asarray(M_benchmark_per_sensor, dtype=object)
        M_benchmark_min = (
            _safe_min_int(M_benchmark_arr)
            if M_benchmark_arr.size > 0
            else np.nan
        )
        M_benchmark_max = (
            _safe_max_int(M_benchmark_arr)
            if M_benchmark_arr.size > 0
            else np.nan
        )
        M_benchmark_mean = (
            _safe_mean_float(M_benchmark_arr)
            if M_benchmark_arr.size > 0
            else np.nan
        )

        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] B_total = {params.B:.6e}")
        print(f"[INFO] P_per_sensor = {params.P:.6e}")
        print(f"[INFO] N0 = {params.N0:.6e}")
        print(f"[INFO] S = {params.S}")
        print(f"[INFO] tau = {params.tau:.6e}")
        print(f"[INFO] W = {params.W:.6e}")
        print(f"[INFO] N = {N}")
        print(f"[INFO] B_per_sensor = {params.B_per_sensor}")
        print(f"[INFO] SNR_total = {params.SNR:.6e}")
        print(f"[INFO] SNR_total_dB = {params.SNR_dB:.6f}")
        print(f"[INFO] SNR_per_sensor = {params.SNR_per_sensor}")
        print(f"[INFO] SNR_per_sensor_dB = {params.SNR_per_sensor_dB}")
        print(f"[INFO] M_benchmark_per_sensor = {M_benchmark_per_sensor}")
        print(f"[INFO] M_benchmark_min = {M_benchmark_min}")
        print(f"[INFO] M_benchmark_mean = {M_benchmark_mean}")
        print(f"[INFO] M_benchmark_max = {M_benchmark_max}")
        print(f"[INFO] M_RbCP = {params.M_rbcp}")
        print(f"[INFO] M_time = {params.M_time}")

        mse_cs_sum = 0.0
        mse_ppm_sum = 0.0
        mse_rbcp_sum = 0.0
        mse_rbcp_time_sum = 0.0
        mse_sfc_sum = 0.0
        mse_sfc_sed_sum = 0.0

        cs_count = 0
        ppm_count = 0
        rbcp_count = 0
        rbcp_time_count = 0
        sfc_count = 0
        sfc_sed_count = 0

        sfc_sed_valid_sum = 0.0

        cs_measurements_all = []
        ppm_fs_msg_all = []

        run_sfc = cfg_B.get("mode", {}).get("run_sfc", True)
        run_sfc_sed = cfg_B.get("mode", {}).get("run_sfc_sed", True)

        sfc_channel = None
        if run_sfc or run_sfc_sed:
            sfc_channel = _build_sfc_channel_for_B(
                cfg_B=cfg_B,
                N=N,
                S=params.S
            )

        for i in range(n_trials):
            trial = _run_one_trial(
                cfg=cfg_B,
                rng=rng,
                params=params,
                N=N,
                sfc_channel=sfc_channel
            )

            if np.isfinite(trial["mse_cs_fdma"]):
                mse_cs_sum += trial["mse_cs_fdma"]
                cs_count += 1

            if np.isfinite(trial["mse_ppm_fdma"]):
                mse_ppm_sum += trial["mse_ppm_fdma"]
                ppm_count += 1

            if np.isfinite(trial["mse_rbcp"]):
                mse_rbcp_sum += trial["mse_rbcp"]
                rbcp_count += 1

            if np.isfinite(trial["mse_rbcp_time"]):
                mse_rbcp_time_sum += trial["mse_rbcp_time"]
                rbcp_time_count += 1

            if np.isfinite(trial["mse_sfc"]):
                mse_sfc_sum += trial["mse_sfc"]
                sfc_count += 1

            if np.isfinite(trial["mse_sfc_sed"]):
                mse_sfc_sed_sum += trial["mse_sfc_sed"]
                sfc_sed_count += 1

            if np.isfinite(trial["sfc_sed_valid_fraction"]):
                sfc_sed_valid_sum += trial["sfc_sed_valid_fraction"]

            cs_measurements_all.extend(trial["cs_measurements"])
            ppm_fs_msg_all.extend(trial["ppm_fs_msg"])

            if (i + 1) % max(1, n_trials // 5) == 0:
                print(f"[INFO] Trial progress: {i + 1}/{n_trials}")

        mse_cs = mse_cs_sum / cs_count if cs_count > 0 else np.nan
        mse_ppm = mse_ppm_sum / ppm_count if ppm_count > 0 else np.nan
        mse_rbcp = mse_rbcp_sum / rbcp_count if rbcp_count > 0 else np.nan
        mse_rbcp_time = (
            mse_rbcp_time_sum / rbcp_time_count
            if rbcp_time_count > 0
            else np.nan
        )
        mse_sfc = mse_sfc_sum / sfc_count if sfc_count > 0 else np.nan
        mse_sfc_sed = (
            mse_sfc_sed_sum / sfc_sed_count
            if sfc_sed_count > 0
            else np.nan
        )

        sfc_sed_valid_fraction = (
            sfc_sed_valid_sum / n_trials
            if run_sfc_sed
            else np.nan
        )

        cs_measurements_all = np.asarray(cs_measurements_all, dtype=float)
        ppm_fs_msg_all = np.asarray(ppm_fs_msg_all, dtype=float)

        cs_measurements_min = (
            float(np.min(cs_measurements_all))
            if cs_measurements_all.size > 0
            else np.nan
        )
        cs_measurements_max = (
            float(np.max(cs_measurements_all))
            if cs_measurements_all.size > 0
            else np.nan
        )

        ppm_fs_msg_min = (
            float(np.min(ppm_fs_msg_all))
            if ppm_fs_msg_all.size > 0
            else np.nan
        )
        ppm_fs_msg_max = (
            float(np.max(ppm_fs_msg_all))
            if ppm_fs_msg_all.size > 0
            else np.nan
        )

        _print_metric("mse_benchmark_fdma", mse_benchmark_fdma)
        _print_metric("mse_cs_fdma", mse_cs)
        _print_metric("mse_ppm_fdma", mse_ppm)
        _print_metric("mse_rbcp", mse_rbcp)
        _print_metric("mse_rbcp_time", mse_rbcp_time)
        _print_metric("mse_sfc", mse_sfc)
        _print_metric("mse_sfc_sed", mse_sfc_sed)
        _print_metric("sfc_sed_valid_fraction", sfc_sed_valid_fraction)

        results.append({
            "B": float(B),
            "SNR_total": float(params.SNR),
            "SNR_total_dB": float(params.SNR_dB),
            "SNR_sensor_min": float(np.min(params.SNR_per_sensor)),
            "SNR_sensor_max": float(np.max(params.SNR_per_sensor)),
            "M_benchmark_min": _safe_table_value(M_benchmark_min),
            "M_benchmark_mean": M_benchmark_mean,
            "M_benchmark_max": _safe_table_value(M_benchmark_max),
            "M_rbcp": _safe_table_value(params.M_rbcp),
            "M_time": int(params.M_time),
            "mse_benchmark_fdma": mse_benchmark_fdma,
            "mse_cs_fdma": mse_cs,
            "mse_ppm_fdma": mse_ppm,
            "mse_rbcp": mse_rbcp,
            "mse_rbcp_time": mse_rbcp_time,
            "mse_sfc": mse_sfc,
            "mse_sfc_sed": mse_sfc_sed,
            "sfc_sed_valid_fraction": sfc_sed_valid_fraction,
            "cs_measurements_min": cs_measurements_min,
            "cs_measurements_max": cs_measurements_max,
            "ppm_fs_msg_min": ppm_fs_msg_min,
            "ppm_fs_msg_max": ppm_fs_msg_max,
            "num_trials": n_trials,
        })

    return pd.DataFrame(results)


# =============================================================================
# ONE MONTE CARLO TRIAL
# =============================================================================

def _run_one_trial(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    params,
    N: int,
    sfc_channel: Optional[SFCChannel] = None
) -> Dict[str, Any]:
    """
    Run one fair-comparison Monte Carlo trial.
    """

    n_periods = int(
        cfg.get("signal", {}).get(
            "n_periods",
            cfg.get("simulation", {}).get("n_periods", 1)
        )
    )

    tau = params.tau
    Tt = cfg["signal"]["Tt"]

    t = np.arange(0.0, tau, Tt)

    x_ref = _generate_common_source(
        cfg=cfg,
        rng=rng,
        params=params,
        N=N,
        t=t,
        n_periods=n_periods
    )

    mse_cs_fdma = np.nan
    mse_ppm_fdma = np.nan
    mse_rbcp = np.nan
    mse_rbcp_time = np.nan
    mse_sfc = np.nan
    mse_sfc_sed = np.nan
    sfc_sed_valid_fraction = np.nan

    cs_measurements = []
    ppm_fs_msg = []

    if cfg.get("mode", {}).get("run_cs_fdma", True):
        mse_cs_fdma, cs_measurements = _run_cs_fdma_branch(
            x_ref=x_ref,
            t=t,
            cfg=cfg,
            params=params,
            rng=rng
        )

    if cfg.get("mode", {}).get("run_ppm_fdma", True):
        mse_ppm_fdma, ppm_fs_msg = _run_ppm_fdma_branch(
            x_ref=x_ref,
            t=t,
            cfg=cfg,
            params=params,
            rng=rng
        )

    if cfg.get("mode", {}).get("run_rbcp", True):
        mse_rbcp = _run_rbcp_branch(
            x_ref=x_ref,
            t=t,
            cfg=cfg,
            params=params,
            N=N,
            M=params.M_rbcp
        )

    if cfg.get("mode", {}).get("run_rbcp_time", True):
        mse_rbcp_time = _run_rbcp_branch(
            x_ref=x_ref,
            t=t,
            cfg=cfg,
            params=params,
            N=N,
            M=params.M_time
        )

    run_sfc = cfg.get("mode", {}).get("run_sfc", True)
    run_sfc_sed = cfg.get("mode", {}).get("run_sfc_sed", True)

    if run_sfc or run_sfc_sed:
        mse_sfc, mse_sfc_sed, sfc_sed_valid_fraction = _run_sfc_family_branch(
            x_ref=x_ref,
            t=t,
            cfg=cfg,
            params=params,
            N=N,
            sfc_channel=sfc_channel,
            run_native_sfc=run_sfc,
            run_sfc_sed=run_sfc_sed
        )

    return {
        "mse_cs_fdma": mse_cs_fdma,
        "mse_ppm_fdma": mse_ppm_fdma,
        "mse_rbcp": mse_rbcp,
        "mse_rbcp_time": mse_rbcp_time,
        "mse_sfc": mse_sfc,
        "mse_sfc_sed": mse_sfc_sed,
        "sfc_sed_valid_fraction": sfc_sed_valid_fraction,
        "cs_measurements": cs_measurements,
        "ppm_fs_msg": ppm_fs_msg,
    }


# =============================================================================
# COMMON SOURCE GENERATION
# =============================================================================

def _generate_common_source(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    params,
    N: int,
    t: np.ndarray,
    n_periods: int
) -> np.ndarray:
    """
    Generate common band-limited source signals.

    Output shape:
        (time, periods, sensors)
    """

    _ = N

    Tt = cfg["signal"]["Tt"]
    tau = params.tau
    S = params.S

    n_time = len(t)
    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=(n_time, n_periods, S))
    elif dist == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=(n_time, n_periods, S))
    else:
        raise ValueError("signal.distribution must be 'uniform' or 'gaussian'.")

    W_filter = float(cfg["signal"].get("W", params.W))

    if W_filter <= 0:
        raise ValueError("signal.W must be positive.")

    x_filtered = np.zeros_like(x_raw)

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw[:, p, s],
                W_filter,
                Tt,
                tau
            )

    peak_to_peak = cfg["signal"].get("peak_to_peak", 0.0)

    if peak_to_peak != 0:
        for p in range(n_periods):
            for s in range(S):
                current_p2p = np.max(x_filtered[:, p, s]) - np.min(x_filtered[:, p, s])
                if current_p2p != 0:
                    x_filtered[:, p, s] *= peak_to_peak / current_p2p

    dc_enabled = cfg.get("dc", {}).get("enabled", False)
    x_out = np.zeros_like(x_filtered)

    for p in range(n_periods):
        for s in range(S):
            if dc_enabled:
                x_out[:, p, s] = x_filtered[:, p, s]
            else:
                dc = Tt * np.sum(x_filtered[:, p, s]) / tau
                x_out[:, p, s] = x_filtered[:, p, s] - dc

    return x_out


# =============================================================================
# SHARED PHASE / FOURIER HELPERS
# =============================================================================

def _compute_phase_coefficients_from_reference(
    x_ref: np.ndarray,
    Tt: float,
    cfg: Dict[str, Any],
    params,
    N: int
):
    """
    Compute ta/tb and phase core from a source tensor.
    """

    n_periods = x_ref.shape[1]

    fourier_core = FourierCoefficientCore(
        T=params.tau,
        harmonics=N,
        sensor_nodes=params.S
    )

    an, bn, _ = fourier_core.calc_an_bn_dft(
        x_ref,
        Tt,
        normalize=cfg["signal"].get("normalize_dft", True),
        norm=cfg["signal"].get("normalization_target", 3.99)
    )

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    ta, tb = phase_core.calc_ta_tb(an, bn)

    return np.real(ta), np.real(tb), phase_core


def _reconstruct_from_ta_tb(
    ta: np.ndarray,
    tb: np.ndarray,
    t: np.ndarray,
    params
) -> np.ndarray:
    """
    Reconstruct signal tensor from ta/tb.
    """

    ta = np.asarray(ta)
    tb = np.asarray(tb)

    if ta.shape != tb.shape:
        raise ValueError("ta and tb must have the same shape.")

    if ta.ndim != 3:
        raise ValueError("ta and tb must have shape (periods, N, sensors).")

    n_periods, _, S = ta.shape
    x_hat = np.zeros((len(t), n_periods, S), dtype=float)

    w0 = 2.0 * np.pi / params.tau

    for p in range(n_periods):
        for s in range(S):
            x_hat[:, p, s] = recover_signal(
                ta[p, :, s],
                tb[p, :, s],
                t,
                w0
            )

    return x_hat


def _quantize_ta_tb_tensor(
    ta: np.ndarray,
    tb: np.ndarray,
    w0: float,
    M: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize ta/tb tensors period-by-period and sensor-by-sensor.
    """

    ta = np.asarray(ta, dtype=float)
    tb = np.asarray(tb, dtype=float)

    if ta.shape != tb.shape:
        raise ValueError("ta and tb must have the same shape.")

    if ta.ndim != 3:
        raise ValueError("ta and tb must have shape (periods, N, sensors).")

    n_periods, _, S = ta.shape

    ta_q = np.zeros_like(ta, dtype=float)
    tb_q = np.zeros_like(tb, dtype=float)

    for p in range(n_periods):
        for s in range(S):
            ta_q[p, :, s], tb_q[p, :, s] = quantize_ta_tb(
                ta[p, :, s],
                tb[p, :, s],
                w0,
                M
            )

    return ta_q, tb_q


# =============================================================================
# BENCHMARK + FDMA
# =============================================================================

def _run_benchmark_fdma_branch(
    cfg: Dict[str, Any],
    params
) -> Tuple[float, np.ndarray]:
    """
    Run analytical Benchmark + FDMA branch.

    The Benchmark branch computes feasible per-sensor quantization bins M_s
    from the per-sensor physical budget:

        B_s = alpha_s * B
        SNR_s = P / (B_s * N0)

    and returns the average absolute quantization MSE:

        MSE_s = peak_to_peak^2 / (12 M_s^2)
    """

    if not cfg.get("mode", {}).get("run_benchmark_fdma", True):
        return np.nan, np.array([], dtype=object)

    benchmark_cfg = cfg.get("benchmark", {})
    sampling_rate = float(benchmark_cfg.get("sampling_rate", params.W))
    effective_rate_factor = float(benchmark_cfg.get("effective_rate_factor", 1.0))
    effective_sampling_rate = effective_rate_factor * sampling_rate

    peak_to_peak = float(cfg["signal"].get("peak_to_peak", 1.0))

    M_vec = compute_benchmark_M_per_sensor(
        S=params.S,
        tau=params.tau,
        B=params.B,
        P=params.P,
        N0=params.N0,
        sampling_rate=effective_sampling_rate,
        bandwidth_allocation=params.bandwidth_allocation,
        force_power_of_two=params.quantization_force_power_of_two,
        rounding_mode=params.quantization_rounding_mode,
    )

    print(f"[INFO] Benchmark sampling_rate = {sampling_rate:.6e}")
    print(f"[INFO] Benchmark effective_rate_factor = {effective_rate_factor:.6e}")
    print(f"[INFO] Benchmark effective_sampling_rate = {effective_sampling_rate:.6e}")
    print(f"[INFO] Benchmark M_per_sensor = {M_vec}")

    mse_sum = 0.0
    count = 0

    for M in M_vec:
        M_int = int(M)

        if M_int < 2:
            continue

        try:
            mse_sum += (peak_to_peak ** 2) / (12.0 * (float(M_int) ** 2))
        except OverflowError:
            mse_sum += 0.0

        count += 1

    if count == 0:
        return np.nan, np.asarray(M_vec, dtype=object)

    mse = mse_sum / count

    print(f"[INFO] Benchmark M_min = {_safe_min_int(M_vec)}")
    print(f"[INFO] Benchmark M_mean = {_safe_mean_float(M_vec)}")
    print(f"[INFO] Benchmark M_max = {_safe_max_int(M_vec)}")
    print(f"[INFO] Benchmark analytical MSE = {mse:.8e}")

    return mse, np.asarray(M_vec, dtype=object)


# =============================================================================
# CS + FDMA
# =============================================================================

def _run_cs_fdma_branch(
    x_ref: np.ndarray,
    t: np.ndarray,
    cfg: Dict[str, Any],
    params,
    rng: np.random.Generator
) -> Tuple[float, List[int]]:
    """
    Run CS + FDMA branch.
    """

    cs_cfg = cfg.get("cs", {})

    bits_per_measurement = float(cs_cfg.get("bits_per_measurement", 8.0))
    sparsity_cfg = int(cs_cfg.get("sparsity", 5))
    basis = cs_cfg.get("basis", "dct")
    sensing_matrix = cs_cfg.get("sensing_matrix", "gaussian")
    normalize_dictionary_columns = cs_cfg.get("normalize_dictionary_columns", True)
    store_true_representation = cs_cfg.get("store_true_representation", False)

    n_time, _, S = x_ref.shape

    x_hat = np.zeros_like(x_ref)
    measurements_per_sensor = []

    for s in range(S):
        B_sensor = float(params.B_per_sensor[s])
        snr_sensor = float(params.P / (B_sensor * params.N0))
        capacity_sensor = B_sensor * np.log2(1.0 + snr_sensor)
        bits_per_cycle = capacity_sensor * params.tau

        n_measurements = int(np.floor(bits_per_cycle / bits_per_measurement))
        n_measurements = max(1, min(n_measurements, n_time))

        sparsity_eff = min(sparsity_cfg, n_measurements)
        measurements_per_sensor.append(n_measurements)

        cs_random_state = cs_cfg.get("random_state", None)
        if cs_random_state is None:
            cs_random_state = int(rng.integers(0, 2**32 - 1))

        cs_core = CSAcquisitionCore(
            n_measurements=n_measurements,
            sparsity=sparsity_eff,
            basis=basis,
            sensing_matrix=sensing_matrix,
            random_state=cs_random_state,
            normalize_dictionary_columns=normalize_dictionary_columns,
            store_true_representation=store_true_representation,
        )

        x_sensor = x_ref[:, :, s:s + 1]

        acq = cs_core.acquire(
            x=x_sensor,
            t=t
        )

        rec = cs_core.reconstruct(acq)

        x_hat[:, :, s:s + 1] = rec.reconstructed_signal

    mse = float(np.mean((x_ref - x_hat) ** 2))

    return mse, measurements_per_sensor


# =============================================================================
# PPM + FDMA
# =============================================================================

def _run_ppm_fdma_branch(
    x_ref: np.ndarray,
    t: np.ndarray,
    cfg: Dict[str, Any],
    params,
    rng: np.random.Generator
) -> Tuple[float, List[float]]:
    """
    Run PPM + FDMA branch.
    """

    ppm_cfg = cfg.get("ppm", {})
    _, _, S = x_ref.shape
    Tt = float(cfg["signal"]["Tt"])

    x_hat = np.zeros_like(x_ref)
    fs_msg_list = []

    for s in range(S):
        B_sensor = float(params.B_per_sensor[s])
        snr_sensor = float(params.P / (B_sensor * params.N0))

        timing = _resolve_ppm_timing_from_B_sensor(
            ppm_cfg=ppm_cfg,
            params=params,
            B_sensor=B_sensor,
            Tt=Tt
        )

        fs_msg = timing["fs_msg"]
        pulse_width = timing["pulse_width"]

        fs_msg_list.append(fs_msg)

        ppm_core = PPMCore(
            fc=fs_msg,
            pulse_width=pulse_width,
            rec_pulse=ppm_cfg.get("rec_pulse", 0.0),
            pulse_type=ppm_cfg.get("pulse_type", "raised_cosine"),
            rolloff=ppm_cfg.get("rolloff", 0.99),
            span=ppm_cfg.get("span", 12),
            eps_margin=ppm_cfg.get("eps_margin", 1e-3),
            interp_mode=ppm_cfg.get("interp_mode", "linear"),
            periodic_replicas=ppm_cfg.get("periodic_replicas", 10),
            clip_recovered_to_unit_interval=ppm_cfg.get(
                "clip_recovered_to_unit_interval",
                True
            ),
        )

        x_sensor = x_ref[:, :, s:s + 1]

        mod_result = ppm_core.modulate(
            x=x_sensor,
            t=t
        )

        tx = np.array(mod_result.tx_waveform, dtype=float, copy=True)

        if ppm_cfg.get("normalize_sensor_power", True):
            tx = _normalize_tensor_power(
                tx,
                target_power=params.P
            )

        rx = _add_awgn_from_snr(
            x=tx,
            snr_linear=snr_sensor,
            rng=rng
        )

        demod_result = ppm_core.demodulate(
            y=rx,
            t=t,
            modulation_result=mod_result,
            reconstruct_continuous=True
        )

        if demod_result.recovered_continuous is None:
            raise RuntimeError("PPM demodulation did not return recovered_continuous.")

        x_hat[:, :, s:s + 1] = demod_result.recovered_continuous

    mse = float(np.mean((x_ref - x_hat) ** 2))

    return mse, fs_msg_list


def _resolve_ppm_timing_from_B_sensor(
    ppm_cfg: Dict[str, Any],
    params,
    B_sensor: float,
    Tt: Optional[float] = None
) -> Dict[str, float]:
    """
    Resolve PPM timing from the per-sensor FDMA bandwidth and numerical time grid.
    """

    if B_sensor <= 0:
        raise ValueError("B_sensor must be positive.")

    pulse_type = str(ppm_cfg.get("pulse_type", "raised_cosine")).lower()
    rolloff = float(ppm_cfg.get("rolloff", 0.99))

    requested_fs_msg = ppm_cfg.get("fs_msg", params.W)

    if isinstance(requested_fs_msg, str) and requested_fs_msg.lower() == "auto":
        requested_fs_msg = params.W

    requested_fs_msg = float(requested_fs_msg)

    if requested_fs_msg <= 0:
        raise ValueError("ppm.fs_msg must be positive.")

    enforce_bandwidth = ppm_cfg.get("enforce_bandwidth_from_B_sensor", True)

    if pulse_type in {"raised_cosine", "root_raised_cosine", "rrc", "rc"}:
        max_fs_msg_from_B_sensor = 2.0 * B_sensor / (1.0 + rolloff)
    else:
        max_fs_msg_from_B_sensor = B_sensor

    min_samples_per_symbol = int(ppm_cfg.get("min_samples_per_symbol", 8))

    if min_samples_per_symbol < 2:
        raise ValueError("ppm.min_samples_per_symbol must be >= 2.")

    if Tt is not None:
        Tt = float(Tt)

        if Tt <= 0:
            raise ValueError("Tt must be positive.")

        max_fs_msg_from_time_grid = 1.0 / (min_samples_per_symbol * Tt)
    else:
        max_fs_msg_from_time_grid = np.inf

    if enforce_bandwidth:
        fs_msg = min(
            requested_fs_msg,
            max_fs_msg_from_B_sensor,
            max_fs_msg_from_time_grid
        )
    else:
        fs_msg = min(
            requested_fs_msg,
            max_fs_msg_from_time_grid
        )

    if fs_msg <= 0:
        raise ValueError("Resolved PPM fs_msg must be positive.")

    if "pulse_width" in ppm_cfg and ppm_cfg["pulse_width"] is not None:
        pulse_width = float(ppm_cfg["pulse_width"])
    else:
        pulse_width_fraction = float(ppm_cfg.get("pulse_width_fraction", 0.1))
        pulse_width = pulse_width_fraction / fs_msg

    Tc = 1.0 / fs_msg

    if pulse_width >= Tc:
        pulse_width = 0.9 * Tc

    return {
        "requested_fs_msg": requested_fs_msg,
        "max_fs_msg_from_B_sensor": max_fs_msg_from_B_sensor,
        "max_fs_msg_from_time_grid": max_fs_msg_from_time_grid,
        "fs_msg": fs_msg,
        "was_fs_msg_clipped": bool(fs_msg < requested_fs_msg),
        "pulse_width": pulse_width,
        "min_samples_per_symbol": min_samples_per_symbol,
    }


# =============================================================================
# RbCP / RbCP_time
# =============================================================================

def _run_rbcp_branch(
    x_ref: np.ndarray,
    t: np.ndarray,
    cfg: Dict[str, Any],
    params,
    N: int,
    M
) -> float:
    """
    Run RbCP-style direct phase quantization/reconstruction.
    """

    if M is None:
        return np.nan

    M = int(M)

    if M < 2:
        return np.nan

    Tt = cfg["signal"]["Tt"]

    ta, tb, _ = _compute_phase_coefficients_from_reference(
        x_ref=x_ref,
        Tt=Tt,
        cfg=cfg,
        params=params,
        N=N
    )

    w0 = 2.0 * np.pi / params.tau

    ta_q, tb_q = _quantize_ta_tb_tensor(
        ta=ta,
        tb=tb,
        w0=w0,
        M=M
    )

    x_hat = _reconstruct_from_ta_tb(
        ta=ta_q,
        tb=tb_q,
        t=t,
        params=params
    )

    return float(np.mean((x_ref - x_hat) ** 2))


# =============================================================================
# SFC / SFC + SED
# =============================================================================

def _run_sfc_family_branch(
    x_ref: np.ndarray,
    t: np.ndarray,
    cfg: Dict[str, Any],
    params,
    N: int,
    sfc_channel: Optional[SFCChannel],
    run_native_sfc: bool,
    run_sfc_sed: bool
) -> Tuple[float, float, float]:
    """
    Run native SFC and SFC+SED using the same detected event matrix.
    """

    if sfc_channel is None:
        return np.nan, np.nan, np.nan

    Tt = cfg["signal"]["Tt"]
    n_periods = x_ref.shape[1]

    ta, tb, phase_core = _compute_phase_coefficients_from_reference(
        x_ref=x_ref,
        Tt=Tt,
        cfg=cfg,
        params=params,
        N=N
    )

    events = phase_core.ta_tb_to_events(ta, tb)

    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    mse_sfc = np.nan
    mse_sfc_sed = np.nan
    valid_fraction = np.nan

    if run_native_sfc:
        mse_sfc = _mse_from_sfc_events(
            events_for_rec=events_est,
            x_ref=x_ref,
            t=t,
            params=params,
            phase_core=phase_core,
            period_valid_mask=None
        )

    if run_sfc_sed:
        period_slots = events_est.shape[0] // n_periods

        sed_result = detect_semantic_errors(
            events_est=events_est,
            period_slots=period_slots,
            N=N,
            sensor_x_event=_build_sensor_x_event(params.S, N),
            discard_invalid_periods=cfg.get("sed", {}).get("discard_invalid_periods", True)
        )

        corrected_events_est, period_valid_mask = _extract_sed_outputs(sed_result)
        valid_fraction = float(np.mean(period_valid_mask))

        if np.any(period_valid_mask):
            mse_sfc_sed = _mse_from_sfc_events(
                events_for_rec=corrected_events_est,
                x_ref=x_ref,
                t=t,
                params=params,
                phase_core=phase_core,
                period_valid_mask=period_valid_mask
            )
        else:
            mse_sfc_sed = np.nan

    return mse_sfc, mse_sfc_sed, valid_fraction


def _mse_from_sfc_events(
    events_for_rec: np.ndarray,
    x_ref: np.ndarray,
    t: np.ndarray,
    params,
    phase_core: PhaseCoefficientCore,
    period_valid_mask: Optional[np.ndarray] = None
) -> float:
    """
    Convert event estimates to ta/tb, reconstruct, and compute MSE.
    """

    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_for_rec)

    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    x_hat = _reconstruct_from_ta_tb(
        ta=ta_rec,
        tb=tb_rec,
        t=t,
        params=params
    )

    if period_valid_mask is None:
        return float(np.mean((x_ref - x_hat) ** 2))

    valid_periods = np.asarray(period_valid_mask, dtype=bool)

    if not np.any(valid_periods):
        return np.nan

    return float(
        np.nanmean(
            (x_ref[:, valid_periods, :] - x_hat[:, valid_periods, :]) ** 2
        )
    )


def _build_sfc_channel_for_B(
    cfg_B: Dict[str, Any],
    N: int,
    S: int
) -> SFCChannel:
    """
    Build SFCChannel for the current B point.
    """

    cfg_sfc = copy.deepcopy(cfg_B)

    if "channel" not in cfg_sfc:
        cfg_sfc["channel"] = {}

    cfg_sfc["channel"]["sensor_x_event"] = _build_sensor_x_event(S, N)
    cfg_sfc["channel"]["collision_mode"] = cfg_sfc["channel"].get("collision_mode", "sum")
    cfg_sfc["channel"]["type"] = cfg_sfc["channel"].get("type", "awgn")
    cfg_sfc["channel"]["detection_mode"] = cfg_sfc["channel"].get("detection_mode", "threshold")
    cfg_sfc["channel"]["score_threshold"] = cfg_sfc["channel"].get(
        "score_threshold",
        cfg_sfc["system"]["L"]
    )

    if "threshold" not in cfg_sfc["channel"]:
        cfg_sfc["channel"]["threshold_factor"] = cfg_sfc["channel"].get(
            "threshold_factor",
            0.5
        )

    if "reproducibility" not in cfg_sfc:
        cfg_sfc["reproducibility"] = {}

    if "seed" not in cfg_sfc["reproducibility"]:
        cfg_sfc["reproducibility"]["seed"] = cfg_B.get(
            "reproducibility", {}
        ).get(
            "seed",
            cfg_B.get("monte_carlo", {}).get("seed", 12345)
        )

    return SFCChannel(cfg_sfc)


def _build_sensor_x_event(S: int, N: int) -> np.ndarray:
    """
    Build sensor-event association matrix.
    """

    num_event_ids = 2 * N * S
    sensor_x_event = np.zeros((S, num_event_ids), dtype=float)

    for s in range(S):
        start = 2 * s * N
        stop = 2 * (s + 1) * N
        sensor_x_event[s, start:stop] = 1.0

    return sensor_x_event


def _extract_sed_outputs(sed_result):
    """
    Extract corrected_events_est and period_valid_mask from SED output.
    """

    if isinstance(sed_result, dict):
        corrected_events_est = sed_result["corrected_events_est"]
        period_valid_mask = np.asarray(sed_result["period_valid_mask"], dtype=bool)
        return corrected_events_est, period_valid_mask

    corrected_events_est = getattr(sed_result, "corrected_events_est")
    period_valid_mask = np.asarray(getattr(sed_result, "period_valid_mask"), dtype=bool)

    return corrected_events_est, period_valid_mask


# =============================================================================
# NUMERICAL HELPERS
# =============================================================================

def _normalize_tensor_power(
    x: np.ndarray,
    target_power: float
) -> np.ndarray:
    """
    Normalize average power of a single-sensor tensor to target_power.
    """

    if target_power <= 0:
        raise ValueError("target_power must be positive.")

    x = np.asarray(x, dtype=float)
    p = np.mean(x ** 2)

    if np.isclose(p, 0.0):
        return np.array(x, copy=True)

    return x * np.sqrt(target_power / p)


def _add_awgn_from_snr(
    x: np.ndarray,
    snr_linear: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add AWGN using target SNR in linear scale.
    """

    if snr_linear <= 0:
        raise ValueError("snr_linear must be positive.")

    x = np.asarray(x, dtype=float)
    p_signal = np.mean(x ** 2)

    if np.isclose(p_signal, 0.0):
        return np.array(x, copy=True)

    p_noise = p_signal / snr_linear
    noise = rng.normal(0.0, np.sqrt(p_noise), size=x.shape)

    return x + noise


def _print_metric(name: str, value: float):
    """
    Print metric with NaN-safe formatting.
    """

    if np.isfinite(value):
        print(f"[INFO] {name} = {value:.8e}")
    else:
        print(f"[INFO] {name} = nan")


def _safe_min_int(values) -> int:
    """
    Return minimum of integer-like values as Python int.
    """

    values = list(values)

    if len(values) == 0:
        raise ValueError("Cannot compute minimum of an empty list.")

    return int(min(int(v) for v in values))


def _safe_max_int(values) -> int:
    """
    Return maximum of integer-like values as Python int.
    """

    values = list(values)

    if len(values) == 0:
        raise ValueError("Cannot compute maximum of an empty list.")

    return int(max(int(v) for v in values))


def _safe_mean_float(values) -> float:
    """
    Return mean of integer-like values as float.
    """

    values = [int(v) for v in list(values)]

    if len(values) == 0:
        raise ValueError("Cannot compute mean of an empty list.")

    return float(sum(values) / len(values))


def _fits_int64(value) -> bool:
    """
    Return True if value fits signed int64.
    """

    try:
        v = int(value)
    except Exception:
        return False

    return -(2**63) <= v <= 2**63 - 1


def _safe_table_value(value):
    """
    Return value as int if it fits int64, otherwise as string.

    This avoids pandas/numpy integer overflow for very large M_RbCP values.
    """

    if isinstance(value, float) and np.isnan(value):
        return np.nan

    if _fits_int64(value):
        return int(value)

    return str(value)


# =============================================================================
# SAVE UTILITY
# =============================================================================

def save_dat_file(
    df: pd.DataFrame,
    path: str,
    delimiter: str = "\t"
):
    """
    Save the dataset to a .dat-compatible tabular file.
    """

    df.to_csv(
        path,
        sep=delimiter,
        index=False,
        float_format="%.8e"
    )


__all__ = [
    "generate_fair_methods_comparison_vs_B_data",
    "save_dat_file",
]
