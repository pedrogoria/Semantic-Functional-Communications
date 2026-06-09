"""
sfc/pipelines/rbcp_mse_vs_B.py

Pipeline for reproducing manuscript Figure 5: MSE versus total bandwidth B.

Compared methods
----------------
- Benchmark Approach
- RbCP
- RbCP_time
- SFC

Physical-budget convention
--------------------------
This pipeline follows a Figure-5-style fixed per-sensor SNR model.

For each B point:
    - N0 is kept fixed;
    - target per-sensor SNR_dB is kept fixed;
    - P(B) is derived from:

        SNR_s = P / (B_s * N0)

      therefore:

        P(B) = SNR_s * B_s * N0.

With a scalar P shared by all sensors, fixed per-sensor SNR requires uniform
bandwidth allocation. For nonuniform allocation, use fixed P,N0 or allow
sensor-dependent powers P_s.

Noise convention
----------------
This pipeline itself does not add waveform-domain AWGN directly.

When the SFC physical channel is used, the channel must follow the project-wide
physical-channel convention:

    simulated additive noise is generated from N0 directly at the
    matched-filter/resource-output level.

Therefore, the simulated SFC noise variance must not be B_s * N0.

Capacity / quantization convention
----------------------------------
SNR and Shannon capacity are still used to determine feasible quantization
resolutions:

    SNR_s = P / (B_s * N0)
    C_s   = B_s * log2(1 + SNR_s)

For RbCP and Benchmark, the feasible number of bins is computed from the same
per-sensor channel budget using core theory/system-parameter helpers.

Quantization policy
-------------------
This pipeline relies on the defaults implemented in the core:
- M and M_RbCP are free integers by default;
- no power-of-two restriction unless explicitly requested in cfg.

Source-bandwidth convention
---------------------------
The source signal is filtered using the configured source bandwidth:

    signal.W

The pipeline must NOT redefine the filtering bandwidth from N using:

    W_eff = 2 * N / tau

The role of N is to define the number of representation harmonics.
The role of W is to define the source-signal bandwidth used by the signal filter.

Core-use policy
---------------
- Filtering uses sfc.core.filters.filter_periodic.
- Fourier coefficients use sfc.core.fourier.FourierCoefficientCore.
- Phase coefficients use sfc.core.phase_cof.PhaseCoefficientCore.
- ta/tb quantization uses sfc.core.quantization.quantize_ta_tb.
- Harmonic reconstruction uses sfc.core.reconstruction.recover_signal.
- Benchmark M uses sfc.core.system_parameters.compute_benchmark_M_per_sensor.
- Fixed-SNR power derivation uses sfc.core.theory.compute_sensor_bandwidths.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.system_parameters import (
    build_derived_system_parameters,
    compute_benchmark_M_per_sensor,
)
from sfc.core.theory import (
    compute_sensor_bandwidths,
    compute_snr_linear,
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_rbcp_mse_vs_B_data(cfg):
    """
    Generate the Figure 5 dataset.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - B
        - P_derived
        - SNR_total_dB
        - SNR_sensor_dB
        - M_benchmark_min
        - M_benchmark_max
        - M_rbcp
        - M_time
        - mse_benchmark
        - mse_rbcp
        - mse_rbcp_time
        - mse_sfc
        - num_trials
    """
    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    print("[INFO] Starting rbcp_mse_vs_B data generation")
    print(f"[INFO] Monte Carlo seed = {cfg['monte_carlo']['seed']}")
    print(
        f"[INFO] Sweep B from {cfg['sweep']['B']['start']} "
        f"to {cfg['sweep']['B']['stop']} "
        f"step {cfg['sweep']['B']['step']}"
    )
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

        # ---------------------------------------------------------------------
        # Figure-5-style power model:
        # keep per-sensor SNR fixed and N0 fixed, derive P(B).
        # ---------------------------------------------------------------------
        cfg_B["system"]["B"] = float(B)
        cfg_B["system"]["P"] = _derive_power_from_fixed_sensor_snr_and_n0(cfg_B)

        params = build_derived_system_parameters(cfg_B)

        N = int(cfg_B["signal"].get("N_override", params.N))
        n_trials = int(cfg_B["monte_carlo"]["interactions"])

        benchmark_cfg = cfg_B.get("benchmark", {})
        sampling_rate = float(benchmark_cfg.get("sampling_rate", params.W))
        effective_rate_factor = float(benchmark_cfg.get("effective_rate_factor", 2.0))
        effective_sampling_rate = effective_rate_factor * sampling_rate

        M_benchmark_vec = compute_benchmark_M_per_sensor(
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

        M_benchmark_min = _safe_min_int(M_benchmark_vec)
        M_benchmark_max = _safe_max_int(M_benchmark_vec)

        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] B = {B:.1f} Hz")
        print(f"[INFO] P(B) = {cfg_B['system']['P']:.6e}")
        print(
            f"[INFO] S = {params.S} | R = {params.R} | L = {params.L} | "
            f"tau = {params.tau:.3f} s | W = {params.W:.3f} Hz | N = {N}"
        )
        print(f"[INFO] N0 = {cfg_B['system']['N0']:.6e}")
        print(f"[INFO] bandwidth_allocation = {params.bandwidth_allocation}")
        print(f"[INFO] B_per_sensor = {params.B_per_sensor}")
        print(
            f"[INFO] SNR_total_dB = {params.SNR_dB:.3f} | "
            f"SNR_total = {params.SNR:.6e}"
        )
        print(f"[INFO] SNR_per_sensor_dB = {params.SNR_per_sensor_dB}")
        print(f"[INFO] SNR_per_sensor = {params.SNR_per_sensor}")
        print(f"[INFO] effective_sampling_rate = {effective_sampling_rate:.6e}")
        print(f"[INFO] M_benchmark_vec = {M_benchmark_vec}")
        print(f"[INFO] M_benchmark_min = {M_benchmark_min}")
        print(f"[INFO] M_benchmark_max = {M_benchmark_max}")
        print(f"[INFO] M_RbCP = {params.M_rbcp}")
        print(f"[INFO] M_RbCP per sensor = {params.M_rbcp_per_sensor}")
        print(f"[INFO] M_time = {params.M_time}")
        print(
            f"[INFO] quantization_force_power_of_two = "
            f"{params.quantization_force_power_of_two}"
        )
        print(f"[INFO] quantization_rounding_mode = {params.quantization_rounding_mode}")

        # ---------------------------------------------------------------------
        # Build one SFC channel for this B-point and reuse in all trials.
        #
        # Important:
        # SFCChannel must internally use the project-wide PhysicalChannel
        # convention: noise generated from N0 directly, not B_s*N0.
        # ---------------------------------------------------------------------
        sfc_channel = None
        if cfg_B.get("mode", {}).get("run_sfc", True):
            sfc_channel = _build_sfc_channel_for_B(cfg_B, N, params.S)

        # ---------------------------------------------------------------------
        # Monte Carlo accumulation.
        # ---------------------------------------------------------------------
        mse_benchmark_sum = 0.0
        mse_rbcp_sum = 0.0
        mse_rbcp_time_sum = 0.0
        mse_sfc_sum = 0.0

        benchmark_count = 0
        rbcp_count = 0
        rbcp_time_count = 0
        sfc_count = 0

        for i in range(n_trials):
            trial = _run_one_trial(
                cfg=cfg_B,
                rng=rng,
                N=N,
                sfc_channel=sfc_channel,
            )

            if np.isfinite(trial["mse_benchmark"]):
                mse_benchmark_sum += trial["mse_benchmark"]
                benchmark_count += 1

            if np.isfinite(trial["mse_rbcp"]):
                mse_rbcp_sum += trial["mse_rbcp"]
                rbcp_count += 1

            if np.isfinite(trial["mse_rbcp_time"]):
                mse_rbcp_time_sum += trial["mse_rbcp_time"]
                rbcp_time_count += 1

            if np.isfinite(trial["mse_sfc"]):
                mse_sfc_sum += trial["mse_sfc"]
                sfc_count += 1

            if (i + 1) % max(1, n_trials // 5) == 0:
                print(f"[INFO] Trial progress: {i + 1}/{n_trials}")

        mse_benchmark = (
            mse_benchmark_sum / benchmark_count
            if benchmark_count > 0
            else np.nan
        )
        mse_rbcp = (
            mse_rbcp_sum / rbcp_count
            if rbcp_count > 0
            else np.nan
        )
        mse_rbcp_time = (
            mse_rbcp_time_sum / rbcp_time_count
            if rbcp_time_count > 0
            else np.nan
        )
        mse_sfc = (
            mse_sfc_sum / sfc_count
            if sfc_count > 0
            else np.nan
        )

        print(
            f"[INFO] mse_benchmark = {mse_benchmark:.6e}"
            if np.isfinite(mse_benchmark)
            else "[INFO] mse_benchmark = nan"
        )
        print(
            f"[INFO] mse_rbcp = {mse_rbcp:.6e}"
            if np.isfinite(mse_rbcp)
            else "[INFO] mse_rbcp = nan"
        )
        print(
            f"[INFO] mse_rbcp_time = {mse_rbcp_time:.6e}"
            if np.isfinite(mse_rbcp_time)
            else "[INFO] mse_rbcp_time = nan"
        )
        print(
            f"[INFO] mse_sfc = {mse_sfc:.6e}"
            if np.isfinite(mse_sfc)
            else "[INFO] mse_sfc = nan"
        )

        results.append({
            "B": float(B),
            "P_derived": float(cfg_B["system"]["P"]),
            "SNR_total_dB": float(params.SNR_dB),
            "SNR_sensor_dB": float(params.SNR_per_sensor_dB[0]),
            "SNR_sensor_dB_min": float(np.min(params.SNR_per_sensor_dB)),
            "SNR_sensor_dB_max": float(np.max(params.SNR_per_sensor_dB)),
            "M_benchmark_min": _safe_table_value(M_benchmark_min),
            "M_benchmark_max": _safe_table_value(M_benchmark_max),
            "M_rbcp": _safe_table_value(params.M_rbcp),
            "M_time": int(params.M_time),
            "mse_benchmark": mse_benchmark,
            "mse_rbcp": mse_rbcp,
            "mse_rbcp_time": mse_rbcp_time,
            "mse_sfc": mse_sfc,
            "num_trials": n_trials,
        })

    return pd.DataFrame(results)


# =============================================================================
# FIGURE-5-STYLE POWER MODEL
# =============================================================================

def _derive_power_from_fixed_sensor_snr_and_n0(cfg):
    """
    Derive scalar P(B) from fixed per-sensor SNR and fixed N0.

    The configured SNR_dB is interpreted as the target per-sensor SNR:

        SNR_s = P / (B_s * N0)

    Therefore:

        P(B) = SNR_s * B_s * N0

    Because this pipeline uses one scalar P shared by all sensors, fixed
    per-sensor SNR is only compatible with uniform bandwidth allocation.

    For nonuniform allocation, use fixed P,N0 or sensor-dependent powers P_s.
    """
    SNR_s = compute_snr_linear(cfg["system"]["SNR_dB"])
    B = float(cfg["system"]["B"])
    N0 = float(cfg["system"]["N0"])
    S = int(cfg["system"]["S"])

    allocation, B_per_sensor = compute_sensor_bandwidths(
        B=B,
        S=S,
        bandwidth_allocation=cfg["system"].get("bandwidth_allocation", None),
    )

    if not np.allclose(allocation, np.ones(S) / S):
        raise ValueError(
            "fixed per-sensor SNR with scalar P requires uniform bandwidth allocation. "
            "For nonuniform allocation, either allow sensor-dependent powers P_s "
            "or use a fixed P,N0 power model."
        )

    return float(SNR_s * float(B_per_sensor[0]) * N0)


# =============================================================================
# SFC CHANNEL BUILDER
# =============================================================================

def _build_sfc_channel_for_B(cfg_B, N, S):
    """
    Build one SFCChannel instance to be reused for all trials of the current B.
    """
    cfg_sfc = copy.deepcopy(cfg_B)

    if "channel" not in cfg_sfc:
        cfg_sfc["channel"] = {}

    cfg_sfc["channel"]["sensor_x_event"] = _build_sensor_x_event(S, N)
    cfg_sfc["channel"]["collision_mode"] = cfg_sfc["channel"].get(
        "collision_mode",
        "sum",
    )
    cfg_sfc["channel"]["type"] = cfg_sfc["channel"].get("type", "awgn")
    cfg_sfc["channel"]["detection_mode"] = cfg_sfc["channel"].get(
        "detection_mode",
        "threshold",
    )
    cfg_sfc["channel"]["score_threshold"] = cfg_sfc["channel"].get(
        "score_threshold",
        cfg_sfc["system"]["L"],
    )

    if "threshold" not in cfg_sfc["channel"]:
        cfg_sfc["channel"]["threshold_factor"] = cfg_sfc["channel"].get(
            "threshold_factor",
            0.5,
        )

    if "reproducibility" not in cfg_sfc:
        cfg_sfc["reproducibility"] = {}

    if "seed" not in cfg_sfc["reproducibility"]:
        cfg_sfc["reproducibility"]["seed"] = cfg_B.get(
            "reproducibility",
            {},
        ).get(
            "seed",
            cfg_B.get("monte_carlo", {}).get("seed", 12345),
        )

    return SFCChannel(cfg_sfc)


def _build_sensor_x_event(S, N):
    """
    Build the sensor-event association matrix.

    Event ordering:
    for each sensor s:
        [ta events for N harmonics][tb events for N harmonics]

    Total number of event IDs:
        2 * N * S
    """
    num_event_ids = 2 * N * S
    sensor_x_event = np.zeros((S, num_event_ids), dtype=float)

    for s in range(S):
        start = 2 * s * N
        stop = 2 * (s + 1) * N
        sensor_x_event[s, start:stop] = 1.0

    return sensor_x_event


# =============================================================================
# ONE MONTE CARLO TRIAL
# =============================================================================

def _run_one_trial(cfg, rng, N, sfc_channel=None):
    """
    Run one Monte Carlo trial and return the four MSE values.

    Returns
    -------
    dict
        {
            "mse_benchmark": ...,
            "mse_rbcp": ...,
            "mse_rbcp_time": ...,
            "mse_sfc": ...
        }
    """
    params = build_derived_system_parameters(cfg)

    n_periods = int(
        cfg.get("signal", {}).get(
            "n_periods",
            cfg.get("simulation", {}).get("n_periods", 1),
        )
    )

    tau = params.tau
    Tt = cfg["signal"]["Tt"]

    t = np.arange(0, tau, Tt)
    n_time = len(t)

    # -------------------------------------------------------------------------
    # 1. Generate S random signals.
    # -------------------------------------------------------------------------
    x_raw = _generate_signals(
        cfg=cfg,
        rng=rng,
        num_time_samples=n_time,
        n_periods=n_periods,
        S=params.S,
    )

    # -------------------------------------------------------------------------
    # 2. Band-limit using configured signal.W.
    # -------------------------------------------------------------------------
    x_filtered = _filter_signals(
        x_raw=x_raw,
        cfg=cfg,
        params=params,
        tau=tau,
        Tt=Tt,
    )

    # -------------------------------------------------------------------------
    # 3. Peak-to-peak control.
    # -------------------------------------------------------------------------
    x_filtered = _apply_peak_to_peak_control(
        x_filtered,
        cfg["signal"].get("peak_to_peak", 0.0),
    )

    # -------------------------------------------------------------------------
    # 4. DC handling.
    # -------------------------------------------------------------------------
    x_zero_mean = _apply_dc_handling(
        x_filtered=x_filtered,
        tau=tau,
        Tt=Tt,
        dc_enabled=cfg.get("dc", {}).get("enabled", False),
    )

    # -------------------------------------------------------------------------
    # 5. Fourier coefficients.
    # -------------------------------------------------------------------------
    an, bn, _ = _compute_fourier_coefficients(
        x_zero_mean=x_zero_mean,
        tau=tau,
        N=N,
        S=params.S,
        Tt=Tt,
        normalize_dft=cfg["signal"].get("normalize_dft", True),
        normalization_target=cfg["signal"].get("normalization_target", 3.99),
    )

    # -------------------------------------------------------------------------
    # 6. ta/tb.
    # -------------------------------------------------------------------------
    ta, tb = _compute_phase_coefficients(
        an=an,
        bn=bn,
        tau=tau,
        N=N,
        S=params.S,
        cfg=cfg,
        params=params,
        n_periods=n_periods,
    )

    # -------------------------------------------------------------------------
    # 7. Benchmark.
    # -------------------------------------------------------------------------
    mse_benchmark = _run_benchmark_branch(
        cfg=cfg,
        params=params,
    )

    # -------------------------------------------------------------------------
    # 8. RbCP.
    # -------------------------------------------------------------------------
    mse_rbcp = _run_rbcp_branch(
        ta=ta,
        tb=tb,
        x_ref=x_zero_mean,
        t=t,
        tau=tau,
        M_rbcp=params.M_rbcp,
    )

    # -------------------------------------------------------------------------
    # 9. RbCP_time: error-free time model, no physical channel.
    # -------------------------------------------------------------------------
    mse_rbcp_time = _run_rbcp_time_branch(
        ta=ta,
        tb=tb,
        cfg=cfg,
        params=params,
        N=N,
        x_ref=x_zero_mean,
        t=t,
        n_periods=n_periods,
    )

    # -------------------------------------------------------------------------
    # 10. SFC.
    # -------------------------------------------------------------------------
    mse_sfc = _run_sfc_branch(
        ta=ta,
        tb=tb,
        cfg=cfg,
        params=params,
        N=N,
        x_ref=x_zero_mean,
        t=t,
        n_periods=n_periods,
        sfc_channel=sfc_channel,
    )

    return {
        "mse_benchmark": mse_benchmark,
        "mse_rbcp": mse_rbcp,
        "mse_rbcp_time": mse_rbcp_time,
        "mse_sfc": mse_sfc,
    }


# =============================================================================
# SIGNAL GENERATION / PREPROCESSING
# =============================================================================

def _generate_signals(cfg, rng, num_time_samples, n_periods, S):
    """
    Generate S distinct signals across periods.

    Output shape:
        (time, periods, sensors)
    """
    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        return rng.uniform(-1, 1, size=(num_time_samples, n_periods, S))

    if dist == "gaussian":
        return rng.normal(0, 1, size=(num_time_samples, n_periods, S))

    raise ValueError("Invalid distribution. Use 'uniform' or 'gaussian'.")


def _filter_signals(x_raw, cfg, params, tau, Tt):
    """
    Band-limit each signal using configured signal.W.

    IMPORTANT
    ---------
    The source filter bandwidth must come from:

        signal.W

    It must NOT be redefined from N as:

        W_eff = 2 * N / tau

    N controls the number of representation harmonics. W controls the
    source-signal bandwidth.
    """
    W_filter = float(cfg["signal"].get("W", params.W))

    if W_filter <= 0:
        raise ValueError("signal.W must be positive.")

    x_filtered = np.zeros_like(x_raw)

    _, n_periods, S = x_raw.shape

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw[:, p, s],
                W_filter,
                Tt,
                tau,
            )

    return x_filtered


def _apply_peak_to_peak_control(x_filtered, peak_to_peak):
    """
    Apply peak-to-peak control independently per (period, sensor).
    """
    if peak_to_peak == 0:
        return x_filtered

    x_out = np.copy(x_filtered)
    _, n_periods, S = x_filtered.shape

    for p in range(n_periods):
        for s in range(S):
            current_p2p = np.max(x_filtered[:, p, s]) - np.min(x_filtered[:, p, s])
            if current_p2p != 0:
                x_out[:, p, s] = x_filtered[:, p, s] * (peak_to_peak / current_p2p)

    return x_out


def _apply_dc_handling(x_filtered, tau, Tt, dc_enabled):
    """
    Remove DC component unless dc_enabled is True.
    """
    x_zero_mean = np.zeros_like(x_filtered)
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
# FOURIER / PHASE
# =============================================================================

def _compute_fourier_coefficients(
    x_zero_mean,
    tau,
    N,
    S,
    Tt,
    normalize_dft,
    normalization_target,
):
    """
    Compute Fourier coefficients for all periods/sensors.
    """
    fourier_core = FourierCoefficientCore(
        T=tau,
        harmonics=N,
        sensor_nodes=S,
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=normalize_dft,
        norm=normalization_target,
    )

    return an, bn, x_used


def _compute_phase_coefficients(an, bn, tau, N, S, cfg, params, n_periods):
    """
    Compute ta/tb for all periods and sensors.
    """
    phase_core = PhaseCoefficientCore(
        T=tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=S,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001),
    )

    ta, tb = phase_core.calc_ta_tb(an, bn)

    ta = np.real(ta)
    tb = np.real(tb)

    return ta, tb


# =============================================================================
# BENCHMARK BRANCH
# =============================================================================

def _run_benchmark_branch(cfg, params):
    """
    Benchmark Approach for Figure 5.

    This branch is treated analytically, not through Monte Carlo reconstruction.

    The Benchmark uses the feasible number of bins M computed from the
    per-sensor channel budget:

        B_s = alpha_s * B
        SNR_s = P / (B_s * N0)

    and computes the absolute MSE:

        MSE = (peak_to_peak^2) / (12 * M^2)
    """
    if not cfg.get("mode", {}).get("run_benchmark", True):
        return np.nan

    benchmark_cfg = cfg.get("benchmark", {})
    sampling_rate = float(benchmark_cfg.get("sampling_rate", params.W))
    effective_rate_factor = float(benchmark_cfg.get("effective_rate_factor", 2.0))
    peak_to_peak = float(cfg["signal"]["peak_to_peak"])

    M_vec = compute_benchmark_M_per_sensor(
        S=params.S,
        tau=params.tau,
        B=params.B,
        P=params.P,
        N0=params.N0,
        sampling_rate=effective_rate_factor * sampling_rate,
        bandwidth_allocation=params.bandwidth_allocation,
        force_power_of_two=params.quantization_force_power_of_two,
        rounding_mode=params.quantization_rounding_mode,
    )

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
        return np.nan

    return mse_sum / count


# =============================================================================
# RbCP BRANCH
# =============================================================================

def _run_rbcp_branch(ta, tb, x_ref, t, tau, M_rbcp):
    """
    Run direct RbCP and return average MSE over sensors/periods.
    """
    M_rbcp = int(M_rbcp)

    if M_rbcp < 2:
        return np.nan

    w0 = 2 * np.pi / tau

    ta_q, tb_q = _quantize_ta_tb_tensor(
        ta=ta,
        tb=tb,
        w0=w0,
        M=M_rbcp,
    )

    _, n_periods, S = x_ref.shape
    mse_sum = 0.0
    count = 0

    for p in range(n_periods):
        for s in range(S):
            x_rec = recover_signal(
                ta_q[p, :, s],
                tb_q[p, :, s],
                t,
                w0,
            )

            mse_sum += np.mean((x_ref[:, p, s] - x_rec) ** 2)
            count += 1

    return mse_sum / count


def _quantize_ta_tb_tensor(ta, tb, w0, M):
    """
    Quantize ta/tb tensors period-by-period and sensor-by-sensor.

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

    n_periods, _, S = ta.shape

    ta_q = np.zeros_like(ta, dtype=float)
    tb_q = np.zeros_like(tb, dtype=float)

    for p in range(n_periods):
        for s in range(S):
            ta_q[p, :, s], tb_q[p, :, s] = quantize_ta_tb(
                ta[p, :, s],
                tb[p, :, s],
                w0,
                M,
            )

    return ta_q, tb_q


# =============================================================================
# RbCP_time BRANCH
# =============================================================================

def _run_rbcp_time_branch(ta, tb, cfg, params, N, x_ref, t, n_periods):
    """
    Error-free time-model branch:

        signal -> ta/tb -> events -> ta/tb -> signal

    No physical channel is used here.
    """
    if not cfg.get("mode", {}).get("run_rbcp_time", True):
        return np.nan

    w0 = 2 * np.pi / params.tau

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001),
    )

    events = phase_core.ta_tb_to_events(ta, tb)
    ta_rec, tb_rec = phase_core.event_to_ta_tb(events)

    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    _, _, S_dim = x_ref.shape
    mse_sum = 0.0
    count = 0

    for p in range(n_periods):
        for s in range(S_dim):
            x_rec = recover_signal(
                ta_rec[p, :, s],
                tb_rec[p, :, s],
                t,
                w0,
            )

            mse_sum += np.mean((x_ref[:, p, s] - x_rec) ** 2)
            count += 1

    return mse_sum / count


# =============================================================================
# SFC BRANCH
# =============================================================================

def _run_sfc_branch(ta, tb, cfg, params, N, x_ref, t, n_periods, sfc_channel):
    """
    Run SFC branch and return average MSE over sensors/periods.

    IMPORTANT
    ---------
    - uses non-quantized ta/tb;
    - does NOT use semantic error detection in Figure 5;
    - physical noise, if channel.type='awgn', must be handled inside SFCChannel
      using the core PhysicalChannel convention: N0 directly, not B_s*N0.
    """
    if not cfg.get("mode", {}).get("run_sfc", True):
        return np.nan

    if sfc_channel is None:
        return np.nan

    w0 = 2 * np.pi / params.tau

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001),
    )

    events = phase_core.ta_tb_to_events(ta, tb)

    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)

    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    _, _, S_dim = x_ref.shape
    mse_sum = 0.0
    count = 0

    for p in range(n_periods):
        for s in range(S_dim):
            x_rec = recover_signal(
                ta_rec[p, :, s],
                tb_rec[p, :, s],
                t,
                w0,
            )

            mse_sum += np.mean((x_ref[:, p, s] - x_rec) ** 2)
            count += 1

    return mse_sum / count


# =============================================================================
# SAFE VALUE HELPERS
# =============================================================================

def _safe_min_int(values):
    """
    Return the minimum of an array/list of integer-like values as Python int.
    """
    values = list(values)

    if len(values) == 0:
        raise ValueError("Cannot compute minimum of an empty list.")

    return int(min(int(v) for v in values))


def _safe_max_int(values):
    """
    Return the maximum of an array/list of integer-like values as Python int.
    """
    values = list(values)

    if len(values) == 0:
        raise ValueError("Cannot compute maximum of an empty list.")

    return int(max(int(v) for v in values))


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

    This avoids pandas/numpy integer overflow for very large M values.
    """
    if _fits_int64(value):
        return int(value)

    return str(value)


# =============================================================================
# SAVE UTILITY
# =============================================================================

def save_dat_file(df, path, delimiter="\t"):
    """
    Save the figure dataset to a .dat-compatible tabular file.
    """
    df.to_csv(
        path,
        sep=delimiter,
        index=False,
        float_format="%.8e",
    )


__all__ = [
    "generate_rbcp_mse_vs_B_data",
    "save_dat_file",
]
