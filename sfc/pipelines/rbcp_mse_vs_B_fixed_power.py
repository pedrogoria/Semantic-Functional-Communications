"""
sfc/pipelines/rbcp_mse_vs_B_fixed_power.py

Pipeline for reproducing manuscript Figure 6:

    MSE versus B for:
    - Benchmark Approach
    - RbCP
    - RbCP_time
    - SFC

Figure 6 regime
---------------
- P is fixed
- N0 is fixed
- SNR varies with B according to:
      SNR(B) = P / (B * N0)

Important modeling choices
--------------------------
1. Bandwidth sharing:
   Communication-budget-based quantities use the per-sensor bandwidth slice
   derived centrally through build_derived_system_parameters(cfg).

2. RbCP_time:
   Interpreted as the error-free time-model boundary:
       signal -> ta/tb -> events -> ta/tb -> signal
   i.e., NO physical channel and NO SFC/MAC layer here.

3. No semantic-based error detection:
   Figure 6 does NOT use semantic error detection.

4. Quantization policy:
   This pipeline relies on the defaults already implemented in the core:
   - M and M_RbCP are free integers by default
   - no power-of-two restriction unless explicitly requested in cfg
"""

import copy
import numpy as np
import pandas as pd

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.system_parameters import (
    build_derived_system_parameters,
    compute_benchmark_M_per_sensor,
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_rbcp_mse_vs_B_fixed_power_data(cfg):
    """
    Generate the Figure 6 dataset.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - B
        - SNR_dB_derived
        - M_benchmark_min
        - M_benchmark_mean
        - M_benchmark_max
        - M_time
        - M_rbcp
        - mse_benchmark
        - mse_rbcp
        - mse_rbcp_time
        - mse_sfc
        - num_trials
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    print("[INFO] Starting rbcp_mse_vs_B_fixed_power data generation")
    print(f"[INFO] Monte Carlo seed = {cfg['monte_carlo']['seed']}")
    print(
        f"[INFO] Sweep B from {cfg['sweep']['B']['start']} "
        f"to {cfg['sweep']['B']['stop']} "
        f"step {cfg['sweep']['B']['step']}"
    )
    print(f"[INFO] Trials per B = {cfg['monte_carlo']['interactions']}")

    b_cfg = cfg["sweep"]["B"]
    b_values = np.arange(b_cfg["start"], b_cfg["stop"], b_cfg["step"])

    results = []

    for B in b_values:
        cfg_B = copy.deepcopy(cfg)
        cfg_B["system"]["B"] = float(B)

        # ---------------------------------------------------------------------
        # Figure 6 regime: P fixed, N0 fixed, derive SNR(B)
        # ---------------------------------------------------------------------
        cfg_B["system"]["SNR_dB"] = _derive_snr_db_from_fixed_P_and_N0(cfg_B)

        params = build_derived_system_parameters(cfg_B)

        N = cfg_B["signal"].get("N_override", params.N)
        n_trials = cfg_B["monte_carlo"]["interactions"]

        # ---------------------------------------------------------------------
        # Diagnostic quantities from the core
        # ---------------------------------------------------------------------
        benchmark_cfg = cfg_B.get("benchmark", {})
        sampling_rate = benchmark_cfg.get("sampling_rate", params.W)
        effective_rate_factor = benchmark_cfg.get("effective_rate_factor", 1.0)

        # We preserve the historical role of effective_rate_factor by applying
        # it as an equivalent scaling of the sampling rate.
        M_benchmark_per_sensor = compute_benchmark_M_per_sensor(
            S=params.S,
            tau=params.tau,
            B=params.B,
            SNR=params.SNR,
            sampling_rate=effective_rate_factor * sampling_rate,
            bandwidth_allocation=params.bandwidth_allocation,
            force_power_of_two=params.quantization_force_power_of_two,
            rounding_mode=params.quantization_rounding_mode
        )

        M_benchmark_min = float(np.min(M_benchmark_per_sensor))
        M_benchmark_mean = float(np.mean(M_benchmark_per_sensor))
        M_benchmark_max = float(np.max(M_benchmark_per_sensor))
        M_time = int(params.M_time)
        M_rbcp = int(params.M_rbcp)

        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] B = {B:.1f} Hz")
        print(f"[INFO] P = {cfg_B['system']['P']:.6e} (fixed)")
        print(f"[INFO] N0 = {cfg_B['system']['N0']:.6e} (fixed)")
        print(f"[INFO] SNR_dB(B) = {cfg_B['system']['SNR_dB']:.6f}")
        print(f"[INFO] SNR(B) = {params.SNR:.6e}")
        print(
            f"[INFO] S = {params.S} | R = {params.R} | L = {params.L} | "
            f"tau = {params.tau:.3f} s | W = {params.W:.3f} Hz | N = {N}"
        )
        print(f"[INFO] bandwidth_allocation = {params.bandwidth_allocation}")
        print(f"[INFO] B_per_sensor = {params.B_per_sensor}")
        print(f"[INFO] M_benchmark_per_sensor = {M_benchmark_per_sensor}")
        print(f"[INFO] M_RbCP per sensor = {params.M_rbcp_per_sensor}")
        print(f"[INFO] M_benchmark_min = {M_benchmark_min}")
        print(f"[INFO] M_benchmark_mean = {M_benchmark_mean}")
        print(f"[INFO] M_benchmark_max = {M_benchmark_max}")
        print(f"[INFO] M_time = {M_time}")
        print(f"[INFO] M_RbCP = {M_rbcp}")
        print(f"[INFO] quantization_force_power_of_two = {params.quantization_force_power_of_two}")
        print(f"[INFO] quantization_rounding_mode = {params.quantization_rounding_mode}")

        # ---------------------------------------------------------------------
        # Reuse one SFC channel per B-point
        # ---------------------------------------------------------------------
        sfc_channel = None
        if cfg_B["mode"].get("run_sfc", False):
            sfc_channel = _build_sfc_channel_for_B(cfg_B, N, params.S)

        # ---------------------------------------------------------------------
        # Benchmark is analytical -> compute once per B
        # ---------------------------------------------------------------------
        mse_benchmark = _run_benchmark_branch(
            cfg=cfg_B,
            params=params,
            M_benchmark_per_sensor=M_benchmark_per_sensor
        )

        # ---------------------------------------------------------------------
        # Monte Carlo for the other branches
        # ---------------------------------------------------------------------
        mse_rbcp_sum = 0.0
        mse_rbcp_time_sum = 0.0
        mse_sfc_sum = 0.0

        for i in range(n_trials):
            trial = _run_one_trial(
                cfg_B,
                rng,
                N,
                M_rbcp=M_rbcp,
                sfc_channel=sfc_channel
            )

            mse_rbcp_sum += trial["mse_rbcp"]
            mse_rbcp_time_sum += trial["mse_rbcp_time"]
            mse_sfc_sum += trial["mse_sfc"]

            if (i + 1) % max(1, n_trials // 5) == 0:
                print(f"[INFO] Trial progress: {i + 1}/{n_trials}")

        mse_rbcp = mse_rbcp_sum / n_trials
        mse_rbcp_time = mse_rbcp_time_sum / n_trials
        mse_sfc = mse_sfc_sum / n_trials

        print(f"[INFO] mse_benchmark = {mse_benchmark:.6e}")
        print(f"[INFO] mse_rbcp = {mse_rbcp:.6e}")
        print(f"[INFO] mse_rbcp_time = {mse_rbcp_time:.6e}")
        print(f"[INFO] mse_sfc = {mse_sfc:.6e}")

        results.append({
            "B": B,
            "SNR_dB_derived": cfg_B["system"]["SNR_dB"],
            "M_benchmark_min": M_benchmark_min,
            "M_benchmark_mean": M_benchmark_mean,
            "M_benchmark_max": M_benchmark_max,
            "M_time": M_time,
            "M_rbcp": M_rbcp,
            "mse_benchmark": mse_benchmark,
            "mse_rbcp": mse_rbcp,
            "mse_rbcp_time": mse_rbcp_time,
            "mse_sfc": mse_sfc,
            "num_trials": n_trials,
        })

    return pd.DataFrame(results)


# =============================================================================
# FIXED-POWER SNR MODEL
# =============================================================================

def _derive_snr_db_from_fixed_P_and_N0(cfg):
    """
    Derive SNR_dB(B) from fixed P and fixed N0 using:

        SNR(B) = P / (B * N0)
    """

    P = cfg["system"]["P"]
    B = cfg["system"]["B"]
    N0 = cfg["system"]["N0"]

    snr = P / (B * N0)
    return 10.0 * np.log10(snr)


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
    sensor_x_event = np.zeros((S, num_event_ids))

    for s in range(S):
        start = 2 * s * N
        stop = 2 * (s + 1) * N
        sensor_x_event[s, start:stop] = 1.0

    return sensor_x_event


# =============================================================================
# ONE MONTE CARLO TRIAL
# =============================================================================

def _run_one_trial(cfg, rng, N, M_rbcp, sfc_channel=None):
    """
    Run one Monte Carlo trial and return the MSE values for:
    - RbCP
    - RbCP_time
    - SFC
    """

    params = build_derived_system_parameters(cfg)

    n_periods = 1
    tau = params.tau
    Tt = cfg["signal"]["Tt"]
    t = np.arange(0, tau, Tt)
    n_time = len(t)

    # 1. Generate random signals
    x_raw = _generate_signals(
        cfg=cfg,
        rng=rng,
        num_time_samples=n_time,
        n_periods=n_periods,
        S=params.S
    )

    # 2. Band-limit
    x_filtered = _filter_signals(x_raw, N, tau, Tt)

    # 3. Peak-to-peak control
    x_filtered = _apply_peak_to_peak_control(
        x_filtered,
        cfg["signal"]["peak_to_peak"]
    )

    # 4. DC handling
    x_zero_mean = _apply_dc_handling(
        x_filtered=x_filtered,
        tau=tau,
        Tt=Tt,
        dc_enabled=cfg.get("dc", {}).get("enabled", False)
    )

    # 5. Fourier coefficients
    an, bn, _ = _compute_fourier_coefficients(
        x_zero_mean=x_zero_mean,
        tau=tau,
        N=N,
        S=params.S,
        Tt=Tt,
        normalize_dft=cfg["signal"]["normalize_dft"],
        normalization_target=cfg["signal"]["normalization_target"]
    )

    # 6. ta/tb
    ta, tb = _compute_phase_coefficients(
        an=an,
        bn=bn,
        tau=tau,
        N=N,
        S=params.S,
        cfg=cfg,
        params=params,
        n_periods=n_periods
    )

    # 7. RbCP
    mse_rbcp = _run_rbcp_branch(
        ta=ta,
        tb=tb,
        x_ref=x_zero_mean,
        t=t,
        tau=tau,
        M_rbcp=M_rbcp
    )

    # 8. RbCP_time
    mse_rbcp_time = _run_rbcp_time_branch(
        ta=ta,
        tb=tb,
        cfg=cfg,
        params=params,
        N=N,
        x_ref=x_zero_mean,
        t=t,
        n_periods=n_periods
    )

    # 9. SFC
    mse_sfc = _run_sfc_branch(
        ta=ta,
        tb=tb,
        cfg=cfg,
        params=params,
        N=N,
        x_ref=x_zero_mean,
        t=t,
        n_periods=n_periods,
        sfc_channel=sfc_channel
    )

    return {
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

    raise ValueError("Invalid distribution")


def _filter_signals(x_raw, N, tau, Tt):
    """
    Band-limit each signal using W_eff = 2N / tau.
    """

    W_eff = 2 * N / tau
    x_filtered = np.zeros_like(x_raw)

    _, n_periods, S = x_raw.shape

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw[:, p, s],
                W_eff,
                Tt,
                tau
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

def _compute_fourier_coefficients(x_zero_mean, tau, N, S, Tt,
                                  normalize_dft, normalization_target):
    """
    Compute Fourier coefficients for all periods/sensors.
    """

    fourier_core = FourierCoefficientCore(
        T=tau,
        harmonics=N,
        sensor_nodes=S
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=normalize_dft,
        norm=normalization_target
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
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    ta, tb = phase_core.calc_ta_tb(an, bn)

    ta = np.real(ta)
    tb = np.real(tb)

    return ta, tb


# =============================================================================
# BENCHMARK BRANCH
# =============================================================================

def _run_benchmark_branch(cfg, params, M_benchmark_per_sensor):
    """
    Benchmark Approach for Figure 6.

    Analytical ABSOLUTE MSE:

        MSE_abs = (peak_to_peak^2) / (12 * M^2)

    where M is obtained from the core.
    """

    if not cfg["mode"].get("run_benchmark", False):
        return np.nan

    peak_to_peak = cfg["signal"]["peak_to_peak"]

    mse_sum = 0.0
    for M in M_benchmark_per_sensor:
        mse_sum += (peak_to_peak ** 2) / (12.0 * (float(M) ** 2))

    return mse_sum / len(M_benchmark_per_sensor)


# =============================================================================
# RbCP BRANCH
# =============================================================================

def _run_rbcp_branch(ta, tb, x_ref, t, tau, M_rbcp):
    """
    Run direct RbCP and return average MSE over sensors/periods.
    """

    ta_q, tb_q = quantize_ta_tb(
        ta,
        tb,
        2 * np.pi / tau,
        M_rbcp
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
                2 * np.pi / tau
            )

            mse_sum += np.mean((x_ref[:, p, s] - x_rec) ** 2)
            count += 1

    return mse_sum / count


# =============================================================================
# RbCP_time BRANCH
# =============================================================================

def _run_rbcp_time_branch(ta, tb, cfg, params, N, x_ref, t, n_periods):
    """
    Error-free time-model branch:

        signal -> ta/tb -> events -> ta/tb -> signal

    No physical channel is used here.
    """

    if not cfg["mode"].get("run_rbcp_time", False):
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
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    events = phase_core.ta_tb_to_events(ta, tb)
    ta_rec, tb_rec = phase_core.event_to_ta_tb(events)

    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    _, _, S = x_ref.shape
    mse_sum = 0.0
    count = 0

    for p in range(n_periods):
        for s in range(S):
            x_rec = recover_signal(
                ta_rec[p, :, s],
                tb_rec[p, :, s],
                t,
                w0
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

    IMPORTANT:
    - uses non-quantized ta/tb
    - does NOT use semantic error detection in Figure 6
    """

    if not cfg["mode"].get("run_sfc", False):
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
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    events = phase_core.ta_tb_to_events(ta, tb)

    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)

    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    _, _, S = x_ref.shape
    mse_sum = 0.0
    count = 0

    for p in range(n_periods):
        for s in range(S):
            x_rec = recover_signal(
                ta_rec[p, :, s],
                tb_rec[p, :, s],
                t,
                w0
            )

            mse_sum += np.mean((x_ref[:, p, s] - x_rec) ** 2)
            count += 1

    return mse_sum / count


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
        float_format="%.8e"
    )
