"""
sfc/pipelines/rbcp_mse_vs_B.py

Pipeline for reproducing manuscript Figure 5:

    MSE versus B for:
    - Benchmark Approach
    - RbCP
    - RbCP_time
    - SFC

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
   Figure 5 does NOT use semantic error detection.

4. Manuscript-faithful power model:
   Keep SNR_dB fixed and N0 fixed, and derive P(B) from:
       SNR = P / (B * N0)
   Therefore:
       P(B) = SNR * B * N0
"""

import copy
import numpy as np
import pandas as pd

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.nyquist import Nyquist
from sfc.core.channel.SFCChannel import SFCChannel

from sfc.core.system_parameters import build_derived_system_parameters
from sfc.core.theory import (
    compute_snr_linear,
    compute_benchmark_bits_per_sample_single_sensor,
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

    # -------------------------------------------------------------------------
    # Sweep values of B
    # -------------------------------------------------------------------------
    b_cfg = cfg["sweep"]["B"]
    b_values = np.arange(b_cfg["start"], b_cfg["stop"], b_cfg["step"])

    results = []

    for B in b_values:
        cfg_B = copy.deepcopy(cfg)

        # ---------------------------------------------------------------------
        # Manuscript-faithful power model:
        # keep SNR fixed and N0 fixed, derive P(B)
        # ---------------------------------------------------------------------
        cfg_B["system"]["B"] = float(B)
        cfg_B["system"]["P"] = _derive_power_from_fixed_snr_and_n0(cfg_B)

        params = build_derived_system_parameters(cfg_B)

        N = cfg_B["signal"].get("N_override", params.N)
        S = params.S
        n_trials = cfg_B["monte_carlo"]["interactions"]

        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] B = {B:.1f} Hz")
        print(f"[INFO] P(B) = {cfg_B['system']['P']:.6e}")
        print(f"[INFO] S = {S} | R = {params.R} | L = {params.L} | tau = {params.tau:.3f} s | W = {params.W:.3f} Hz | N = {N}")
        print(f"[INFO] SNR_dB = {params.SNR_dB:.1f} | SNR = {params.SNR:.4e}")
        print(f"[INFO] N0 = {cfg_B['system']['N0']:.6e}")
        print(f"[INFO] bandwidth_allocation = {params.bandwidth_allocation}")
        print(f"[INFO] B_per_sensor = {params.B_per_sensor}")
        print(f"[INFO] M_RbCP = {params.M_rbcp}")
        print(f"[INFO] M_RbCP per sensor = {params.M_rbcp_per_sensor}")

        # ---------------------------------------------------------------------
        # Build one SFC channel for this B-point and reuse in all trials
        # ---------------------------------------------------------------------
        sfc_channel = None
        if cfg_B["mode"].get("run_sfc", False):
            sfc_channel = _build_sfc_channel_for_B(cfg_B, N, S)

        # ---------------------------------------------------------------------
        # Monte Carlo accumulation
        # ---------------------------------------------------------------------
        mse_benchmark_sum = 0.0
        mse_rbcp_sum = 0.0
        mse_rbcp_time_sum = 0.0
        mse_sfc_sum = 0.0

        for i in range(n_trials):
            trial = _run_one_trial(cfg_B, rng, N, sfc_channel=sfc_channel)

            mse_benchmark_sum += trial["mse_benchmark"]
            mse_rbcp_sum += trial["mse_rbcp"]
            mse_rbcp_time_sum += trial["mse_rbcp_time"]
            mse_sfc_sum += trial["mse_sfc"]

            if (i + 1) % max(1, n_trials // 5) == 0:
                print(f"[INFO] Trial progress: {i + 1}/{n_trials}")

        mse_benchmark = mse_benchmark_sum / n_trials
        mse_rbcp = mse_rbcp_sum / n_trials
        mse_rbcp_time = mse_rbcp_time_sum / n_trials
        mse_sfc = mse_sfc_sum / n_trials

        print(f"[INFO] mse_benchmark = {mse_benchmark:.6e}")
        print(f"[INFO] mse_rbcp = {mse_rbcp:.6e}")
        print(f"[INFO] mse_rbcp_time = {mse_rbcp_time:.6e}")
        print(f"[INFO] mse_sfc = {mse_sfc:.6e}")

        results.append({
            "B": B,
            "P_derived": cfg_B["system"]["P"],
            "mse_benchmark": mse_benchmark,
            "mse_rbcp": mse_rbcp,
            "mse_rbcp_time": mse_rbcp_time,
            "mse_sfc": mse_sfc,
            "num_trials": n_trials,
        })

    return pd.DataFrame(results)


# =============================================================================
# MANUSCRIPT-FAITHFUL POWER MODEL
# =============================================================================

def _derive_power_from_fixed_snr_and_n0(cfg):
    """
    Derive P(B) from fixed SNR and fixed N0 using:

        SNR = P / (B * N0)

    Therefore:
        P = SNR * B * N0
    """

    SNR = compute_snr_linear(cfg["system"]["SNR_dB"])
    B = cfg["system"]["B"]
    N0 = cfg["system"]["N0"]

    return SNR * B * N0


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

    # One period per trial for Figure 5
    n_periods = 1
    S = params.S
    tau = params.tau
    Tt = cfg["signal"]["Tt"]

    t = np.arange(0, tau, Tt)
    n_time = len(t)

    # -------------------------------------------------------------------------
    # 1. Generate S random signals
    # -------------------------------------------------------------------------
    x_raw = _generate_signals(
        cfg=cfg,
        rng=rng,
        num_time_samples=n_time,
        n_periods=n_periods,
        S=S
    )

    # -------------------------------------------------------------------------
    # 2. Band-limit
    # -------------------------------------------------------------------------
    x_filtered = _filter_signals(x_raw, N, tau, Tt)

    # -------------------------------------------------------------------------
    # 3. Peak-to-peak control
    # -------------------------------------------------------------------------
    x_filtered = _apply_peak_to_peak_control(
        x_filtered,
        cfg["signal"]["peak_to_peak"]
    )

    # -------------------------------------------------------------------------
    # 4. DC handling
    # -------------------------------------------------------------------------
    x_zero_mean = _apply_dc_handling(
        x_filtered=x_filtered,
        tau=tau,
        Tt=Tt,
        dc_enabled=cfg.get("dc", {}).get("enabled", False)
    )

    # -------------------------------------------------------------------------
    # 5. Fourier coefficients
    # -------------------------------------------------------------------------
    an, bn, x_used = _compute_fourier_coefficients(
        x_zero_mean=x_zero_mean,
        tau=tau,
        N=N,
        S=S,
        Tt=Tt,
        normalize_dft=cfg["signal"]["normalize_dft"],
        normalization_target=cfg["signal"]["normalization_target"]
    )

    # -------------------------------------------------------------------------
    # 6. ta/tb
    # -------------------------------------------------------------------------
    ta, tb = _compute_phase_coefficients(
        an=an,
        bn=bn,
        tau=tau,
        N=N,
        S=S,
        cfg=cfg,
        params=params,
        n_periods=n_periods
    )

    # -------------------------------------------------------------------------
    # 7. Benchmark
    # -------------------------------------------------------------------------
    mse_benchmark = _run_benchmark_branch(
        cfg=cfg,
        params=params
    )

    # -------------------------------------------------------------------------
    # 8. RbCP
    # -------------------------------------------------------------------------
    mse_rbcp = _run_rbcp_branch(
        ta=ta,
        tb=tb,
        x_ref=x_zero_mean,
        t=t,
        tau=tau,
        M_rbcp=params.M_rbcp
    )

    # -------------------------------------------------------------------------
    # 9. RbCP_time (error-free time model, no channel)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 10. SFC
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
        sfc_channel=sfc_channel
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

    elif dist == "gaussian":
        return rng.normal(0, 1, size=(num_time_samples, n_periods, S))

    else:
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

def _compute_fourier_coefficients(x_zero_mean, tau, N, S, Tt, normalize_dft, normalization_target):
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

def _run_benchmark_branch(cfg, params):
    """
    Benchmark Approach for Figure 5.

    IMPORTANT
    ---------
    This branch is treated analytically, not through Monte Carlo reconstruction.

    The benchmark uses:
        MSE = (peak_to_peak^2) / (12 * M^2)

    where M is the number of quantization bins for the Benchmark Approach.

    Design choices
    --------------
    1. The total communication budget is converted into an effective number
       of bins M.

    2. Bandwidth sharing is respected through:
           params.B_per_sensor[s]

    3. A configurable rate scaling factor is allowed:
           effective_rate_factor

       so that the effective bits/sample become:

           bits_per_sample =
               [tau * B_sensor * log2(1 + SNR)] /
               [effective_rate_factor * tau * sampling_rate]

       which simplifies to:

           bits_per_sample =
               [B_sensor * log2(1 + SNR)] /
               [effective_rate_factor * sampling_rate]

    4. The resulting M is taken as:
           M = floor(2 ** bits_per_sample)

       and the absolute (not normalized) MSE is:
           (peak_to_peak^2) / (12 * M^2)

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary.

    params : DerivedSystemParameters
        Centralized parameter object.

    Returns
    -------
    float
        Average benchmark MSE across sensors.
    """

    if not cfg["mode"].get("run_benchmark", False):
        return np.nan

    benchmark_cfg = cfg.get("benchmark", {})

    sampling_rate = benchmark_cfg.get("sampling_rate", params.W)

    # ------------------------------------------------------------------
    # IMPORTANT TUNING KNOB
    #
    # For manuscript/figure matching, this factor can be set to 2.0.
    # If you want the strictest direct interpretation, use 1.0.
    # ------------------------------------------------------------------
    effective_rate_factor = benchmark_cfg.get("effective_rate_factor", 2.0)

    peak_to_peak = cfg["signal"]["peak_to_peak"]

    mse_sum = 0.0

    for s in range(params.S):
        B_sensor = params.B_per_sensor[s]

        # --------------------------------------------------------------
        # Effective bits per sample for this sensor
        # --------------------------------------------------------------
        bits_per_sample = (
            B_sensor * np.log2(1.0 + params.SNR)
        ) / (
            effective_rate_factor * sampling_rate
        )

        # Keep at least one bin
        bits_per_sample = max(bits_per_sample, 1e-12)

        bits_int = int(np.floor(bits_per_sample))
        bits_int = max(bits_int, 1)

        # Number of bins for the Benchmark quantizer
        M = int(2 ** bits_int)

        # Absolute MSE (not normalized)
        mse_s = (peak_to_peak ** 2) / (12.0 * (M ** 2))

        mse_sum += mse_s

    return mse_sum / params.S


# =============================================================================
# RBCP BRANCH
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
# RBCP_time BRANCH
# =============================================================================

def _run_rbcp_time_branch(ta, tb, cfg, params, N, x_ref, t, n_periods):
    """
    Error-free time-model branch:

        signal -> ta/tb -> events -> ta/tb -> signal

    No physical channel is used here.
    """

    if not cfg["mode"].get("run_rbcp_time", False):
        return np.nan

    S = params.S
    w0 = 2 * np.pi / params.tau

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=S,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
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
    - does NOT use semantic error detection in Figure 5
    """

    if not cfg["mode"].get("run_sfc", False):
        return np.nan

    S = params.S
    w0 = 2 * np.pi / params.tau

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=S,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    events = phase_core.ta_tb_to_events(ta, tb)
    events_est = sfc_channel(events)

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
