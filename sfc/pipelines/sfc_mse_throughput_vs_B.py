"""
sfc/pipelines/sfc_mse_throughput_vs_B.py

Pipeline per transmission cycle of tau seconds.Pipeline for reproducing manuscript Figure 7-style experiment:

   With the current semantic-based error-detection policy:
   - if one period is valid, all S signals of that period are counted as
     successfully received;
   - if one period is invalid, zero signals are counted.

   Therefore:
       throughput = (S * num_valid_periods) / (num_periods * tau)

   Since this pipeline uses n_periods = 1 per Monte Carlo trial, the empirical
   throughput becomes:
       throughput = (S / tau) * valid_period_fraction

4. Figure-6-style physical regime:
   - P is fixed
   - N0 is fixed
   - SNR varies with B according to:
         SNR(B) = P / (B * N0)

5. IMPORTANT CORRECTION:
   If the SED discards an invalid period, that period must NOT be included in the
   SFC MSE average. Therefore:
   - throughput is averaged across all trials
   - mse_sfc is averaged only across valid trials/periods
"""

import copy
import numpy as np
import pandas as pd

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.reconstruction import recover_signal
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.system_parameters import build_derived_system_parameters
from sfc.core.semantic_error_detection import detect_semantic_errors


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_sfc_mse_throughput_vs_B_data(cfg):
    """
    Generate the Figure 7-style dataset.

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
        - mse_sfc
        - mse_rbcp_time
        - throughput_sfc
        - throughput_max
        - valid_period_fraction
        - num_valid_trials
        - num_trials
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    print("[INFO] Starting sfc_mse_throughput_vs_B data generation")
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
        # Figure-7 / fixed-power regime:
        # keep P fixed and N0 fixed, derive SNR(B)
        # ---------------------------------------------------------------------
        cfg_B["system"]["SNR_dB"] = _derive_snr_db_from_fixed_P_and_N0(cfg_B)

        params = build_derived_system_parameters(cfg_B)

        N = cfg_B["signal"].get("N_override", params.N)
        n_trials = cfg_B["monte_carlo"]["interactions"]

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
        print(f"[INFO] M_time = {params.M_time}")
        print(f"[INFO] semantic_error_detection = {cfg_B['mode'].get('semantic_error_detection', False)}")

        # ---------------------------------------------------------------------
        # Build one SFC channel for this B-point and reuse in all trials
        # ---------------------------------------------------------------------
        sfc_channel = None
        if cfg_B["mode"].get("run_sfc", False):
            sfc_channel = _build_sfc_channel_for_B(cfg_B, N, params.S)

        # ---------------------------------------------------------------------
        # Monte Carlo accumulation
        # ---------------------------------------------------------------------
        mse_sfc_sum = 0.0
        n_valid_trials = 0

        mse_rbcp_time_sum = 0.0
        throughput_sum = 0.0
        valid_fraction_sum = 0.0

        for i in range(n_trials):
            trial = _run_one_trial(
                cfg=cfg_B,
                rng=rng,
                N=N,
                sfc_channel=sfc_channel
            )

            # RbCP_time is always averaged over all trials
            mse_rbcp_time_sum += trial["mse_rbcp_time"]

            # Throughput is always averaged over all trials
            throughput_sum += trial["throughput_sfc"]
            valid_fraction_sum += trial["valid_period_fraction"]

            # SFC MSE must be averaged only over valid trials
            if trial["is_valid_trial"] and not np.isnan(trial["mse_sfc"]):
                mse_sfc_sum += trial["mse_sfc"]
                n_valid_trials += 1

            if (i + 1) % max(1, n_trials // 5) == 0:
                print(f"[INFO] Trial progress: {i + 1}/{n_trials}")

        if n_valid_trials > 0:
            mse_sfc = mse_sfc_sum / n_valid_trials
        else:
            mse_sfc = np.nan

        mse_rbcp_time = mse_rbcp_time_sum / n_trials
        throughput_sfc = throughput_sum / n_trials
        valid_period_fraction = valid_fraction_sum / n_trials
        throughput_max = params.S / params.tau

        print(f"[INFO] mse_sfc = {mse_sfc:.6e}" if not np.isnan(mse_sfc) else "[INFO] mse_sfc = nan (no valid trials)")
        print(f"[INFO] mse_rbcp_time = {mse_rbcp_time:.6e}")
        print(f"[INFO] throughput_sfc = {throughput_sfc:.6e}")
        print(f"[INFO] throughput_max = {throughput_max:.6e}")
        print(f"[INFO] valid_period_fraction = {valid_period_fraction:.6e}")
        print(f"[INFO] num_valid_trials = {n_valid_trials}/{n_trials}")

        results.append({
            "B": B,
            "SNR_dB_derived": cfg_B["system"]["SNR_dB"],
            "mse_sfc": mse_sfc,
            "mse_rbcp_time": mse_rbcp_time,
            "throughput_sfc": throughput_sfc,
            "throughput_max": throughput_max,
            "valid_period_fraction": valid_period_fraction,
            "num_valid_trials": n_valid_trials,
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

def _run_one_trial(cfg, rng, N, sfc_channel=None):
    """
    Run one Monte Carlo trial and return:

    - mse_sfc
    - mse_rbcp_time
    - throughput_sfc
    - valid_period_fraction
    - is_valid_trial
    """

    params = build_derived_system_parameters(cfg)

    # One period per trial
    n_periods = 1
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
        S=params.S
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
    an, bn, _ = _compute_fourier_coefficients(
        x_zero_mean=x_zero_mean,
        tau=tau,
        N=N,
        S=params.S,
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
        S=params.S,
        cfg=cfg,
        params=params,
        n_periods=n_periods
    )

    # -------------------------------------------------------------------------
    # 7. RbCP_time (error-free time model)
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
    # 8. SFC + semantic-based error detection
    # -------------------------------------------------------------------------
    mse_sfc, throughput_sfc, valid_period_fraction, is_valid_trial = _run_sfc_branch_with_sed(
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
        "mse_sfc": mse_sfc,
        "mse_rbcp_time": mse_rbcp_time,
        "throughput_sfc": throughput_sfc,
        "valid_period_fraction": valid_period_fraction,
        "is_valid_trial": is_valid_trial,
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
# RbCP_time BRANCH
# =============================================================================

def _run_rbcp_time_branch(ta, tb, cfg, params, N, x_ref, t, n_periods):
    """
    Error-free time-model branch:

        signal -> ta/tb -> events -> ta/tb -> signal
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
# SFC + SED BRANCH
# =============================================================================

def _run_sfc_branch_with_sed(ta, tb, cfg, params, N, x_ref, t, n_periods, sfc_channel):
    """
    Run SFC branch with semantic-based error detection.

    IMPORTANT
    ---------
    Discarded invalid periods are NOT included in the MSE average.

    Returns
    -------
    tuple
        (mse_sfc, throughput_sfc, valid_period_fraction, is_valid_trial)
    """

    if not cfg["mode"].get("run_sfc", False):
        return np.nan, np.nan, np.nan, False

    if sfc_channel is None:
        return np.nan, np.nan, np.nan, False

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

    # Channel
    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    # -------------------------------------------------------------------------
    # IMPORTANT FIX
    # -------------------------------------------------------------------------
    # The SFC channel may return an event matrix whose number of rows differs
    # slightly from params.M_time due to the internal implementation.
    # Since this pipeline uses n_periods = 1, the safest segmentation for SED is
    # to use the ACTUAL number of rows observed at the channel output.
    # -------------------------------------------------------------------------
    event_slots_total = events_est.shape[0]

    if n_periods == 1:
        period_slots_for_sed = event_slots_total
    else:
        if event_slots_total % n_periods != 0:
            raise ValueError(
                f"event_slots_total={event_slots_total} is not divisible by "
                f"n_periods={n_periods}. Period segmentation is ambiguous."
            )
        period_slots_for_sed = event_slots_total // n_periods

    # Semantic-based error detection
    sensor_x_event = _build_sensor_x_event(params.S, N)
    sed_cfg = cfg.get("sed", {})
    discard_invalid_periods = sed_cfg.get("discard_invalid_periods", True)

    sed_result = detect_semantic_errors(
        events_est=events_est,
        period_slots=period_slots_for_sed,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=discard_invalid_periods
    )

    corrected_events_est, period_valid_mask = _extract_sed_outputs(sed_result)

    # Throughput is averaged over all periods/trials
    valid_fraction = float(np.mean(period_valid_mask))
    throughput = (params.S * np.sum(period_valid_mask)) / (len(period_valid_mask) * params.tau)

    # In the current Figure-7 pipeline, n_periods = 1
    is_valid_trial = bool(period_valid_mask[0])

    # If the only period in this trial is invalid, do NOT compute MSE.
    if not is_valid_trial:
        return np.nan, float(throughput), valid_fraction, False

    # Reconstruction from corrected events only if valid
    ta_rec, tb_rec = phase_core.event_to_ta_tb(corrected_events_est)
    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    _, _, S = x_ref.shape
    mse_sum = 0.0
    count = 0

    for p in range(n_periods):
        # Only valid periods enter the MSE
        if not period_valid_mask[p]:
            continue

        for s in range(S):
            x_rec = recover_signal(
                ta_rec[p, :, s],
                tb_rec[p, :, s],
                t,
                w0
            )

            mse_sum += np.mean((x_ref[:, p, s] - x_rec) ** 2)
            count += 1

    if count == 0:
        return np.nan, float(throughput), valid_fraction, False

    mse_sfc = mse_sum / count

    return mse_sfc, float(throughput), valid_fraction, True


def _extract_sed_outputs(sed_result):
    """
    Extract (corrected_events_est, period_valid_mask) from the SED output.

    Supports:
    - dataclass-like result with attributes
    - dict-like result
    """

    if isinstance(sed_result, dict):
        corrected_events_est = sed_result["corrected_events_est"]
        period_valid_mask = np.asarray(sed_result["period_valid_mask"], dtype=bool)
        return corrected_events_est, period_valid_mask

    corrected_events_est = getattr(sed_result, "corrected_events_est")
    period_valid_mask = np.asarray(getattr(sed_result, "period_valid_mask"), dtype=bool)

    return corrected_events_est, period_valid_mask


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
