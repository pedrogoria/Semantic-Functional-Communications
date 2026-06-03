"""
sfc/pipelines/sfc_duplicate_rx.py

Lean pipeline for reproducing Figure 4 of the manuscript.

Figure 4 target
---------------
Average probability of receiving duplicate values (epsilon) versus B,
including:
1. theoretical upper bound from Lemma 5
2. Monte Carlo with random signals, clean channel
3. Monte Carlo with random signals, AWGN channel
4. Monte Carlo with uniformly distributed phase parameters t_z^(s,n), clean channel
5. Monte Carlo with uniformly distributed phase parameters t_z^(s,n), AWGN channel

Duplicate criterion
-------------------
A duplicate reception event is declared if, within one transmission cycle
of duration tau, the receiver estimates more than one value for the same
parameter t_z^(s,n).

In the event-matrix representation, this corresponds to:
    column_sum(event_id) > 1
for at least one event_id.

Important
---------
This pipeline does NOT reconstruct the signals.
Its sole objective is to evaluate epsilon(B).

Output columns
--------------
- B
- epsilon_upper_bound
- epsilon_random_clean
- epsilon_random_awgn
- epsilon_uniform_clean
- epsilon_uniform_awgn
- num_trials
"""

import copy
import numpy as np
import pandas as pd

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.system_parameters import build_derived_system_parameters
from sfc.core.theory import epsilon_upper_bound


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_sfc_duplicate_rx_data(cfg):
    """
    Generate the Figure 4 dataset.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - B
        - epsilon_upper_bound
        - epsilon_random_clean
        - epsilon_random_awgn
        - epsilon_uniform_clean
        - epsilon_uniform_awgn
        - num_trials
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    print("[INFO] Starting sfc_duplicate_rx data generation")
    print(f"[INFO] Monte Carlo seed = {cfg['monte_carlo']['seed']}")
    print(
        f"[INFO] Sweep B from {cfg['sweep']['B']['start']} "
        f"to {cfg['sweep']['B']['stop']} "
        f"step {cfg['sweep']['B']['step']}"
    )
    print(f"[INFO] Trials per B = {cfg['duplicate_rx']['interactions']}")

    # -------------------------------------------------------------------------
    # Sweep values of B
    # -------------------------------------------------------------------------
    b_cfg = cfg["sweep"]["B"]
    b_values = np.arange(b_cfg["start"], b_cfg["stop"], b_cfg["step"])

    results = []

    for B in b_values:
        cfg_B = copy.deepcopy(cfg)
        cfg_B["system"]["B"] = float(B)

        params = build_derived_system_parameters(cfg_B)

        # Figure 4 explicitly uses N=3 in the manuscript; allow override
        N = cfg_B["signal"].get("N_override", params.N)
        S = params.S

        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] B = {B:.1f} Hz")
        print(
            f"[INFO] S = {params.S} | R = {params.R} | L = {params.L} | "
            f"tau = {params.tau:.3f} s | W = {params.W:.3f} Hz | N = {N}"
        )
        print(f"[INFO] SNR_dB = {params.SNR_dB:.1f} | SNR = {params.SNR:.4e}")
        print(f"[INFO] bandwidth_allocation = {params.bandwidth_allocation}")
        print(f"[INFO] B_per_sensor = {params.B_per_sensor}")
        print(f"[INFO] M_time = {params.M_time}")
        print(f"[INFO] 2*N*S = {2 * N * S}")

        # ---------------------------------------------------------------------
        # One clean channel and one AWGN channel for this B-point
        # Reused in all trials, keeping the same maps/codebook.
        # ---------------------------------------------------------------------
        sfc_channel_clean, sfc_channel_awgn = _build_sfc_channel_pair_for_B(
            cfg_B=cfg_B,
            N=N,
            S=S
        )

        # ---------------------------------------------------------------------
        # Theoretical upper bound from Lemma 5
        # ---------------------------------------------------------------------
        eps_upper = epsilon_upper_bound(
            M_time=params.M_time,
            N=N,
            S=S
        )
        print(f"[INFO] epsilon_upper_bound = {eps_upper:.6e}")

        # ---------------------------------------------------------------------
        # Monte Carlo
        # ---------------------------------------------------------------------
        n_trials = cfg_B["duplicate_rx"]["interactions"]

        eps_random_clean = np.nan
        eps_random_awgn = np.nan
        eps_uniform_clean = np.nan
        eps_uniform_awgn = np.nan

        # ----------------------------
        # Random signals
        # ----------------------------
        if cfg_B["mode"].get("monte_carlo_random_signals", False):
            hits_clean = 0
            hits_awgn = 0

            for i in range(n_trials):
                dup_clean, dup_awgn = _trial_random_signals(
                    cfg=cfg_B,
                    rng=rng,
                    N=N,
                    sfc_channel_clean=sfc_channel_clean,
                    sfc_channel_awgn=sfc_channel_awgn
                )

                hits_clean += int(dup_clean)
                hits_awgn += int(dup_awgn)

                if (i + 1) % max(1, n_trials // 5) == 0:
                    print(f"[INFO] Random-signals MC progress: {i + 1}/{n_trials}")

            eps_random_clean = hits_clean / n_trials
            eps_random_awgn = hits_awgn / n_trials

            print(
                f"[INFO] Random signals results: "
                f"clean = {eps_random_clean:.6e} | awgn = {eps_random_awgn:.6e}"
            )

        # ----------------------------
        # Uniform t
        # ----------------------------
        if cfg_B["mode"].get("monte_carlo_uniform_t", False):
            hits_clean = 0
            hits_awgn = 0

            for i in range(n_trials):
                dup_clean, dup_awgn = _trial_uniform_t(
                    cfg=cfg_B,
                    rng=rng,
                    N=N,
                    sfc_channel_clean=sfc_channel_clean,
                    sfc_channel_awgn=sfc_channel_awgn
                )

                hits_clean += int(dup_clean)
                hits_awgn += int(dup_awgn)

                if (i + 1) % max(1, n_trials // 5) == 0:
                    print(f"[INFO] Uniform-t MC progress: {i + 1}/{n_trials}")

            eps_uniform_clean = hits_clean / n_trials
            eps_uniform_awgn = hits_awgn / n_trials

            print(
                f"[INFO] Uniform-t results: "
                f"clean = {eps_uniform_clean:.6e} | awgn = {eps_uniform_awgn:.6e}"
            )

        results.append({
            "B": B,
            "epsilon_upper_bound": eps_upper,
            "epsilon_random_clean": eps_random_clean,
            "epsilon_random_awgn": eps_random_awgn,
            "epsilon_uniform_clean": eps_uniform_clean,
            "epsilon_uniform_awgn": eps_uniform_awgn,
            "num_trials": n_trials,
        })

        print("[INFO] Finished current B-point")

    return pd.DataFrame(results)


# =============================================================================
# BUILD TWO CHANNELS PER B
# =============================================================================

def _build_sfc_channel_pair_for_B(cfg_B, N, S):
    """
    Build one CLEAN SFCChannel and one AWGN SFCChannel for the current B-point.

    The two channels reuse the same seed and sensor-event map, so the generated
    maps/codebook remain consistent for the current B.
    """

    def _make_cfg(base_cfg, channel_type):
        cfg_sfc = copy.deepcopy(base_cfg)

        if "channel" not in cfg_sfc:
            cfg_sfc["channel"] = {}

        cfg_sfc["channel"]["sensor_x_event"] = _build_sensor_x_event(S, N)
        cfg_sfc["channel"]["collision_mode"] = cfg_sfc["channel"].get("collision_mode", "sum")
        cfg_sfc["channel"]["type"] = channel_type
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

        # Compatibility with SFCChannel expected seed location
        if "reproducibility" not in cfg_sfc:
            cfg_sfc["reproducibility"] = {}

        if "seed" not in cfg_sfc["reproducibility"]:
            cfg_sfc["reproducibility"]["seed"] = cfg_B.get(
                "reproducibility", {}
            ).get(
                "seed",
                cfg_B.get("monte_carlo", {}).get("seed", 12345)
            )

        return cfg_sfc

    cfg_clean = _make_cfg(cfg_B, "clean")
    cfg_awgn = _make_cfg(cfg_B, "awgn")

    sfc_channel_clean = SFCChannel(cfg_clean)
    sfc_channel_awgn = SFCChannel(cfg_awgn)

    return sfc_channel_clean, sfc_channel_awgn


# =============================================================================
# RANDOM-SIGNALS MONTE CARLO TRIAL
# =============================================================================

def _trial_random_signals(cfg, rng, N, sfc_channel_clean, sfc_channel_awgn):
    """
    One Monte Carlo trial using random signals x_s(t).

    Returns
    -------
    tuple(bool, bool)
        (duplicate_in_clean, duplicate_in_awgn)
    """

    params = build_derived_system_parameters(cfg)

    # One period per duplicate test, consistent with "per tau seconds"
    n_periods = 1
    S = params.S
    Tt = cfg["signal"]["Tt"]
    tau = params.tau

    t = np.arange(0, tau, Tt)
    n_time = len(t)

    # -------------------------------------------------------------------------
    # 1. Generate S random signals
    # -------------------------------------------------------------------------
    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x_raw = rng.uniform(-1, 1, size=(n_time, n_periods, S))
    elif dist == "gaussian":
        x_raw = rng.normal(0, 1, size=(n_time, n_periods, S))
    else:
        raise ValueError("Invalid distribution")

    # -------------------------------------------------------------------------
    # 2. Band-limit
    # -------------------------------------------------------------------------
    # Use the configured source bandwidth W.
    # IMPORTANT:
    #   W is a signal/source parameter from the YAML.
    #   Do NOT redefine W from N.
    #
    # The relation between W and N should be handled when deriving N, e.g.:
    #   N = floor(W * tau / 2)
    #
    # But once W is configured, filtering must use W directly.
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

    # -------------------------------------------------------------------------
    # 3. Peak-to-peak control (optional)
    # -------------------------------------------------------------------------
    p2p_target = cfg["signal"]["peak_to_peak"]

    if p2p_target != 0:
        for s in range(S):
            current_p2p = np.max(x_filtered[:, 0, s]) - np.min(x_filtered[:, 0, s])
            if current_p2p != 0:
                x_filtered[:, 0, s] = x_filtered[:, 0, s] * (p2p_target / current_p2p)

    # -------------------------------------------------------------------------
    # 4. DC handling
    # -------------------------------------------------------------------------
    dc_enabled = cfg.get("dc", {}).get("enabled", False)
    x_zero_mean = np.zeros_like(x_filtered)

    for s in range(S):
        if not dc_enabled:
            dc = Tt * np.sum(x_filtered[:, 0, s]) / tau
            x_zero_mean[:, 0, s] = x_filtered[:, 0, s] - dc
        else:
            x_zero_mean[:, 0, s] = x_filtered[:, 0, s]

    # -------------------------------------------------------------------------
    # 5. Fourier coefficients
    # -------------------------------------------------------------------------
    fourier_core = FourierCoefficientCore(
        T=tau,
        harmonics=N,
        sensor_nodes=S
    )

    an, bn, _ = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=cfg["signal"]["normalize_dft"],
        norm=cfg["signal"]["normalization_target"]
    )

    # -------------------------------------------------------------------------
    # 6. ta/tb using PhaseCoefficientCore
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 7. ta/tb -> events
    # -------------------------------------------------------------------------
    events = phase_core.ta_tb_to_events(ta, tb)

    # -------------------------------------------------------------------------
    # 8. SFC channels
    # -------------------------------------------------------------------------
    out_clean = sfc_channel_clean(events)
    out_awgn = sfc_channel_awgn(events)

    events_est_clean = _extract_events_est(out_clean)
    events_est_awgn = _extract_events_est(out_awgn)

    # -------------------------------------------------------------------------
    # 9. Duplicate criterion
    # -------------------------------------------------------------------------
    dup_clean = _has_duplicate_reception(events_est_clean)
    dup_awgn = _has_duplicate_reception(events_est_awgn)

    return dup_clean, dup_awgn


# =============================================================================
# UNIFORM-T MONTE CARLO TRIAL
# =============================================================================

def _trial_uniform_t(cfg, rng, N, sfc_channel_clean, sfc_channel_awgn):
    """
    One Monte Carlo trial using uniformly distributed ta/tb.

    Returns
    -------
    tuple(bool, bool)
        (duplicate_in_clean, duplicate_in_awgn)
    """

    params = build_derived_system_parameters(cfg)

    n_periods = 1
    S = params.S
    tau = params.tau
    w0 = 2 * np.pi / tau

    # -------------------------------------------------------------------------
    # 1. Generate ta/tb directly in canonical intervals
    #
    # For each harmonic n:
    #   t_z^(s,n) ~ Uniform[-pi/(n w0), pi/(n w0)]
    # -------------------------------------------------------------------------
    ta = np.zeros((n_periods, N, S))
    tb = np.zeros((n_periods, N, S))

    for n_idx in range(N):
        n_h = n_idx + 1
        lo = -np.pi / (n_h * w0)
        hi = +np.pi / (n_h * w0)

        ta[0, n_idx, :] = rng.uniform(lo, hi, size=S)
        tb[0, n_idx, :] = rng.uniform(lo, hi, size=S)

    # -------------------------------------------------------------------------
    # 2. ta/tb -> events
    # -------------------------------------------------------------------------
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

    events = phase_core.ta_tb_to_events(ta, tb)

    # -------------------------------------------------------------------------
    # 3. SFC channels
    # -------------------------------------------------------------------------
    out_clean = sfc_channel_clean(events)
    out_awgn = sfc_channel_awgn(events)

    events_est_clean = _extract_events_est(out_clean)
    events_est_awgn = _extract_events_est(out_awgn)

    # -------------------------------------------------------------------------
    # 4. Duplicate criterion
    # -------------------------------------------------------------------------
    dup_clean = _has_duplicate_reception(events_est_clean)
    dup_awgn = _has_duplicate_reception(events_est_awgn)

    return dup_clean, dup_awgn


# =============================================================================
# SENSOR-EVENT ASSOCIATION
# =============================================================================

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
# SFC OUTPUT HELPERS
# =============================================================================

def _extract_events_est(channel_output):
    """
    Extract events_est from the SFC channel output.

    Supports both:
    - np.ndarray directly
    - dict-like outputs containing "events_est"
    """

    if isinstance(channel_output, dict):
        if "events_est" not in channel_output:
            raise KeyError("SFC channel output dict does not contain 'events_est'")
        return channel_output["events_est"]

    return channel_output


# =============================================================================
# DUPLICATE CRITERION
# =============================================================================

def _has_duplicate_reception(events_est):
    """
    Check whether at least one duplicate reception occurred.

    Interpretation
    --------------
    If the same event ID is detected in more than one time slot within the same
    cycle, then the receiver has more than one recovered value for the same
    t_z^(s,n), which is exactly the duplicate condition described in the paper.

    Parameters
    ----------
    events_est : np.ndarray
        Expected shape:
            (event_slots_total, num_event_ids)

    Returns
    -------
    bool
        True if there is a duplicate for at least one event ID.
    """

    assert len(events_est.shape) == 2, \
        "events_est must have shape (event_slots_total, num_event_ids)"

    col_sums = np.sum(events_est, axis=0)

    return np.any(col_sums > 1)


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
