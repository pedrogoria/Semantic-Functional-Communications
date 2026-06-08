"""
sfc/pipelines/benchmark_ppm_sfc_vs_B.py

Fair comparison pipeline for:

    1. Benchmark + FDMA
    2. PPM + FDMA
    3. RbCP_time
    4. SFC
    5. SFC + SED

Comparison principle
--------------------
All five stacks are compared under the same physical communication budget:

- total communication bandwidth: B
- number of sensors: S
- average available transmit power per sensor: P
- same tau
- same source signals x^(s)(t)

Two physical-budget regimes are supported through the YAML:

1. fixed_P_and_N0
   - keep P fixed
   - keep N0 fixed
   - derive:
         SNR(B) = P / (B * N0)

2. fixed_P_and_SNR
   - keep P fixed
   - keep SNR fixed
   - derive:
         N0(B) = P / (B * SNR)

Design choices
--------------
1. Benchmark + FDMA
   - uniformly sample the source signal
   - scalar quantization under the feasible communication budget
   - FDMA interpretation: each sensor receives a bandwidth slice B_s
   - continuous-time reconstruction via sinc interpolation
   - truncation policy for sinc:
         periodic_replicas = 10
     i.e. replicas from -10*tau to +10*tau

2. PPM + FDMA
   - same source signal as Benchmark
   - each sensor receives an FDMA bandwidth slice B_s
   - power budget per sensor remains P
   - AWGN is applied consistently with the FDMA slice of each sensor
   - continuous-time reconstruction is the one provided by PPMCore, which
     should follow the SAME periodic sinc truncation convention:
         periodic_replicas = 10

3. RbCP_time
   - error-free time-model boundary:
         signal -> ta/tb -> events -> ta/tb -> signal
   - no physical channel
   - native functional reconstruction via recover_signal(...)

4. SFC
   - native SFC stack without semantic error detection
   - chain:
         signal -> Fourier -> ta/tb -> events -> SFCChannel -> events_est
                -> ta/tb_est -> recover_signal(...)

5. SFC + SED
   - native SFC stack with semantic-based error detection
   - invalid periods are discarded
   - IMPORTANT:
         invalid periods do NOT contribute to the MSE
   - therefore:
         mse_sfc_sed is averaged only over valid trials
         valid_period_fraction_sfc_sed is averaged over all trials

IMPORTANT
---------
This file is the correct integration point for the fair-comparison logic.

Current expected supporting modules
-----------------------------------
- sfc.core.mac.fdma.FDMACore
- sfc.core.modulation.ppm.PPMCore   (if already stabilized)
- existing SFC core components already used elsewhere in the project
- sfc.core.semantic_error_detection.detect_semantic_errors
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.reconstruction import recover_signal
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.system_parameters import (
    build_derived_system_parameters,
    compute_benchmark_M_per_sensor,
)
from sfc.core.mac.fdma import FDMACore
from sfc.core.mac.base import MACInput
from sfc.core.semantic_error_detection import detect_semantic_errors

# PPM is optional while the new modulation architecture stabilizes
try:
    from sfc.core.modulation.ppm import PPMCore
except Exception:
    PPMCore = None


# =============================================================================
# GLOBAL FAIR-COMPARISON CHOICE
# =============================================================================

PERIODIC_REPLICAS = 10


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_benchmark_ppm_sfc_vs_B_data(cfg):
    """
    Generate the fair-comparison dataset.

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
        - N0_derived
        - mse_benchmark_fdma
        - mse_ppm_fdma
        - mse_rbcp_time
        - mse_sfc
        - mse_sfc_sed
        - valid_period_fraction_sfc_sed
        - num_valid_trials_sfc_sed
        - num_trials
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    b_cfg = cfg["sweep"]["B"]
    b_values = np.arange(b_cfg["start"], b_cfg["stop"], b_cfg["step"])

    print("[INFO] Starting fair comparison:")
    print("[INFO]   Benchmark+FDMA vs PPM+FDMA vs RbCP_time vs SFC vs SFC+SED")
    print(f"[INFO] Monte Carlo seed = {cfg['monte_carlo']['seed']}")
    print(
        f"[INFO] Sweep B from {b_cfg['start']} to {b_cfg['stop']} "
        f"step {b_cfg['step']}"
    )
    print(f"[INFO] periodic_replicas = {PERIODIC_REPLICAS}")

    results = []

    for B in b_values:
        cfg_B = copy.deepcopy(cfg)
        cfg_B["system"]["B"] = float(B)

        # ---------------------------------------------------------------------
        # Resolve physical-budget regime BEFORE build_derived_system_parameters
        # ---------------------------------------------------------------------
        _apply_power_model(cfg_B)

        params = build_derived_system_parameters(cfg_B)

        N = cfg_B["signal"].get("N_override", params.N)
        n_trials = cfg_B["monte_carlo"]["interactions"]

        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] B = {B:.6f}")
        print(f"[INFO] power_model.mode = {cfg_B.get('power_model', {}).get('mode', 'fixed_P_and_N0')}")
        print(f"[INFO] P = {params.P}")
        print(f"[INFO] N0_derived = {cfg_B['system']['N0']}")
        print(f"[INFO] SNR_dB_derived = {cfg_B['system']['SNR_dB']:.6f}")
        print(f"[INFO] S = {params.S}")
        print(f"[INFO] tau = {params.tau}")
        print(f"[INFO] W = {params.W}")
        print(f"[INFO] B_per_sensor = {params.B_per_sensor}")
        print(f"[INFO] distribution = {cfg_B['signal']['distribution']}")
        print(f"[INFO] semantic_error_detection = {cfg_B['mode'].get('semantic_error_detection', False)}")

        # ------------------------------------------------------------
        # Build native SFC channel once per B-point
        # ------------------------------------------------------------
        sfc_channel = None
        if cfg_B["mode"].get("run_sfc", True):
            sfc_channel = _build_sfc_channel_for_B(cfg_B, N, params.S)

        mse_benchmark_sum = 0.0
        mse_ppm_sum = 0.0
        mse_rbcp_time_sum = 0.0
        mse_sfc_sum = 0.0

        mse_sfc_sed_sum = 0.0
        num_valid_trials_sfc_sed = 0
        valid_period_fraction_sfc_sed_sum = 0.0

        for i in range(n_trials):
            trial = _run_one_trial(
                cfg=cfg_B,
                rng=rng,
                params=params,
                N=N,
                sfc_channel=sfc_channel
            )

            mse_benchmark_sum += trial["mse_benchmark_fdma"]
            mse_ppm_sum += trial["mse_ppm_fdma"]
            mse_rbcp_time_sum += trial["mse_rbcp_time"]
            mse_sfc_sum += trial["mse_sfc"]

            valid_period_fraction_sfc_sed_sum += trial["valid_period_fraction_sfc_sed"]

            if trial["is_valid_sfc_sed_trial"] and not np.isnan(trial["mse_sfc_sed"]):
                mse_sfc_sed_sum += trial["mse_sfc_sed"]
                num_valid_trials_sfc_sed += 1

            if (i + 1) % max(1, n_trials // 5) == 0:
                print(f"[INFO] Trial progress: {i + 1}/{n_trials}")

        mse_benchmark = mse_benchmark_sum / n_trials
        mse_ppm = mse_ppm_sum / n_trials
        mse_rbcp_time = mse_rbcp_time_sum / n_trials
        mse_sfc = mse_sfc_sum / n_trials

        if num_valid_trials_sfc_sed > 0:
            mse_sfc_sed = mse_sfc_sed_sum / num_valid_trials_sfc_sed
        else:
            mse_sfc_sed = np.nan

        valid_period_fraction_sfc_sed = valid_period_fraction_sfc_sed_sum / n_trials

        print(f"[INFO] mse_benchmark_fdma = {mse_benchmark:.6e}")
        print(f"[INFO] mse_ppm_fdma = {mse_ppm:.6e}")
        print(f"[INFO] mse_rbcp_time = {mse_rbcp_time:.6e}")
        print(f"[INFO] mse_sfc = {mse_sfc:.6e}")
        if np.isnan(mse_sfc_sed):
            print("[INFO] mse_sfc_sed = nan (no valid SFC+SED trials)")
        else:
            print(f"[INFO] mse_sfc_sed = {mse_sfc_sed:.6e}")
        print(f"[INFO] valid_period_fraction_sfc_sed = {valid_period_fraction_sfc_sed:.6e}")
        print(f"[INFO] num_valid_trials_sfc_sed = {num_valid_trials_sfc_sed}/{n_trials}")

        results.append({
            "B": B,
            "SNR_dB_derived": cfg_B["system"]["SNR_dB"],
            "N0_derived": cfg_B["system"]["N0"],
            "mse_benchmark_fdma": mse_benchmark,
            "mse_ppm_fdma": mse_ppm,
            "mse_rbcp_time": mse_rbcp_time,
            "mse_sfc": mse_sfc,
            "mse_sfc_sed": mse_sfc_sed,
            "valid_period_fraction_sfc_sed": valid_period_fraction_sfc_sed,
            "num_valid_trials_sfc_sed": num_valid_trials_sfc_sed,
            "num_trials": n_trials,
        })

    return pd.DataFrame(results)


# =============================================================================
# POWER-MODEL SELECTION
# =============================================================================

def _apply_power_model(cfg):
    """
    Resolve the selected physical-budget regime in-place.

    Supported modes
    ---------------
    1. fixed_P_and_N0
       - keep P fixed
       - keep N0 fixed
       - derive SNR_dB(B)

    2. fixed_P_and_SNR
       - keep P fixed
       - keep SNR_dB fixed
       - derive N0(B)
    """

    mode = cfg.get("power_model", {}).get("mode", "fixed_P_and_N0")

    if mode == "fixed_P_and_N0":
        cfg["system"]["SNR_dB"] = _derive_snr_db_from_fixed_P_and_N0(cfg)
        return

    if mode == "fixed_P_and_SNR":
        cfg["system"]["N0"] = _derive_n0_from_fixed_P_and_snr(cfg)
        return

    raise ValueError(
        f"Unsupported power_model.mode: {mode}. "
        f"Supported modes are: 'fixed_P_and_N0', 'fixed_P_and_SNR'."
    )


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


def _derive_n0_from_fixed_P_and_snr(cfg):
    """
    Derive N0(B) from fixed P and fixed SNR using:

        N0(B) = P / (B * SNR)
    """

    P = cfg["system"]["P"]
    B = cfg["system"]["B"]
    SNR_dB = cfg["system"]["SNR_dB"]

    snr = 10.0 ** (SNR_dB / 10.0)
    return P / (B * snr)


# =============================================================================
# ONE MONTE CARLO TRIAL
# =============================================================================

def _run_one_trial(cfg, rng, params, N, sfc_channel=None):
    """
    Run one fair-comparison trial.

    Returns
    -------
    dict
        {
            "mse_benchmark_fdma": ...,
            "mse_ppm_fdma": ...,
            "mse_rbcp_time": ...,
            "mse_sfc": ...,
            "mse_sfc_sed": ...,
            "valid_period_fraction_sfc_sed": ...,
            "is_valid_sfc_sed_trial": ...,
        }
    """

    # ------------------------------------------------------------
    # 1. Generate COMMON source signals for all stacks
    # ------------------------------------------------------------
    common = _generate_common_bandlimited_signals(
        cfg=cfg,
        rng=rng,
        params=params,
        N=N
    )

    x_ref = common["x_ref"]      # shape: (time, periods=1, sensors)
    t = common["t"]
    Tt = common["Tt"]

    # ------------------------------------------------------------
    # 2. Shared Fourier / ta-tb representation for RbCP_time, SFC and SFC+SED
    # ------------------------------------------------------------
    ta, tb = _compute_ta_tb_from_reference(
        cfg=cfg,
        params=params,
        x_ref=x_ref,
        Tt=Tt,
        N=N
    )

    # ------------------------------------------------------------
    # 3. Benchmark + FDMA
    # ------------------------------------------------------------
    mse_benchmark = _run_benchmark_fdma_stack(
        cfg=cfg,
        params=params,
        x_ref=x_ref,
        t=t,
        periodic_replicas=PERIODIC_REPLICAS
    )

    # ------------------------------------------------------------
    # 4. PPM + FDMA
    # ------------------------------------------------------------
    mse_ppm = _run_ppm_fdma_stack(
        cfg=cfg,
        params=params,
        x_ref=x_ref,
        t=t,
        periodic_replicas=PERIODIC_REPLICAS,
        rng=rng
    )

    # ------------------------------------------------------------
    # 5. RbCP_time
    # ------------------------------------------------------------
    mse_rbcp_time = _run_rbcp_time_stack(
        cfg=cfg,
        params=params,
        ta=ta,
        tb=tb,
        x_ref=x_ref,
        t=t,
        N=N
    )

    # ------------------------------------------------------------
    # 6. Native SFC (without SED)
    # ------------------------------------------------------------
    mse_sfc = _run_sfc_stack(
        cfg=cfg,
        params=params,
        ta=ta,
        tb=tb,
        x_ref=x_ref,
        t=t,
        N=N,
        sfc_channel=sfc_channel
    )

    # ------------------------------------------------------------
    # 7. Native SFC + SED
    # ------------------------------------------------------------
    mse_sfc_sed, valid_fraction_sfc_sed, is_valid_sfc_sed_trial = _run_sfc_sed_stack(
        cfg=cfg,
        params=params,
        ta=ta,
        tb=tb,
        x_ref=x_ref,
        t=t,
        N=N,
        sfc_channel=sfc_channel
    )

    return {
        "mse_benchmark_fdma": mse_benchmark,
        "mse_ppm_fdma": mse_ppm,
        "mse_rbcp_time": mse_rbcp_time,
        "mse_sfc": mse_sfc,
        "mse_sfc_sed": mse_sfc_sed,
        "valid_period_fraction_sfc_sed": valid_fraction_sfc_sed,
        "is_valid_sfc_sed_trial": is_valid_sfc_sed_trial,
    }


# =============================================================================
# COMMON SOURCE SIGNALS
# =============================================================================

def _generate_common_bandlimited_signals(cfg, rng, params, N):
    """
    Generate the COMMON source signals used by all stacks.
    """

    tau = params.tau
    Tt = cfg["signal"]["Tt"]
    S = params.S
    n_periods = 1

    t = np.arange(0.0, tau, Tt)
    n_time = len(t)

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=(n_time, n_periods, S))
    elif dist == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=(n_time, n_periods, S))
    else:
        raise ValueError("Invalid distribution")

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

    # Peak-to-peak control
    peak_to_peak = cfg["signal"]["peak_to_peak"]
    if peak_to_peak != 0:
        for p in range(n_periods):
            for s in range(S):
                current_p2p = np.max(x_filtered[:, p, s]) - np.min(x_filtered[:, p, s])
                if current_p2p != 0:
                    x_filtered[:, p, s] *= peak_to_peak / current_p2p

    # DC handling
    dc_enabled = cfg.get("dc", {}).get("enabled", False)
    x_zero_mean = np.zeros_like(x_filtered)

    for p in range(n_periods):
        for s in range(S):
            if not dc_enabled:
                dc = Tt * np.sum(x_filtered[:, p, s]) / tau
                x_zero_mean[:, p, s] = x_filtered[:, p, s] - dc
            else:
                x_zero_mean[:, p, s] = x_filtered[:, p, s]

    return {
        "t": t,
        "Tt": Tt,
        "x_ref": x_zero_mean,
    }


# =============================================================================
# FOURIER / ta-tb SHARED HELPERS
# =============================================================================

def _compute_ta_tb_from_reference(cfg, params, x_ref, Tt, N):
    """
    Compute ta/tb from the common source reference.
    """

    fourier_core = FourierCoefficientCore(
        T=params.tau,
        harmonics=N,
        sensor_nodes=params.S
    )

    an, bn, _ = fourier_core.calc_an_bn_dft(
        x_ref,
        Tt,
        normalize=cfg["signal"]["normalize_dft"],
        norm=cfg["signal"]["normalization_target"]
    )

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=1,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    ta, tb = phase_core.calc_ta_tb(an, bn)

    return np.real(ta), np.real(tb)


# =============================================================================
# BENCHMARK + FDMA
# =============================================================================

def _run_benchmark_fdma_stack(cfg, params, x_ref, t, periodic_replicas=10):
    """
    Fair Benchmark + FDMA stack.
    """

    if not cfg["mode"].get("run_benchmark", True):
        return np.nan

    benchmark_cfg = cfg.get("benchmark", {})
    fdma_cfg = cfg.get("fdma", {})

    sampling_rate = benchmark_cfg.get("sampling_rate", params.W)
    effective_rate_factor = benchmark_cfg.get("effective_rate_factor", 1.0)

    fdma_core = FDMACore(
        S=params.S,
        B_total=params.B,
        P_per_sensor=params.P,
        tau=params.tau,
        bandwidth_allocation=params.bandwidth_allocation,
        normalize_sensor_power=fdma_cfg.get("normalize_sensor_power", False),
        return_nonorthogonal_sum_preview=False,
        frequency_axis_centered_at_zero=fdma_cfg.get(
            "frequency_axis_centered_at_zero",
            True
        ),
    )

    M_vec = compute_benchmark_M_per_sensor(
        S=params.S,
        tau=params.tau,
        B=params.B,
        SNR=params.SNR,
        sampling_rate=effective_rate_factor * sampling_rate,
        bandwidth_allocation=params.bandwidth_allocation,
        force_power_of_two=cfg.get("quantization", {}).get("force_power_of_two", False),
        rounding_mode=cfg.get("quantization", {}).get("rounding_mode", "floor"),
    )

    _ = fdma_core.describe_budget()

    _, _, S = x_ref.shape
    mse_sum = 0.0
    count = 0

    for s in range(S):
        x_sensor = x_ref[:, 0, s]

        # 1) sample
        t_samp = np.arange(0.0, params.tau, 1.0 / sampling_rate)
        x_samp = np.interp(t_samp, t, x_sensor)

        # 2) quantize with M bins
        xq = _uniform_quantize_signal_samples(x_samp, M=int(M_vec[s]))

        # 3) reconstruct with agreed sinc truncation rule
        x_hat = _reconstruct_continuous_from_samples_sinc(
            t_ref=t,
            t_samp=t_samp,
            x_samp_hat=xq,
            tau=params.tau,
            Tc=1.0 / sampling_rate,
            periodic_replicas=periodic_replicas
        )

        mse_sum += np.mean((x_sensor - x_hat) ** 2)
        count += 1

    return mse_sum / count


# =============================================================================
# PPM + FDMA
# =============================================================================

def _run_ppm_fdma_stack(cfg, params, x_ref, t, periodic_replicas=10, rng=None):
    """
    Fair PPM + FDMA stack.
    """

    if not cfg["mode"].get("run_ppm", True):
        return np.nan

    if PPMCore is None:
        raise ImportError(
            "PPMCore could not be imported from sfc.core.modulation.ppm. "
            "Stabilize ppm.py / pulse_shaping.py before running the PPM + FDMA branch."
        )

    if rng is None:
        rng = np.random.default_rng()

    ppm_cfg = cfg.get("ppm", {})
    fdma_cfg = cfg.get("fdma", {})

    fs_msg = ppm_cfg.get("fs_msg", params.W)
    pulse_width = ppm_cfg.get("pulse_width", 0.1 / fs_msg)
    pulse_type = ppm_cfg.get("pulse_type", "raised_cosine")
    rolloff = ppm_cfg.get("rolloff", 0.99)
    span = ppm_cfg.get("span", 12)
    eps_margin = ppm_cfg.get("eps_margin", 1e-3)

    fdma_core = FDMACore(
        S=params.S,
        B_total=params.B,
        P_per_sensor=params.P,
        tau=params.tau,
        bandwidth_allocation=params.bandwidth_allocation,
        normalize_sensor_power=fdma_cfg.get("normalize_sensor_power", True),
        return_nonorthogonal_sum_preview=False,
        frequency_axis_centered_at_zero=fdma_cfg.get(
            "frequency_axis_centered_at_zero",
            True
        ),
    )

    ppm_core = PPMCore(
        fc=fs_msg,
        pulse_width=pulse_width,
        rec_pulse=ppm_cfg.get("rec_pulse", 0.0),
        pulse_type=pulse_type,
        rolloff=rolloff,
        span=span,
        eps_margin=eps_margin,
        interp_mode=ppm_cfg.get("interp_mode", "sinc"),
        periodic_replicas=periodic_replicas,
        clip_recovered_to_unit_interval=ppm_cfg.get(
            "clip_recovered_to_unit_interval",
            True
        ),
    )

    # 1) modulate one waveform per sensor
    mod_result = ppm_core.modulate(x_ref, t)
    tx_waveform = mod_result.tx_waveform  # shape: (time, periods, sensors)

    # 2) apply FDMA fairness/power accounting
    mux_result = fdma_core.multiplex(
        MACInput(
            tx_waveform_per_sensor=tx_waveform,
            t=t,
            metadata={"source": "ppm"}
        ),
        normalize_sensor_power=fdma_cfg.get("normalize_sensor_power", True),
    )

    tx_alloc = mux_result.per_sensor_allocated_waveform
    if tx_alloc is None:
        raise RuntimeError("FDMACore returned no per_sensor_allocated_waveform.")

    # 3) add per-sensor AWGN consistent with the FDMA slice
    rx_alloc = _add_awgn_per_sensor_fdma(
        x=tx_alloc,
        P_per_sensor=params.P,
        B_per_sensor=fdma_core.get_bandwidth_per_sensor(),
        N0=cfg["system"]["N0"],
        rng=rng
    )

    # 4) ideal FDMA demultiplex
    demux_result = fdma_core.demultiplex(
        received_signal=rx_alloc,
        multiplex_result=mux_result
    )

    rx_per_sensor = demux_result.recovered_waveform_per_sensor
    if rx_per_sensor is None:
        raise RuntimeError("FDMACore demultiplex returned no recovered waveform.")

    # 5) demodulate PPM
    demod_result = ppm_core.demodulate(
        y=rx_per_sensor,
        t=t,
        modulation_result=mod_result,
        reconstruct_continuous=True
    )

    x_hat = demod_result.recovered_continuous
    if x_hat is None:
        raise RuntimeError(
            "PPMCore demodulation returned no continuous reconstruction."
        )

    # 6) compare on the same tensor domain
    return float(np.mean((x_ref - x_hat) ** 2))


# =============================================================================
# RBCP_time
# =============================================================================

def _run_rbcp_time_stack(cfg, params, ta, tb, x_ref, t, N):
    """
    Error-free time-model boundary:
        signal -> ta/tb -> events -> ta/tb -> signal
    """

    if not cfg["mode"].get("run_rbcp_time", True):
        return np.nan

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=1,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    events = phase_core.ta_tb_to_events(ta, tb)
    ta_rec, tb_rec = phase_core.event_to_ta_tb(events)

    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    w0 = 2.0 * np.pi / params.tau

    mse_sum = 0.0
    count = 0

    for s in range(params.S):
        x_rec = recover_signal(
            ta_rec[0, :, s],
            tb_rec[0, :, s],
            t,
            w0
        )

        mse_sum += np.mean((x_ref[:, 0, s] - x_rec) ** 2)
        count += 1

    return mse_sum / count


# =============================================================================
# NATIVE SFC (WITHOUT SED)
# =============================================================================

def _run_sfc_stack(cfg, params, ta, tb, x_ref, t, N, sfc_channel=None):
    """
    Native SFC stack without semantic error detection.
    """

    if not cfg["mode"].get("run_sfc", True):
        return np.nan

    if sfc_channel is None:
        return np.nan

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=1,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    events = phase_core.ta_tb_to_events(ta, tb)

    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)
    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    w0 = 2.0 * np.pi / params.tau

    mse_sum = 0.0
    count = 0

    for s in range(params.S):
        x_rec = recover_signal(
            ta_rec[0, :, s],
            tb_rec[0, :, s],
            t,
            w0
        )

        mse_sum += np.mean((x_ref[:, 0, s] - x_rec) ** 2)
        count += 1

    return mse_sum / count


# =============================================================================
# NATIVE SFC + SED STACK
# =============================================================================

def _run_sfc_sed_stack(cfg, params, ta, tb, x_ref, t, N, sfc_channel=None):
    """
    Native SFC stack with semantic-based error detection.

    IMPORTANT
    ---------
    Invalid periods are discarded and do NOT contribute to the MSE.
    Since this pipeline uses 1 period per trial:
    - valid_period_fraction_sfc_sed is either 1.0 or 0.0
    - mse_sfc_sed is nan if the trial is invalid
    """

    if not cfg["mode"].get("run_sfc", True):
        return np.nan, np.nan, False

    if sfc_channel is None:
        return np.nan, np.nan, False

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=1,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    events = phase_core.ta_tb_to_events(ta, tb)

    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    # One period per trial -> safest segmentation is the actual row count
    period_slots = events_est.shape[0]

    sed_cfg = cfg.get("sed", {})
    sensor_x_event = _build_sensor_x_event(params.S, N)

    sed_result = detect_semantic_errors(
        events_est=events_est,
        period_slots=period_slots,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=sed_cfg.get("discard_invalid_periods", True)
    )

    corrected_events_est, period_valid_mask = _extract_sed_outputs(sed_result)

    valid_fraction = float(np.mean(period_valid_mask))
    is_valid_trial = bool(period_valid_mask[0])

    if not is_valid_trial:
        return np.nan, valid_fraction, False

    ta_rec, tb_rec = phase_core.event_to_ta_tb(corrected_events_est)
    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    w0 = 2.0 * np.pi / params.tau

    mse_sum = 0.0
    count = 0

    for s in range(params.S):
        x_rec = recover_signal(
            ta_rec[0, :, s],
            tb_rec[0, :, s],
            t,
            w0
        )

        mse_sum += np.mean((x_ref[:, 0, s] - x_rec) ** 2)
        count += 1

    if count == 0:
        return np.nan, valid_fraction, False

    return mse_sum / count, valid_fraction, True


def _extract_sed_outputs(sed_result):
    """
    Extract (corrected_events_est, period_valid_mask) from the SED result.
    Supports dict-like or dataclass-like objects.
    """

    if isinstance(sed_result, dict):
        corrected_events_est = sed_result["corrected_events_est"]
        period_valid_mask = np.asarray(sed_result["period_valid_mask"], dtype=bool)
        return corrected_events_est, period_valid_mask

    corrected_events_est = getattr(sed_result, "corrected_events_est")
    period_valid_mask = np.asarray(getattr(sed_result, "period_valid_mask"), dtype=bool)

    return corrected_events_est, period_valid_mask


# =============================================================================
# NATIVE SFC CHANNEL BUILDER
# =============================================================================

def _build_sfc_channel_for_B(cfg_B, N, S):
    """
    Build one native SFCChannel instance for the current B-point.
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
    """

    num_event_ids = 2 * N * S
    sensor_x_event = np.zeros((S, num_event_ids))

    for s in range(S):
        start = 2 * s * N
        stop = 2 * (s + 1) * N
        sensor_x_event[s, start:stop] = 1.0

    return sensor_x_event


# =============================================================================
# SAMPLE-BASED FAIR HELPERS
# =============================================================================

def _uniform_quantize_signal_samples(x_samp, M):
    """
    Uniform mid-rise scalar quantizer with M bins.
    """
    x_samp = np.asarray(x_samp, dtype=float)

    if M < 1:
        raise ValueError("M must be >= 1")

    xmin = np.min(x_samp)
    xmax = np.max(x_samp)

    if np.isclose(xmax, xmin):
        return np.full_like(x_samp, xmin)

    delta = (xmax - xmin) / M
    idx = np.floor((x_samp - xmin) / delta)
    idx = np.clip(idx, 0, M - 1)

    return xmin + (idx + 0.5) * delta


def _reconstruct_continuous_from_samples_sinc(
    t_ref,
    t_samp,
    x_samp_hat,
    tau,
    Tc,
    periodic_replicas=10
):
    """
    Reconstruct a continuous-time waveform from samples using truncated periodic
    sinc interpolation.

    Truncation convention
    ---------------------
        periodic_replicas = 10

    means:
    - replicas from -10*tau to +10*tau
    - total of 21 periodic blocks contributing to the sum
    """

    t_ref = np.asarray(t_ref, dtype=float)
    t_samp = np.asarray(t_samp, dtype=float)
    x_samp_hat = np.asarray(x_samp_hat, dtype=float)

    y = np.zeros_like(t_ref)

    replica_ids = np.arange(-periodic_replicas, periodic_replicas + 1, dtype=int)

    for m in replica_ids:
        shifted_times = t_samp + m * tau
        for k in range(len(x_samp_hat)):
            y += x_samp_hat[k] * np.sinc((t_ref - shifted_times[k]) / Tc)

    return y


def _add_awgn_per_sensor_fdma(x, P_per_sensor, B_per_sensor, N0, rng):
    """
    Add per-sensor AWGN under the FDMA interpretation.

    Interpretation
    --------------
    For each sensor s:
        SNR_s = P_per_sensor / (B_per_sensor[s] * N0)
    """

    x = np.asarray(x, dtype=float)
    y = np.array(x, copy=True)

    _, _, S = x.shape

    for s in range(S):
        Bs = float(B_per_sensor[s])

        if Bs <= 0:
            continue

        snr_s = P_per_sensor / (Bs * N0)

        ps = np.mean(x[:, :, s] ** 2)
        if np.isclose(ps, 0.0):
            continue

        pn = ps / snr_s
        noise = rng.normal(0.0, np.sqrt(pn), size=x[:, :, s].shape)

        y[:, :, s] = x[:, :, s] + noise

    return y


# =============================================================================
# SAVE UTILITY
# =============================================================================

def save_dat_file(df, path, delimiter="\t"):
    """
    Save the comparison dataset to a .dat-compatible tabular file.
    """

    df.to_csv(
        path,
        sep=delimiter,
        index=False,
        float_format="%.8e"
    )
