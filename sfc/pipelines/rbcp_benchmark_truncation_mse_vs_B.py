"""
sfc/pipelines/rbcp_benchmark_truncation_mse_vs_B.py

Diagnostic pipeline comparing truncation policies for Benchmark and RbCP.

This figure evaluates, as a function of B:

- Benchmark with free integer M
- Benchmark with power-of-two M
- RbCP with free integer M_RbCP
- RbCP with power-of-two M_RbCP

Important simplifications
-------------------------
1. No channel:
   - no SFC
   - no RbCP_time
   - no AWGN
   - no semantic error detection

2. Figure-5-style power model:
   Keep per-sensor SNR_dB fixed and N0 fixed, derive P(B) from:
       SNR_s = P / (B_s * N0)
   Therefore:
       P(B) = SNR_s * B_s * N0

   For uniform bandwidth allocation:
       B_s = B / S

   With a scalar P shared by all sensors, fixed per-sensor SNR requires
   uniform bandwidth allocation. For nonuniform allocation, use fixed P,N0
   or allow sensor-dependent powers P_s.

3. Benchmark:
   Treated analytically in absolute MSE:
       MSE = (peak_to_peak^2) / (12 * M^2)

4. RbCP:
   Evaluated by Monte Carlo signal generation and reconstruction using:
       ta/tb -> quantization -> signal reconstruction

5. Truncation policies are compared explicitly using the YAML block:
   comparison:
     benchmark:
       free_integer:
         force_power_of_two: false
         rounding_mode: "floor"
       power_of_two:
         force_power_of_two: true
         rounding_mode: "floor"
     rbcp:
       free_integer:
         force_power_of_two: false
         rounding_mode: "floor"
       power_of_two:
         force_power_of_two: true
         rounding_mode: "floor"
"""

import copy
import numpy as np
import pandas as pd

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.system_parameters import (
    build_derived_system_parameters,
)

from sfc.core.theory import (
    compute_snr_linear,
    compute_benchmark_M_per_sensor,
    compute_M_rbcp,
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_rbcp_benchmark_truncation_mse_vs_B_data(cfg):
    """
    Generate the truncation-comparison dataset.

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
        - M_benchmark_free
        - M_benchmark_pow2
        - M_rbcp_free
        - M_rbcp_pow2
        - mse_benchmark_free
        - mse_benchmark_pow2
        - mse_rbcp_free
        - mse_rbcp_pow2
        - num_trials
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    print("[INFO] Starting rbcp_benchmark_truncation_mse_vs_B data generation")
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

        # ---------------------------------------------------------------------
        # Figure-5-style power model:
        # keep SNR fixed and N0 fixed, derive P(B)
        # ---------------------------------------------------------------------
        cfg_B["system"]["B"] = float(B)
        cfg_B["system"]["P"] = _derive_power_from_fixed_sensor_snr_and_n0(cfg_B)

        params = build_derived_system_parameters(cfg_B)

        N = cfg_B["signal"].get("N_override", params.N)
        n_trials = cfg_B["monte_carlo"]["interactions"]

        # ---------------------------------------------------------------------
        # Comparison policies
        # ---------------------------------------------------------------------
        bench_free_cfg = cfg_B["comparison"]["benchmark"]["free_integer"]
        bench_pow2_cfg = cfg_B["comparison"]["benchmark"]["power_of_two"]

        rbcp_free_cfg = cfg_B["comparison"]["rbcp"]["free_integer"]
        rbcp_pow2_cfg = cfg_B["comparison"]["rbcp"]["power_of_two"]

        # ---------------------------------------------------------------------
        # Benchmark M per sensor (from core)
        # ---------------------------------------------------------------------
        benchmark_cfg = cfg_B.get("benchmark", {})
        sampling_rate = benchmark_cfg.get("sampling_rate", params.W)
        effective_rate_factor = benchmark_cfg.get("effective_rate_factor", 1.0)

        # ---------------------------------------------------------------------
        # Benchmark M per sensor (CORRECTED: use per-sensor SNR)
        # ---------------------------------------------------------------------
        M_benchmark_free_vec = compute_benchmark_M_per_sensor(
            S=params.S,
            tau=params.tau,
            B=params.B,
            P=params.P,
            N0=cfg_B["system"]["N0"],
            sampling_rate=effective_rate_factor * sampling_rate,
            bandwidth_allocation=params.bandwidth_allocation,
            force_power_of_two=bench_free_cfg["force_power_of_two"],
            rounding_mode=bench_free_cfg["rounding_mode"],
        )

        M_benchmark_pow2_vec = compute_benchmark_M_per_sensor(
            S=params.S,
            tau=params.tau,
            B=params.B,
            P=params.P,
            N0=cfg_B["system"]["N0"],
            sampling_rate=effective_rate_factor * sampling_rate,
            bandwidth_allocation=params.bandwidth_allocation,
            force_power_of_two=bench_pow2_cfg["force_power_of_two"],
            rounding_mode=bench_pow2_cfg["rounding_mode"],
        )

        # Store one scalar diagnostic value for each policy.
        # If bandwidth allocation is equal, min = mean = max anyway.
        M_benchmark_free = int(np.min(M_benchmark_free_vec))
        M_benchmark_pow2 = int(np.min(M_benchmark_pow2_vec))

        # ---------------------------------------------------------------------
        # RbCP M (from core)
        # ---------------------------------------------------------------------
        M_rbcp_free = compute_M_rbcp(
            S=params.S,
            W=params.W,
            tau=params.tau,
            B=params.B,
            P=params.P,
            N0=cfg_B["system"]["N0"],
            bandwidth_allocation=params.bandwidth_allocation,
            force_power_of_two=rbcp_free_cfg["force_power_of_two"],
            rounding_mode=rbcp_free_cfg["rounding_mode"],
        )

        M_rbcp_pow2 = compute_M_rbcp(
            S=params.S,
            W=params.W,
            tau=params.tau,
            B=params.B,
            P=params.P,
            N0=cfg_B["system"]["N0"],
            bandwidth_allocation=params.bandwidth_allocation,
            force_power_of_two=rbcp_pow2_cfg["force_power_of_two"],
            rounding_mode=rbcp_pow2_cfg["rounding_mode"],
        )

        print("\n[INFO] ------------------------------------------------------------")
        print(f"[INFO] B = {B:.1f} Hz")
        print(f"[INFO] P(B) = {cfg_B['system']['P']:.6e}")
        print(
            f"[INFO] S = {params.S} | R = {params.R} | L = {params.L} | "
            f"tau = {params.tau:.3f} s | W = {params.W:.3f} Hz | N = {N}"
        )
        print(f"[INFO] SNR_dB = {params.SNR_dB:.1f} | SNR = {params.SNR:.4e}")
        print(f"[INFO] N0 = {cfg_B['system']['N0']:.6e}")
        print(f"[INFO] bandwidth_allocation = {params.bandwidth_allocation}")
        print(f"[INFO] B_per_sensor = {params.B_per_sensor}")

        print(f"[INFO] M_benchmark_free_vec = {M_benchmark_free_vec}")
        print(f"[INFO] M_benchmark_pow2_vec = {M_benchmark_pow2_vec}")
        print(f"[INFO] M_rbcp_free = {M_rbcp_free}")
        print(f"[INFO] M_rbcp_pow2 = {M_rbcp_pow2}")

        # ---------------------------------------------------------------------
        # Benchmark MSEs (analytical, once per B)
        # ---------------------------------------------------------------------
        mse_benchmark_free = _run_benchmark_branch(
            cfg=cfg_B,
            M_per_sensor=M_benchmark_free_vec
        )

        mse_benchmark_pow2 = _run_benchmark_branch(
            cfg=cfg_B,
            M_per_sensor=M_benchmark_pow2_vec
        )

        # ---------------------------------------------------------------------
        # Monte Carlo for RbCP only
        # ---------------------------------------------------------------------
        mse_rbcp_free_sum = 0.0
        mse_rbcp_pow2_sum = 0.0

        for i in range(n_trials):
            trial = _run_one_trial(
                cfg=cfg_B,
                rng=rng,
                N=N,
                M_rbcp_free=M_rbcp_free,
                M_rbcp_pow2=M_rbcp_pow2
            )

            mse_rbcp_free_sum += trial["mse_rbcp_free"]
            mse_rbcp_pow2_sum += trial["mse_rbcp_pow2"]

            if (i + 1) % max(1, n_trials // 5) == 0:
                print(f"[INFO] Trial progress: {i + 1}/{n_trials}")

        mse_rbcp_free = mse_rbcp_free_sum / n_trials
        mse_rbcp_pow2 = mse_rbcp_pow2_sum / n_trials

        print(f"[INFO] mse_benchmark_free = {mse_benchmark_free:.6e}")
        print(f"[INFO] mse_benchmark_pow2 = {mse_benchmark_pow2:.6e}")
        print(f"[INFO] mse_rbcp_free = {mse_rbcp_free:.6e}")
        print(f"[INFO] mse_rbcp_pow2 = {mse_rbcp_pow2:.6e}")

        results.append({
            "B": B,
            "P_derived": cfg_B["system"]["P"],
            "M_benchmark_free": M_benchmark_free,
            "M_benchmark_pow2": M_benchmark_pow2,
            "M_rbcp_free": M_rbcp_free,
            "M_rbcp_pow2": M_rbcp_pow2,
            "mse_benchmark_free": mse_benchmark_free,
            "mse_benchmark_pow2": mse_benchmark_pow2,
            "mse_rbcp_free": mse_rbcp_free,
            "mse_rbcp_pow2": mse_rbcp_pow2,
            "num_trials": n_trials,
        })

    return pd.DataFrame(results)


# =============================================================================
# FIGURE-5-STYLE POWER MODEL
# =============================================================================

def _derive_power_from_fixed_sensor_snr_and_n0(cfg):
    """
    Derive P(B) from fixed per-sensor SNR and fixed N0.

    The SNR configured in the YAML is interpreted as the SNR of each sensor
    channel:

        SNR_s = P / (B_s * N0)

    Therefore:

        P = SNR_s * B_s * N0

    Notes
    -----
    This helper assumes a scalar P shared by all sensors.

    If bandwidth allocation is uniform, all sensors have the same B_s and the
    same SNR_s.

    If bandwidth allocation is nonuniform, a single scalar P cannot keep the
    same SNR for all sensors. In that case this helper raises an error.
    """

    SNR_s = compute_snr_linear(cfg["system"]["SNR_dB"])
    B = cfg["system"]["B"]
    N0 = cfg["system"]["N0"]
    S = cfg["system"]["S"]

    allocation = cfg["system"].get("bandwidth_allocation", None)

    if allocation is None:
        alpha = 1.0 / S
        B_sensor = alpha * B
        return SNR_s * B_sensor * N0

    allocation = np.asarray(allocation, dtype=float).reshape(-1)

    if len(allocation) != S:
        raise ValueError(
            f"bandwidth_allocation length mismatch: len={len(allocation)} but S={S}"
        )

    if np.any(allocation < 0):
        raise ValueError("bandwidth_allocation must be nonnegative")

    if not np.isclose(np.sum(allocation), 1.0):
        raise ValueError(
            f"bandwidth_allocation must sum to 1. Current sum={np.sum(allocation)}"
        )

    if not np.allclose(allocation, np.ones(S) / S):
        raise ValueError(
            "fixed per-sensor SNR with a scalar P requires uniform bandwidth allocation. "
            "For nonuniform allocation, either allow sensor-dependent P_s or use fixed P,N0."
        )

    B_sensor = B * allocation[0]
    return SNR_s * B_sensor * N0


# =============================================================================
# ONE MONTE CARLO TRIAL
# =============================================================================

def _run_one_trial(cfg, rng, N, M_rbcp_free, M_rbcp_pow2):
    """
    Run one Monte Carlo trial and return the two RbCP MSE values.

    Returns
    -------
    dict
        {
            "mse_rbcp_free": ...,
            "mse_rbcp_pow2": ...
        }
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
    # 7. RbCP free
    # -------------------------------------------------------------------------
    mse_rbcp_free = _run_rbcp_branch(
        ta=ta,
        tb=tb,
        x_ref=x_zero_mean,
        t=t,
        tau=tau,
        M_rbcp=M_rbcp_free
    )

    # -------------------------------------------------------------------------
    # 8. RbCP power-of-two
    # -------------------------------------------------------------------------
    mse_rbcp_pow2 = _run_rbcp_branch(
        ta=ta,
        tb=tb,
        x_ref=x_zero_mean,
        t=t,
        tau=tau,
        M_rbcp=M_rbcp_pow2
    )

    return {
        "mse_rbcp_free": mse_rbcp_free,
        "mse_rbcp_pow2": mse_rbcp_pow2,
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

def _compute_fourier_coefficients(
    x_zero_mean,
    tau,
    N,
    S,
    Tt,
    normalize_dft,
    normalization_target
):
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

def _run_benchmark_branch(cfg, M_per_sensor):
    """
    Analytical Benchmark branch in absolute MSE:

        MSE_abs = (peak_to_peak^2) / (12 * M^2)

    Returns the average over sensors.
    """

    if not cfg["mode"].get("run_benchmark", False):
        return np.nan

    peak_to_peak = cfg["signal"]["peak_to_peak"]

    mse_sum = 0.0
    for M in M_per_sensor:
        mse_sum += (peak_to_peak ** 2) / (12.0 * (float(M) ** 2))

    return mse_sum / len(M_per_sensor)


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
