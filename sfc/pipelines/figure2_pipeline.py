"""
sfc/pipelines/figure2_pipeline.py

Pipeline for reproducing Figure 2 of the manuscript.

This module:
- computes analytical curves (upper bound, MSE*)
- runs Monte Carlo for RbCP curves
- computes Benchmark curve
- returns full dataset ready for plotting

Differences vs Figure 1:
- sweep over M_RbCP instead of N
- multiple fixed N values
- includes Benchmark curve
"""

import numpy as np
import pandas as pd

from sfc.core.theory import (
    compute_q,
    rbcp_mse_upper_bound,
    rbcp_mse_star,
    compute_M_from_M_rbcp,
)
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import calc_ta_tb
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.filters import filter_periodic


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_figure2_data(cfg):
    """
    Generate dataset for Figure 2.

    Returns
    -------
    pandas.DataFrame
        Final dataset with columns:
        - M_RbCP
        - N
        - Q
        - upper_bound
        - mse_star
        - mse_mc_mean
        - mse_mc_std
        - num_trials
        - benchmark_M
        - benchmark_mse
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    m_cfg = cfg["sweep"]["m_rbcp"]
    m_values = np.arange(m_cfg["start"], m_cfg["stop"], m_cfg["step"])

    n_values = cfg["sweep"]["n_values"]

    results = []

    for N in n_values:
        print("[INFO] Processing N =", N)

        for M in m_values:

            Q = compute_q(M)

            # ------------------------------------------------------------
            # THEORY
            # ------------------------------------------------------------
            mse_up = rbcp_mse_upper_bound(N, Q)
            mse_st = rbcp_mse_star(N, Q)

            # ------------------------------------------------------------
            # MONTE CARLO
            # ------------------------------------------------------------
            mc_results = _run_mc_block(N, M, cfg, rng)

            mse_mc = mc_results["mean"]

            # ------------------------------------------------------------
            # BENCHMARK
            # ------------------------------------------------------------
            bench_M, bench_mse = _compute_benchmark(M, cfg)

            print(
                f"[INFO] N={N:2d}  M={M:2d}  "
                f"MC={mse_mc:.6e}  "
                f"MSE*={mse_st:.6e}  UB={mse_up:.6e}"
            )

            results.append({
                "M_RbCP": M,
                "N": N,
                "Q": Q,
                "upper_bound": mse_up,
                "mse_star": mse_st,
                "mse_mc_mean": mc_results["mean"],
                "mse_mc_std": mc_results["std"],
                "num_trials": mc_results["num_trials"],
                "benchmark_M": bench_M,
                "benchmark_mse": bench_mse,
            })

    return pd.DataFrame(results)


# =============================================================================
# MONTE CARLO
# =============================================================================

def _run_mc_block(N, M, cfg, rng):
    """
    Run Monte Carlo block for a fixed pair (N, M_RbCP).
    """

    n_trials = cfg["monte_carlo"]["interactions"]
    mse_values = np.zeros(n_trials)

    for i in range(n_trials):
        mse_values[i] = _run_single_trial(N, M, cfg, rng)

    return {
        "mean": np.mean(mse_values),
        "std": np.std(mse_values),
        "num_trials": n_trials,
    }


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def _run_single_trial(N, M, cfg, rng):
    """
    Single Monte Carlo trial for Figure 2.
    """

    T = _get_signal_period(cfg)
    Tt = cfg["signal"]["Tt"]
    w0 = 2 * np.pi / T
    n_vec = np.arange(1, N + 1)

    # ------------------------------------------------------------
    # SIGNAL
    # ------------------------------------------------------------
    x_raw, t = _generate_signal(cfg, rng)

    W = 2 * N / T
    x_filtered = filter_periodic(x_raw, W, Tt, T)

    # ------------------------------------------------------------
    # PEAK-TO-PEAK CONTROL
    # ------------------------------------------------------------
    p2p_target = cfg["signal"]["peak_to_peak"]
    current_p2p = np.max(x_filtered) - np.min(x_filtered)

    if p2p_target != 0 and current_p2p != 0:
        x_filtered = x_filtered * (p2p_target / current_p2p)

    # ------------------------------------------------------------
    # DC REMOVAL
    # ------------------------------------------------------------
    dc_enabled = cfg.get("dc", {}).get("enabled", False)

    if not dc_enabled:
        dc = Tt * np.sum(x_filtered) / T
        x_rbcp = x_filtered - dc
    else:
        x_rbcp = x_filtered

    # ------------------------------------------------------------
    # FOURIER
    # ------------------------------------------------------------
    fourier_core = FourierCoefficientCore(
        T=T,
        harmonics=N,
        sensor_nodes=1,
        dft_signal_periods=1
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_rbcp,
        Tt,
        normalize=cfg["signal"]["normalize_dft"],
        norm=cfg["signal"]["normalization_target"]
    )

    x_used = x_used[:, 0, 0]

    # ------------------------------------------------------------
    # PHASE (LOG FORMULATION)
    # ------------------------------------------------------------
    ta, tb = calc_ta_tb(an[0, :, 0], bn[0, :, 0], n_vec, w0)
    ta = np.real(ta)
    tb = np.real(tb)

    # ------------------------------------------------------------
    # QUANTIZATION
    # ------------------------------------------------------------
    ta_q, tb_q = quantize_ta_tb(ta, tb, w0, M)

    # ------------------------------------------------------------
    # RECONSTRUCTION
    # ------------------------------------------------------------
    x_hat = recover_signal(ta_q, tb_q, t, w0)

    # ------------------------------------------------------------
    # MSE
    # ------------------------------------------------------------
    mse = np.mean((x_used - x_hat) ** 2)

    return mse


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def _get_signal_period(cfg):
    """
    Get the signal period from the configuration.

    Accepts either:
    - cfg["signal"]["T"]   (legacy convention)
    - cfg["signal"]["tau"] (newer project-wide convention)
    """

    if "T" in cfg["signal"]:
        return cfg["signal"]["T"]

    if "tau" in cfg["signal"]:
        return cfg["signal"]["tau"]

    raise KeyError("Expected cfg['signal']['T'] or cfg['signal']['tau'].")


def _generate_signal(cfg, rng):
    """
    Generate a random signal according to the YAML configuration.
    """

    T = _get_signal_period(cfg)
    Tt = cfg["signal"]["Tt"]

    t = np.arange(0, T, Tt)

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x = rng.uniform(-1, 1, len(t))
    elif dist == "gaussian":
        x = rng.normal(0, 1, len(t))
    else:
        raise ValueError("Invalid distribution")

    return x, t


# =============================================================================
# BENCHMARK
# =============================================================================

def _compute_benchmark(M_rbcp, cfg):
    """
    Compute Benchmark curve for Figure 2.

    According to the manuscript discussion for Figure 2, the Benchmark curve
    should be obtained by mapping M_RbCP to Benchmark M using Eq. (17), and
    then evaluating:

        MSE_x,y = 1 / (12 * M^2)

    Note
    ----
    This is the normalized Benchmark expression used in the manuscript.
    """

    if not cfg["benchmark"]["enabled"]:
        return np.nan, np.nan

    T = _get_signal_period(cfg)

    # Benchmark settings for the manuscript comparison curve
    benchmark_cfg = cfg["benchmark"]
    W_benchmark = benchmark_cfg["W"]

    # Eq. (17): map M_RbCP -> M
    M = compute_M_from_M_rbcp(
        M_rbcp=M_rbcp,
        W=W_benchmark,
        tau=T
    )

    p2p = cfg["benchmark"]["peak_to_peak"]
    mse = (p2p ** 2) / (12.0 * (M ** 2))

    return M, mse


# =============================================================================
# SAVE
# =============================================================================

def save_dat_file(df, path, delimiter="\t"):
    """
    Save the dataset to a .dat-compatible tabular file.
    """

    df.to_csv(
        path,
        sep=delimiter,
        index=False,
        float_format="%.8e"
    )
