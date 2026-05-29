"""
sfc/pipelines/figure1_pipeline.py

Pipeline for reproducing Figure 1 of the manuscript.

This module:
- computes the analytical curves used in Figure 1;
- runs Monte Carlo corroboration using the trusted RbCP pipeline;
- returns a tabular dataset ready to be saved as .dat and plotted later.

IMPORTANT
---------
This pipeline uses the RbCP representation/reconstruction chain only.

It does NOT use:
- the SFC channel;
- transmission maps;
- channel noise;
- multiuser MAC logic.

The Monte Carlo flow implemented here is:

1. generate a random signal (uniform or gaussian);
2. band-limit the signal with the trusted periodic filter;
3. adjust the peak-to-peak of the filtered signal;
4. optionally remove the DC component (for Figure 1: dc.enabled = false);
5. compute Fourier coefficients with trusted normalization logic;
6. compute ta/tb using the trusted complex-logarithm formulation;
7. quantize ta/tb with M_RbCP bins;
8. reconstruct the signal from quantized ta/tb;
9. compute the empirical MSE against the signal effectively used.

All heavy mathematical logic is delegated to trusted core modules.
"""

import numpy as np
import pandas as pd

from sfc.core.theory import (
    compute_q,
    rbcp_mse_upper_bound,
    rbcp_mse_star,
)
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import calc_ta_tb
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.filters import filter_periodic


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_figure1_data(cfg):
    """
    Generate the full dataset required to reproduce Figure 1.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    Returns
    -------
    pandas.DataFrame
        Final dataset containing one row per (N, M_RbCP) pair with:
        - N
        - M_RbCP
        - Q
        - upper_bound
        - mse_star
        - mse_mc_mean
        - mse_mc_std
        - num_trials
    """

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])

    m_values = cfg["sweep"]["m_rbcp_values"]

    n_cfg = cfg["sweep"]["n"]
    n_values = np.arange(n_cfg["start"], n_cfg["stop"], n_cfg["step"])

    results = []

    for M_rbcp in m_values:
        print("[INFO] Processing M_RbCP =", M_rbcp)

        Q = compute_q(M_rbcp)

        for N in n_values:
            # ------------------------------------------------------------
            # THEORETICAL VALUES
            # ------------------------------------------------------------
            mse_up = rbcp_mse_upper_bound(N, Q)
            mse_st = rbcp_mse_star(N, Q)

            # ------------------------------------------------------------
            # MONTE CARLO BLOCK
            # ------------------------------------------------------------
            mc_results = _run_mc_block(
                N=N,
                M=M_rbcp,
                cfg=cfg,
                rng=rng
            )

            mse_mc = mc_results["mean"]

            print(
                f"[INFO] N={N:2d}  M={M_rbcp:2d}  "
                f"MSE_Monte_Carlo={mse_mc:.6e}  "
                f"MSE*={mse_st:.6e}  UB={mse_up:.6e}"
            )

            results.append({
                "N": N,
                "M_RbCP": M_rbcp,
                "Q": Q,
                "upper_bound": mse_up,
                "mse_star": mse_st,
                "mse_mc_mean": mc_results["mean"],
                "mse_mc_std": mc_results["std"],
                "num_trials": mc_results["num_trials"],
            })

    return pd.DataFrame(results)


# =============================================================================
# MONTE CARLO BLOCK
# =============================================================================

def _run_mc_block(N, M, cfg, rng):
    """
    Run the Monte Carlo block for a fixed pair (N, M_RbCP).

    Parameters
    ----------
    N : int
        Number of harmonics used in the figure.

    M : int
        Number of quantization bins M_RbCP.

    cfg : dict
        Parsed YAML configuration.

    rng : numpy.random.Generator
        Random-number generator.

    Returns
    -------
    dict
        Dictionary with:
        - mean
        - std
        - num_trials
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
# SINGLE MONTE CARLO TRIAL
# =============================================================================

def _run_single_trial(N, M, cfg, rng):
    """
    Single Monte Carlo trial for Figure 1 using the trusted RbCP pipeline.

    Steps
    -----
    1. generate a random signal;
    2. band-limit the signal;
    3. adjust the peak-to-peak of the filtered signal;
    4. if dc.enabled is false, remove the DC component explicitly;
    5. compute Fourier coefficients with trusted normalization logic;
    6. compute ta/tb using the trusted complex-logarithm formulation;
    7. quantize ta/tb;
    8. reconstruct the signal;
    9. compute MSE against the signal effectively used.

    Parameters
    ----------
    N : int
        Number of harmonics.

    M : int
        Number of quantization bins M_RbCP.

    cfg : dict
        Parsed YAML configuration.

    rng : numpy.random.Generator
        Random-number generator.

    Returns
    -------
    float
        Empirical MSE for one Monte Carlo realization.
    """

    T = _get_signal_period(cfg)
    Tt = cfg["signal"]["Tt"]
    w0 = 2 * np.pi / T
    n_vec = np.arange(1, N + 1)

    # ------------------------------------------------------------
    # 1. Generate raw signal
    # ------------------------------------------------------------
    x_raw, t = _generate_signal(cfg, rng)

    # ------------------------------------------------------------
    # 2. Band-limit signal
    # Choose W consistent with the target N:
    #     N = floor(W*T/2)
    # For the figure pipeline we use:
    #     W = 2*N/T
    # ------------------------------------------------------------
    W = 2 * N / T
    x_filtered = filter_periodic(x_raw, W, Tt, T)

    # ------------------------------------------------------------
    # 3. Adjust peak-to-peak AFTER filtering
    # ------------------------------------------------------------
    p2p_target = cfg["signal"]["peak_to_peak"]
    current_p2p = np.max(x_filtered) - np.min(x_filtered)

    if p2p_target != 0 and current_p2p != 0:
        x_filtered = x_filtered * (p2p_target / current_p2p)

    # ------------------------------------------------------------
    # 4. DC handling
    #
    # For Figure 1 faithful reproduction:
    #   dc.enabled = false
    # and we explicitly remove the mean before the RbCP pipeline.
    # ------------------------------------------------------------
    dc_enabled = cfg.get("dc", {}).get("enabled", False)

    if not dc_enabled:
        dc_value = Tt * np.sum(x_filtered) / T
        x_rbcp_input = x_filtered - dc_value
    else:
        x_rbcp_input = x_filtered

    # ------------------------------------------------------------
    # 5. Compute Fourier coefficients with trusted normalization logic
    #
    # IMPORTANT:
    # We use the class-based Fourier core because it returns the possibly
    # adjusted signal `x_used`, which must be used later in the MSE.
    # ------------------------------------------------------------
    fourier_core = FourierCoefficientCore(
        T=T,
        harmonics=N,
        sensor_nodes=1,
        dft_signal_periods=cfg["signal"].get("dft_signal_periods", 1)
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_rbcp_input,
        Tt,
        normalize=cfg["signal"]["normalize_dft"],
        norm=cfg["signal"]["normalization_target"]
    )

    x_used_1d = x_used[:, 0, 0]

    # ------------------------------------------------------------
    # 6. Compute ta/tb using ONLY the trusted complex-logarithm path
    # ------------------------------------------------------------
    ta, tb = calc_ta_tb(an[0, :, 0], bn[0, :, 0], n_vec, w0)
    ta = np.real(ta)
    tb = np.real(tb)

    # ------------------------------------------------------------
    # 7. Quantize ta/tb
    # ------------------------------------------------------------
    ta_q, tb_q = quantize_ta_tb(ta, tb, w0, M)

    # ------------------------------------------------------------
    # 8. Reconstruct signal
    # ------------------------------------------------------------
    x_hat = recover_signal(ta_q, tb_q, t, w0)

    # ------------------------------------------------------------
    # 9. Compute empirical MSE against the signal effectively used
    #
    # IMPORTANT:
    # If normalization was triggered in the Fourier step, x_used_1d is the
    # signal that corresponds to the coefficients used to derive ta/tb.
    # ------------------------------------------------------------
    mse = np.mean((x_used_1d - x_hat) ** 2)

    return mse


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def _get_signal_period(cfg):
    """
    Get the signal period from the configuration.

    Accepts either:
    - cfg["signal"]["T"]   (legacy figure configuration)
    - cfg["signal"]["tau"] (newer project-wide convention)

    Returns
    -------
    float
        Signal period.
    """

    if "T" in cfg["signal"]:
        return cfg["signal"]["T"]

    if "tau" in cfg["signal"]:
        return cfg["signal"]["tau"]

    raise KeyError("Expected cfg['signal']['T'] or cfg['signal']['tau'].")


def _generate_signal(cfg, rng):
    """
    Generate a random signal according to the YAML configuration.

    Supported distributions
    -----------------------
    - uniform
    - gaussian

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    rng : numpy.random.Generator
        Random-number generator.

    Returns
    -------
    tuple
        (x, t)

        x : np.ndarray
            Generated 1D raw signal.

        t : np.ndarray
            Time vector built using T/tau and Tt from the configuration.
    """

    T = _get_signal_period(cfg)
    Tt = cfg["signal"]["Tt"]

    t = np.arange(0, T, Tt)

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x = rng.uniform(-1, 1, size=len(t))

    elif dist == "gaussian":
        x = rng.normal(0, 1, size=len(t))

    else:
        raise ValueError(f"Unknown distribution: {dist}")

    return x, t


# =============================================================================
# SAVE UTILITIES
# =============================================================================

def save_dat_file(df, path, delimiter="\t"):
    """
    Save the figure dataset to a .dat-compatible tabular file.

    Parameters
    ----------
    df : pandas.DataFrame
        Final figure dataset.

    path : str
        Output file path.

    delimiter : str, optional
        Delimiter used in the saved file.
        Default is tab.
    """

    df.to_csv(
        path,
        sep=delimiter,
        index=False,
        float_format="%.8e"
    )
