"""
fig1.py

Experiment: Figure 1 reproduction

This module computes:
- Theoretical MSE upper bound (Lemma 4)
- Analytical MSE approximation (Proposition 2)
- Monte Carlo estimation (simplified initial version)

IMPORTANT:
This is a first reproducible version.
Monte Carlo is intentionally simplified and will be replaced
with the full signal-based pipeline later.

Author: SFC Project
"""

import numpy as np
from core.rbcp import compute_Q, mse_upper_bound, mse_star


def run(cfg):
    """
    Run Figure 1 experiment.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary

    Returns
    -------
    data : np.ndarray
        Results matrix
    header : str
        Header string
    """

    # ------------------------------------------------------
    # Load configuration parameters
    # ------------------------------------------------------

    seed = cfg["reproducibility"]["seed"]
    trials = cfg["reproducibility"]["trials"]

    N_list = cfg["parameters"]["N_list"]
    M_list = cfg["parameters"]["M_RbCP_list"]

    np.random.seed(seed)

    results = []

    # ------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------

    for M in M_list:
        Q = compute_Q(M)

        for N in N_list:
            # --- Theoretical results ---
            mse_upper = mse_upper_bound(N, Q)
            mse_star_val = mse_star(N, Q)

            # --- Monte Carlo estimation ---
            mse_mc = monte_carlo_rbcp(N, M, trials)

            results.append([
                N,
                M,
                mse_mc,
                mse_upper,
                mse_star_val,
            ])

    data = np.array(results)

    header = (
        "N\tM_RbCP\tMSE_MC\tMSE_UPPER\tMSE_STAR"
    )

    return data, header


def monte_carlo_rbcp(N, M, trials):
    """
    Simplified Monte Carlo estimation.

    This function approximates the expected behavior under
    Proposition 2 with noise added.

    NOTE:
    This is NOT the full physical simulation yet.
    It is only used to match qualitative behavior.

    Parameters
    ----------
    N : int
        Number of harmonics
    M : int
        Number of quantization levels
    trials : int
        Number of Monte Carlo runs

    Returns
    -------
    float
        Estimated MSE
    """

    Q = compute_Q(M)

    values = []

    for _ in range(trials):
        # Base analytical value
        base = 2 * N * (1 - 2 * Q)

        # Add controlled noise to simulate variability
        noise = np.random.normal(0, 0.05 * base)

        values.append(base + noise)

    return np.mean(values)
