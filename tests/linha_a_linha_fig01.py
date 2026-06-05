"""
tests/debug_rbcp_single_run.py

Single-run RbCP representation / reconstruction debug script.

This script is representational only. It does NOT use:
- P
- N0
- SNR
- channel capacity
- SFCChannel
- FDMA / MAC
- semantic error detection

Flow
----
1. Generate one random signal over one period.
2. Band-limit the signal with filter_periodic.
3. Apply peak-to-peak control after filtering.
4. Optionally remove DC.
5. Compute Fourier coefficients using FourierCoefficientCore.
6. Compute ta/tb using the complex-log formulation.
7. Quantize ta/tb.
8. Reconstruct from:
   - non-quantized ta/tb
   - quantized ta/tb
9. Compute empirical MSE against x_used.
10. Compare with RbCP theory:
   - Q
   - MSE*
   - upper bound
11. Plot signal, reconstruction error, and ta/tb before/after quantization.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PATH HANDLING
# =============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import calc_ta_tb
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.theory import (
    compute_q,
    rbcp_mse_upper_bound,
    rbcp_mse_star,
)


# =============================================================================
# PARAMETERS
# =============================================================================

T = 1.0
Tt = 0.01

N = 5
M_RbCP = 8

peak_to_peak = 8.0
distribution = "uniform"

normalize_dft = True
normalization_target = 3.9
dc_enabled = False

seed = 12345


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_single_rbcp_debug(
    T: float = T,
    Tt: float = Tt,
    N: int = N,
    M_RbCP: int = M_RbCP,
    peak_to_peak: float = peak_to_peak,
    distribution: str = distribution,
    normalize_dft: bool = normalize_dft,
    normalization_target: float = normalization_target,
    dc_enabled: bool = dc_enabled,
    seed: int = seed,
    do_plot: bool = True,
):
    """
    Run one RbCP debug trial.

    Returns
    -------
    dict
        Dictionary containing all relevant intermediate and final quantities.
    """

    rng = np.random.default_rng(seed)

    w0 = 2.0 * np.pi / T
    W = 2.0 * N / T
    n_vec = np.arange(1, N + 1)
    t = np.arange(0.0, T, Tt)

    # -------------------------------------------------------------------------
    # 1. Raw signal
    # -------------------------------------------------------------------------
    if distribution == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=len(t))
    elif distribution == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=len(t))
    else:
        raise ValueError("distribution must be 'uniform' or 'gaussian'.")

    # -------------------------------------------------------------------------
    # 2. Band-limit
    # -------------------------------------------------------------------------
    x_filtered = filter_periodic(
        x_raw,
        W,
        Tt,
        T
    )

    # -------------------------------------------------------------------------
    # 3. Peak-to-peak control after filtering
    # -------------------------------------------------------------------------
    current_p2p = np.max(x_filtered) - np.min(x_filtered)

    if peak_to_peak != 0 and current_p2p != 0:
        x_filtered = x_filtered * (peak_to_peak / current_p2p)

    # -------------------------------------------------------------------------
    # 4. DC handling
    # -------------------------------------------------------------------------
    dc_value = Tt * np.sum(x_filtered) / T

    if not dc_enabled:
        x_rbcp_input = x_filtered - dc_value
    else:
        x_rbcp_input = x_filtered.copy()

    print("[INFO] Peak-to-peak filtered =", np.max(x_filtered) - np.min(x_filtered))
    print("[INFO] DC removed =", not dc_enabled)
    print("[INFO] DC value =", dc_value)

    # -------------------------------------------------------------------------
    # 5. Fourier coefficients with trusted normalization logic
    # -------------------------------------------------------------------------
    fourier_core = FourierCoefficientCore(
        T=T,
        harmonics=N,
        sensor_nodes=1,
        dft_signal_periods=1
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_rbcp_input,
        Tt,
        normalize=normalize_dft,
        norm=normalization_target
    )

    x_used_1d = x_used[:, 0, 0]

    max_ab = np.max(an[0, :, 0] ** 2 + bn[0, :, 0] ** 2)

    print("[INFO] max(an^2 + bn^2) =", max_ab)
    print("[INFO] Peak-to-peak used =", np.max(x_used_1d) - np.min(x_used_1d))

    # -------------------------------------------------------------------------
    # 6. ta/tb via complex-log formulation
    # -------------------------------------------------------------------------
    ta, tb = calc_ta_tb(
        an[0, :, 0],
        bn[0, :, 0],
        n_vec,
        w0
    )

    ta = np.real(ta)
    tb = np.real(tb)

    # -------------------------------------------------------------------------
    # 7. Quantization
    # -------------------------------------------------------------------------
    ta_q, tb_q = quantize_ta_tb(
        ta,
        tb,
        w0,
        M_RbCP
    )

    # -------------------------------------------------------------------------
    # 8. Reconstruction
    # -------------------------------------------------------------------------
    x_r = recover_signal(
        ta,
        tb,
        t,
        w0
    )

    x_hat = recover_signal(
        ta_q,
        tb_q,
        t,
        w0
    )

    # -------------------------------------------------------------------------
    # 9. MSE
    # -------------------------------------------------------------------------
    mse = float(np.mean((x_used_1d - x_hat) ** 2))
    mse_unquantized = float(np.mean((x_used_1d - x_r) ** 2))

    # -------------------------------------------------------------------------
    # 10. Theory
    # -------------------------------------------------------------------------
    Q = compute_q(M_RbCP)
    mse_up = rbcp_mse_upper_bound(N, Q)
    mse_st = rbcp_mse_star(N, Q)

    print("\n[RESULTS]")
    print("N =", N)
    print("M_RbCP =", M_RbCP)
    print("Q =", Q)
    print("MSE quantized =", mse)
    print("MSE unquantized =", mse_unquantized)
    print("MSE* =", mse_st)
    print("Upper bound =", mse_up)

    print("\n[PHASE PARAMETERS]")
    print("ta =", ta)
    print("tb =", tb)
    print("ta_q =", ta_q)
    print("tb_q =", tb_q)

    # -------------------------------------------------------------------------
    # 11. Plots
    # -------------------------------------------------------------------------
    if do_plot:
        plt.figure(figsize=(11, 5))
        plt.plot(t, x_raw, label="x_raw", alpha=0.4)
        plt.plot(t, x_filtered, label="x_filtered, p2p adjusted", linewidth=2)
        plt.plot(t, x_used_1d, label="x_used, after DFT normalization", linewidth=2)
        plt.plot(t, x_hat, label="x_hat, quantized reconstruction", linestyle="--", linewidth=2)
        plt.plot(t, x_r, label="x_r, unquantized reconstruction", linestyle=":", linewidth=2)
        plt.xlabel(r"$t$")
        plt.ylabel("Amplitude")
        plt.title(f"RbCP single run | N={N}, M_RbCP={M_RbCP}, MSE={mse:.3e}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(11, 4))
        plt.plot(t, x_used_1d - x_hat, color="red", label="quantized error")
        plt.plot(t, x_used_1d - x_r, color="black", linestyle="--", label="unquantized error")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x_{\mathrm{used}} - \hat{x}$")
        plt.title("Reconstruction error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(9, 4))
        plt.plot(n_vec, ta, "o-", label="ta")
        plt.plot(n_vec, ta_q, "s--", label="ta_q")
        plt.plot(n_vec, tb, "o-", label="tb")
        plt.plot(n_vec, tb_q, "s--", label="tb_q")
        plt.xlabel("Harmonic index")
        plt.ylabel("Phase parameter")
        plt.title("Phase parameters before/after quantization")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "t": t,
        "x_raw": x_raw,
        "x_filtered": x_filtered,
        "x_rbcp_input": x_rbcp_input,
        "x_used": x_used_1d,
        "an": an,
        "bn": bn,
        "ta": ta,
        "tb": tb,
        "ta_q": ta_q,
        "tb_q": tb_q,
        "x_r": x_r,
        "x_hat": x_hat,
        "mse": mse,
        "mse_unquantized": mse_unquantized,
        "Q": Q,
        "mse_star": mse_st,
        "upper_bound": mse_up,
        "N": N,
        "M_RbCP": M_RbCP,
        "W": W,
        "w0": w0,
    }


if __name__ == "__main__":
    run_single_rbcp_debug()
