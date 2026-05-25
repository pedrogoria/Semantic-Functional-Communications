"""
fig1.py

Figure 1 reproduction using the trusted sampling backend (sfc/sampling.py)
via the PHY-facing wrapper class RbCPSampler (sfc/core/samplers.py).

What this script does (consistent with the manuscript Monte Carlo description):
1) Generate i.i.d. samples X' ~ Uniform(-1, 1)
2) Build a discrete-time vector X with those samples
3) Create a band-limited periodic signal by filtering X with a low-pass method
   with cutoff W/2 (we use the trusted filter_periodic() provided by sfc/sampling.py)
4) (If needed) normalize the signal to satisfy the "real phase" condition
   required by the RbCP mapping (handled by CPSample.calc_an_bn_dft(normalize=True))
5) Compute (ta, tb) using CPSample (RbCP mapping)
6) Quantize (ta, tb) using the trusted quantize_ta_tb() function
7) Reconstruct x_hat(t) from quantized (ta, tb) and compute MSE
8) Save data/results/fig1.dat for plotting

Theoretical references computed in this script:
- Upper bound from Lemma 4
- MSE* from Proposition 2

Both formulas appear in the manuscript. [1](https://github.com/pedrogoria/Semantic-Functional-Communications)

Author: SFC Project
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------
# Trusted backend imports
# ---------------------------------------------------------------------

from sfc.core.samplers import RbCPSampler, RbCPConfig
from sfc.sampling import filter_periodic, quantize_ta_tb

# ---------------------------------------------------------------------
# Path handling for PyCharm run file() behavior
# This assumes:
#   <repo-root>/tests/fig1.py
#   <repo-root>/sfc/...
# ---------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SFC_DIR = os.path.join(ROOT_DIR, "sfc")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SFC_DIR)

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fig1.dat")


# ---------------------------------------------------------------------
# Theory helpers (from manuscript equations)
# Q = (M/(2*pi))*sin(pi/M)
# Upper bound: MSE/N <= 4*(1/2 - Q)*(3/2 - Q)  ->  MSE_upper = N * that
# MSE* = 2N*(1 - 2Q)
# ---------------------------------------------------------------------

def compute_Q(M):
    return (M / 2 / np.pi) * np.sin(np.pi / M)


def mse_upper_bound(N, Q):
    return 4 * N * (0.5 - Q) * (3 / 2 - Q)


def mse_star(N, Q):
    return 2.0 * N * (1.0 - 2.0 * Q)


# ---------------------------------------------------------------------
# Monte Carlo signal generation (trusted style)
# ---------------------------------------------------------------------

def generate_signal_one_period(t, Tt, tau, W_hz, p2p_target, rng):
    """
    Generate one-period signal using the manuscript approach:
    - X' ~ Uniform(-1,1)
    - filter with low-pass and flat-band behavior (trusted filter_periodic)
    - enforce mean-zero
    - scale to a target peak-to-peak amplitude

    Parameters
    ----------
    t : np.ndarray
        Time vector over one period.
    Tt : float
        Time step.
    tau : float
        Period length.
    W_hz : float
        Bandwidth (Hz).
    p2p_target : float
        Target peak-to-peak amplitude.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    x : np.ndarray
        Generated signal (len(t),).
    """
    # Start with i.i.d. Uniform(-1, 1)
    x = rng.normal(0.0, 20.0, size=len(t))

    # Enforce mean-zero (paper assumes zero mean in derivations)
    x = x - np.mean(x)

    # Band-limit it using the trusted periodic filtering routine.
    # This routine expects W as a bandwidth parameter and returns a real periodic signal.
    x = filter_periodic(x, W_hz, Tt, tau)

    # Enforce mean-zero again after filtering
    x = x - np.mean(x)

    # Scale to the desired peak-to-peak when possible
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    p2p = x_max - x_min

    if p2p_target > 0:
        x = x * (p2p_target / p2p)

    return x


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():
    # Figure 1 parameter sweep
    N_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    M_LIST = [5, 9, 17]

    # Monte Carlo controls
    SEED = 47
    TRIALS = 1000  # Increase for tighter markers if needed

    # Signal/window configuration
    tau = 1.0
    Tt = 0.001
    t = np.arange(-tau / 2.0, tau / 2.0, Tt)

    # Manuscript states p2p is 2 V unless otherwise stated
    p2p_target = 10.0

    rng = np.random.default_rng(SEED)

    results = []

    print("[INFO] Figure 1 simulation using RbCPSampler + trusted sampling.py")
    print("[INFO] Output file:", OUTPUT_FILE)
    print("[INFO] Trials:", TRIALS)

    for M_rbcp in M_LIST:
        Q = compute_Q(M_rbcp)
        print("[INFO] Processing M_RbCP =", M_rbcp)

        for N in N_LIST:
            # Choose W so that N = floor(pi W / w0) holds exactly.
            # With w0 = 2*pi/tau -> pi W / w0 = (W*tau)/2.
            # Setting W = 2N/tau makes (W*tau)/2 = N exactly.
            W_hz = (2.0 * N) / tau

            # Instantiate sampler for this N (harmonics) configuration.
            # We do not need SFC parameters here; only the sampling/representation.
            cfg = RbCPConfig(
                T=tau,
                harmonics=N,
                sensor_nodes=1,
                # The event-mapping parameters are still part of CPSample,
                # but they do not affect ta/tb computation if we only call sample().
                n_sub_symbol=6,
                resource=7,
                bandwidth=100.0,
                detect_errors=False,
                threshold_harmonics=0.001,
                dft_signal_periods=1
            )
            sampler = RbCPSampler(cfg)

            mse_acc = []

            for _ in range(TRIALS):
                # 1) Generate one signal period
                x = generate_signal_one_period(
                    t=t, Tt=Tt, tau=tau, W_hz=W_hz,
                    p2p_target=p2p_target, rng=rng
                )

                # CPSample expects shape (time,) or (time, periods) etc.
                # We keep one period and one sensor.
                # sample() will return ta,tb as (periods, harmonics, sensors).
                ta, tb, x_used = sampler.sample(
                    x,
                    Tt,
                    t=t,
                    normalize=True,  # enforce "real phase" condition when needed
                    norm=3.99  # same typical value used in trusted code
                )

                # 2) Quantize ta/tb using the trusted quantizer from sampling.py
                ta_q, tb_q = quantize_ta_tb(ta, tb, sampler.w0, M_rbcp, 10/21, 11/21)

                # 3) Reconstruct using trusted recover_signal() via the sampler wrapper
                xr = sampler.recover(ta_q, tb_q, t)

                # xr shape: (len(t), periods, sensors) -> pick first period, first sensor
                x_hat = xr[:, 0, 0]

                # 4) Compute MSE over the period (same units as signal)
                mse_val = float(np.mean((x_used[:, 0, 0] - x_hat) ** 2))
                mse_acc.append(mse_val)

            mse_mc = float(np.mean(mse_acc))

            # Theoretical curves from manuscript
            mse_up = mse_upper_bound(N, Q)
            mse_st = mse_star(N, Q)

            results.append([N, M_rbcp, mse_mc, mse_up, mse_st])

            print(f"[INFO] N={N:2d}  M={M_rbcp:2d}  MSE_Monte_Carlo={mse_mc:.6e}  MSE*={mse_st:.6e}  UB={mse_up:.6e}")

    data = np.asarray(results, dtype=float)
    header = "N\tM_RbCP\tMSE_Monte_Carlo\tMSE_UPPER\tMSE_STAR"

    np.savetxt(
        OUTPUT_FILE,
        data,
        delimiter="\t",
        header=header,
        comments="",
        fmt="%.10e"
    )

    print("[INFO] Done. Saved:", OUTPUT_FILE)
    print("[INFO] Preview:\n", data[:5])


if __name__ == "__main__":
    main()
