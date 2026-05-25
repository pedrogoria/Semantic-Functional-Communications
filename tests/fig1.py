"""
fig1.py

Figure 1 Monte Carlo reproduction using the manuscript signal-generation procedure.

Monte Carlo signal generation (as described in the manuscript):
- X' ~ Uniform(-1, 1)
- X is a vector of i.i.d. realizations of X'
- x(t) is obtained by low-pass, flat-band FIR filtering with cutoff W/2
- The peak-to-peak value is set to 4 V unless additional scaling is required
  to satisfy the real-phase condition (Lemma 1)
- Channel is assumed error-free (RbCP baseline), i.e., only quantization distortion

This script writes:
    data/results/fig1.dat

Columns:
    N, M_RbCP, MSE_MC, MSE_UPPER, MSE_STAR

Author: SFC Project
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "sfc")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

from sfc.core.rbcp import compute_Q, mse_upper_bound, mse_star
from sfc.core.rbcp_sim import (
    generate_random_bandlimited_signal,
    enforce_lemma1_scaling,
    rbcp_encode_decode_mse_from_signal
)


def main():
    out_dir = os.path.join(ROOT_DIR, "data", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig1.dat")

    # Figure 1 axes and parameters (as in the manuscript)
    N_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    M_list = [4, 8, 16]

    # Monte Carlo controls
    seed = 12345
    trials = 2000

    # Window and sampling
    tau = 1.0
    dt = 0.001
    t = np.arange(-tau / 2.0, tau / 2.0, dt)
    w0 = 2.0 * np.pi / tau

    # Signal generation settings
    p2p_target = 4.0
    fir_taps = 201
    fir_window = "hann"

    rng = np.random.default_rng(seed)

    results = []

    print("[INFO] Running Figure 1 Monte Carlo with FIR low-pass signal generation.")
    print("[INFO] Output:", out_path)

    for M in M_list:
        Q = compute_Q(M)
        print(f"[INFO] Processing M_RbCP = {M}")

        for N in N_list:
            mse_acc = []

            # Bandwidth selection consistent with N = floor(pi W / w0)
            # Here we choose W so that floor(pi W / w0) == N.
            # With w0 = 2*pi/tau -> pi W / w0 = (W*tau)/2
            # A robust choice is W = (2N)/tau (so that (W*tau)/2 = N exactly).
            W_hz = (2.0 * N) / tau

            for _ in range(trials):
                # Generate random band-limited signal x(t)
                x = generate_random_bandlimited_signal(
                    t=t,
                    W_hz=W_hz,
                    p2p_target=p2p_target,
                    fir_taps=fir_taps,
                    window=fir_window,
                    rng=rng
                )

                # Enforce Lemma 1 condition if needed by scaling down
                x_scaled, _scale = enforce_lemma1_scaling(x, t, w0, N)

                # Full encode/decode MSE (channel error-free)
                mse_val = rbcp_encode_decode_mse_from_signal(
                    x=x_scaled,
                    t=t,
                    w0=w0,
                    N_harmonics=N,
                    M_rbcp=M
                )

                mse_acc.append(mse_val)

            mse_mc = float(np.mean(mse_acc))

            # Theoretical references for Figure 1
            mse_up = mse_upper_bound(N, Q)
            mse_st = mse_star(N, Q)

            results.append([N, M, mse_mc, mse_up, mse_st])

    data = np.array(results, dtype=float)

    header = "N\tM_RbCP\tMSE_MC\tMSE_UPPER\tMSE_STAR"

    np.savetxt(
        out_path,
        data,
        delimiter="\t",
        header=header,
        comments="",
        fmt="%.10e"
    )

    print("[INFO] Done.")
    print("[INFO] Preview (first 5 rows):")
    print(data[:5])


if __name__ == "__main__":
    main()
