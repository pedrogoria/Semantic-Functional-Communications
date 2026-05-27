"""
RbCP signal representation vs Benchmark using physical system parameters.

Now fully based on:

- system-level parameters (S, B, R, SNR_dB)
- derived parameters via system_parameters.py
- Nyquist benchmark from core class
"""

import numpy as np

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import calc_ta_tb
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.nyquist import Nyquist

from sfc.core.system_parameters import (
    compute_N,
    compute_M_rbcp,
)


# =============================================================================
# MAIN
# =============================================================================

def generate_rbcp_signal_representation(cfg):

    rng = np.random.default_rng(cfg["reproducibility"]["seed"])

    # -------------------------------------------------------------------------
    # SYSTEM PARAMETERS
    # -------------------------------------------------------------------------
    S = cfg["system"]["S"]
    B = cfg["system"]["B"]
    R = cfg["system"]["R"]

    # ✅ NEW: SNR in dB → convert to linear
    SNR_dB = cfg["system"]["SNR_dB"]
    SNR = 10 ** (SNR_dB / 10.0)

    print(f"[INFO] SNR (dB) = {SNR_dB}")
    print(f"[INFO] SNR (linear) = {SNR:.4e}")

    # -------------------------------------------------------------------------
    # SIGNAL PARAMETERS
    # -------------------------------------------------------------------------
    W = cfg["signal"]["W"]
    tau = cfg["signal"]["tau"]
    Tt = cfg["signal"]["Tt"]

    # -------------------------------------------------------------------------
    # DERIVED PARAMETERS
    # -------------------------------------------------------------------------
    N = compute_N(W, tau)
    M_rbcp = compute_M_rbcp(S, W, tau, B, SNR)

    print(f"[INFO] Derived N = {N}")
    print(f"[INFO] Derived M_RbCP = {M_rbcp}")

    w0 = 2 * np.pi / tau
    n_vec = np.arange(1, N + 1)

    t = np.arange(0, tau, Tt)

    # -------------------------------------------------------------------------
    # SIGNAL GENERATION
    # -------------------------------------------------------------------------
    x_raw = _generate_signal(cfg, rng, len(t))

    # -------------------------------------------------------------------------
    # BANDLIMIT
    # -------------------------------------------------------------------------
    W_eff = 2 * N / tau
    x_filtered = filter_periodic(x_raw, W_eff, Tt, tau)

    # -------------------------------------------------------------------------
    # PEAK-TO-PEAK CONTROL
    # -------------------------------------------------------------------------
    p2p_target = cfg["signal"]["peak_to_peak"]
    current_p2p = np.max(x_filtered) - np.min(x_filtered)

    if p2p_target != 0:
        x_filtered = x_filtered * (p2p_target / current_p2p)

    # -------------------------------------------------------------------------
    # DC REMOVAL
    # -------------------------------------------------------------------------
    dc = Tt * np.sum(x_filtered) / tau
    x_zero_mean = x_filtered - dc

    # -------------------------------------------------------------------------
    # =========================
    # RbCP
    # =========================
    # -------------------------------------------------------------------------
    fourier_core = FourierCoefficientCore(
        T=tau,
        harmonics=N,
        sensor_nodes=1
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=cfg["signal"]["normalize_dft"],
        norm=cfg["signal"]["normalization_target"]
    )

    x_used = x_used[:, 0, 0]

    ta, tb = calc_ta_tb(an[0, :, 0], bn[0, :, 0], n_vec, w0)
    ta = np.real(ta)
    tb = np.real(tb)

    ta_q, tb_q = quantize_ta_tb(ta, tb, w0, M_rbcp)

    x_rbcp = recover_signal(ta_q, tb_q, t, w0)

    # -------------------------------------------------------------------------
    # =========================
    # BENCHMARK (NYQUIST + SHANNON)
    # =========================
    # -------------------------------------------------------------------------
    x_benchmark = _benchmark_nyquist(
        x_zero_mean,
        tau,
        Tt,
        W,
        B,
        SNR
    )

    # -------------------------------------------------------------------------
    return {
        "t": t,
        "x_filtered": x_filtered,
        "x_zero_mean": x_zero_mean,
        "x_rbcp": x_rbcp,
        "x_benchmark": x_benchmark,
    }


# =============================================================================
# SIGNAL
# =============================================================================

def _generate_signal(cfg, rng, n):

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        return rng.uniform(-1, 1, n)
    elif dist == "gaussian":
        return rng.normal(0, 1, n)
    else:
        raise ValueError("Invalid distribution")


# =============================================================================
# BENCHMARK (NYQUIST CAPACITY-BASED)
# =============================================================================

def _benchmark_nyquist(x, tau, Tt, W, B, SNR):

    # ------------------------------------------------------------
    # Sampling at 1/W (as defined in manuscript)
    # ------------------------------------------------------------
    sampling_rate = W

    # ------------------------------------------------------------
    # Channel capacity (linear SNR)
    # ------------------------------------------------------------
    C = B * np.log2(1 + SNR)

    bits_total = tau * C

    num_samples = int(np.floor(tau * sampling_rate))

    bits_per_sample = bits_total / num_samples

    bits_int = int(np.floor(bits_per_sample))
    bits_int = max(bits_int, 1)

    print(f"[INFO] Benchmark bits/sample = {bits_int}")

    # ------------------------------------------------------------
    # Nyquist core
    # ------------------------------------------------------------
    nyq = Nyquist(
        T=tau,
        Tt=Tt,
        sampling_rate=sampling_rate,
        sensor_nodes=1,
        bits_codeword=bits_int,
        snr_dB=100.0,
        bandwidth=1e6
    )

    t = np.arange(0, tau, Tt)

    xs = nyq(x, t, quantize=False)
    xs_q = nyq.quantize(xs)
    x_rec = nyq.recover_signal(xs_q)

    return x_rec
