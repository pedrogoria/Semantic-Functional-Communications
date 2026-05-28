"""
sfc/pipelines/rbcp_signal_representation.py

RbCP signal representation vs Benchmark + SFC using physical system parameters.

This module produces:

- x_filtered
- x_zero_mean
- x_rbcp
- x_sfc
- x_benchmark

System parameters are physical:
- S
- P
- B
- R
- L
- SNR_dB
- W
- tau

Derived parameters are obtained centrally from:
    build_derived_system_parameters(cfg)

IMPORTANT
---------
✅ NEW / INTERPRETATIVE LOGIC
For this figure, we propagate one representative signal through the SFC stack.

That means:
- the figure still plots one signal
- but M_RbCP is derived using the full system-level S and bandwidth sharing
- all event IDs of this representative signal are assigned locally to sensor 0
  in a figure-specific sensor_x_event used only inside this pipeline

IMPORTANT CHANGE REQUESTED
--------------------------
The SFC branch is fed with:
    ta, tb

and NOT with:
    ta_q, tb_q

That is:
- x_rbcp uses quantized ta/tb
- x_sfc uses non-quantized ta/tb and relies on the SFC event/channel stack

BANDWIDTH SHARING
-----------------
The total bandwidth B is the total system bandwidth.

Only the communication-budget-based quantities use per-sensor bandwidth slices:
- M_RbCP
- Benchmark / Nyquist bits-per-sample

For this representative-signal figure, the benchmark branch uses the bandwidth
slice of sensor 0:

    B_sensor = params.B_per_sensor[0]

If no bandwidth allocation is provided in the YAML, the split is equal among
all S sensors.
"""

import copy
import numpy as np

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import (
    calc_ta_tb,
    PhaseCoefficientCore,
)
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.nyquist import Nyquist
from sfc.core.channel.SFCChannel import SFCChannel

from sfc.core.system_parameters import (
    build_derived_system_parameters,
    compute_benchmark_bits_per_sample_single_sensor,
)


# =============================================================================
# MAIN
# =============================================================================

def generate_rbcp_signal_representation(cfg):
    """
    Generate the full deterministic experiment for:
    - RbCP
    - SFC
    - Nyquist Benchmark
    """

    rng = np.random.default_rng(cfg["reproducibility"]["seed"])

    # -------------------------------------------------------------------------
    # Centralized physical/system-derived parameters
    # -------------------------------------------------------------------------
    params = build_derived_system_parameters(cfg)

    S = params.S
    B = params.B
    R = params.R
    L = params.L

    SNR_dB = params.SNR_dB
    SNR = params.SNR

    W = params.W
    tau = params.tau

    N = params.N
    M_rbcp = params.M_rbcp

    print(f"[INFO] SNR (dB) = {SNR_dB}")
    print(f"[INFO] SNR (linear) = {SNR:.4e}")
    print(f"[INFO] Derived N = {N}")
    print(f"[INFO] Derived M_RbCP = {M_rbcp}")
    print(f"[INFO] Bandwidth allocation = {params.bandwidth_allocation}")
    print(f"[INFO] B_per_sensor = {params.B_per_sensor}")

    w0 = 2 * np.pi / tau
    n_vec = np.arange(1, N + 1)

    Tt = cfg["signal"]["Tt"]
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
    dc_enabled = cfg.get("dc", {}).get("enabled", False)

    if not dc_enabled:
        dc = Tt * np.sum(x_filtered) / tau
        x_zero_mean = x_filtered - dc
    else:
        x_zero_mean = x_filtered

    # -------------------------------------------------------------------------
    # =========================
    # RbCP BRANCH
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

    # -------------------------------------------------------------------------
    # Quantized version only for direct RbCP reconstruction
    # -------------------------------------------------------------------------
    ta_q, tb_q = quantize_ta_tb(ta, tb, w0, M_rbcp)

    x_rbcp = recover_signal(ta_q, tb_q, t, w0)

    # -------------------------------------------------------------------------
    # =========================
    # SFC BRANCH
    # =========================
    #
    # IMPORTANT CHANGE:
    # SFC is fed with ta and tb (NOT ta_q / tb_q)
    # -------------------------------------------------------------------------
    x_sfc, sfc_debug = _sfc_reconstruction(
        ta=ta,
        tb=tb,
        t=t,
        tau=tau,
        w0=w0,
        N=N,
        cfg=cfg
    )

    # -------------------------------------------------------------------------
    # =========================
    # BENCHMARK BRANCH
    # =========================
    #
    # IMPORTANT:
    # Benchmark uses the per-sensor bandwidth slice of the representative
    # signal (sensor 0), not the total B.
    # -------------------------------------------------------------------------
    x_benchmark = _benchmark_nyquist(
        x_zero_mean=x_zero_mean,
        tau=tau,
        Tt=Tt,
        W=W,
        SNR=SNR,
        B_sensor=params.B_per_sensor[0],
        cfg=cfg
    )

    return {
        "t": t,
        "x_filtered": x_filtered,
        "x_zero_mean": x_zero_mean,
        "x_rbcp": x_rbcp,
        "x_sfc": x_sfc,
        "x_benchmark": x_benchmark,
        "ta": ta,
        "tb": tb,
        "ta_q": ta_q,
        "tb_q": tb_q,
        "sfc_debug": sfc_debug,
    }


# =============================================================================
# SIGNAL
# =============================================================================

def _generate_signal(cfg, rng, n):
    """
    Generate one representative signal for the figure.
    """

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        return rng.uniform(-1, 1, n)
    elif dist == "gaussian":
        return rng.normal(0, 1, n)
    else:
        raise ValueError("Invalid distribution")


# =============================================================================
# SFC RECONSTRUCTION
# =============================================================================

def _sfc_reconstruction(ta, tb, t, tau, w0, N, cfg):
    """
    Reconstruct the signal after the SFC stack.

    Flow
    ----
    ta/tb -> events -> SFCChannel(events) -> events_est -> ta/tb_est -> x_sfc

    IMPORTANT
    ---------
    This branch is intentionally fed with:
        ta, tb

    and NOT with:
        ta_q, tb_q

    ✅ NEW / INTERPRETATIVE LOGIC
    ----------------------------
    For this figure, all event IDs of the representative signal are assigned
    to sensor 0 in a local sensor_x_event.
    """

    phase_core = PhaseCoefficientCore(
        T=tau,
        harmonics=N,
        n_sub_symbol=cfg["system"]["L"],
        resource=cfg["system"]["R"],
        sensor_nodes=1,
        bandwidth=cfg["system"]["B"],
        detect_errors=False,
        periods=1,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    # trusted shape: (num_periods, harmonics, sensors)
    ta_3d = ta.reshape(1, N, 1)
    tb_3d = tb.reshape(1, N, 1)

    events = phase_core.ta_tb_to_events(ta_3d, tb_3d)

    # -------------------------------------------------------------------------
    # Build local cfg for SFCChannel
    # -------------------------------------------------------------------------
    cfg_sfc = copy.deepcopy(cfg)

    num_event_ids = events.shape[1]
    S = cfg_sfc["system"]["S"]

    sensor_x_event = np.zeros((S, num_event_ids))
    sensor_x_event[0, :] = 1.0

    if "channel" not in cfg_sfc:
        cfg_sfc["channel"] = {}

    cfg_sfc["channel"]["sensor_x_event"] = sensor_x_event

    cfg_sfc["channel"].setdefault("collision_mode", "sum")
    cfg_sfc["channel"].setdefault("type", "awgn")
    cfg_sfc["channel"].setdefault("detection_mode", "threshold")
    cfg_sfc["channel"].setdefault("score_threshold", cfg_sfc["system"]["L"])

    # If absolute threshold is not set, let detector derive it centrally
    if "threshold" not in cfg_sfc["channel"]:
        cfg_sfc["channel"].setdefault("threshold_factor", 0.5)

    sfc_channel = SFCChannel(cfg_sfc)

    out = sfc_channel(events, return_intermediates=True)
    events_est = out["events_est"]

    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)

    ta_rec = np.real(ta_rec[0, :, 0])
    tb_rec = np.real(tb_rec[0, :, 0])

    x_sfc = recover_signal(ta_rec, tb_rec, t, w0)

    debug = {
        "events": events,
        "events_est": events_est,
        "ta_rec": ta_rec,
        "tb_rec": tb_rec,
        "channel_intermediates": out,
    }

    return x_sfc, debug


# =============================================================================
# BENCHMARK (NYQUIST CAPACITY-BASED)
# =============================================================================

def _benchmark_nyquist(x_zero_mean, tau, Tt, W, SNR, B_sensor, cfg):
    """
    Benchmark using the Nyquist core.

    Rules
    -----
    - sampling_rate is configurable from YAML:
          benchmark.sampling_rate
    - if omitted, defaults to W

    IMPORTANT
    ---------
    The benchmark communication budget uses the bandwidth slice of the
    representative sensor:

        B_sensor

    and NOT the total system bandwidth B.

    Capacity
    --------
    C_sensor = B_sensor * log2(1 + SNR)

    The number of bits per sample is derived from:
        bits_total = tau * C_sensor
        bits_per_sample = bits_total / num_samples
    """

    benchmark_cfg = cfg.get("benchmark", {})

    # configurable from YAML, default = W
    sampling_rate = benchmark_cfg.get("sampling_rate", W)

    bits_int = compute_benchmark_bits_per_sample_single_sensor(
        tau=tau,
        B_sensor=B_sensor,
        SNR=SNR,
        sampling_rate=sampling_rate
    )

    print(f"[INFO] Benchmark sampling_rate = {sampling_rate}")
    print(f"[INFO] Benchmark B_sensor = {B_sensor}")
    print(f"[INFO] Benchmark bits/sample = {bits_int}")

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

    xs = nyq(x_zero_mean, t, quantize=False)
    xs_q = nyq.quantize(xs)
    x_rec = nyq.recover_signal(xs_q)
