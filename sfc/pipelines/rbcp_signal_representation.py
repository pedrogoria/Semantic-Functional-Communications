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
- N0
- B
- R
- L
- W
- tau

Derived parameters are obtained centrally from:

    build_derived_system_parameters(cfg)

IMPORTANT
---------
For this figure, we propagate one representative signal through the SFC stack.

That means:
- the figure still plots one signal
- but M_RbCP is derived using the full system-level S and bandwidth sharing
- all event IDs of this representative signal are assigned locally to sensor 0
  in a figure-specific sensor_x_event used only inside this pipeline

SFC branch
----------
The SFC branch is fed with:

    ta, tb

and NOT with:

    ta_q, tb_q

That is:

- x_rbcp uses quantized ta/tb
- x_sfc uses non-quantized ta/tb and relies on the SFC event/channel stack

Bandwidth sharing
-----------------
The total bandwidth B is the total system bandwidth.

Only the communication-budget-based quantities use per-sensor bandwidth slices:

- M_RbCP
- Benchmark / Nyquist M

For this representative-signal figure, the Benchmark branch uses the bandwidth
slice of sensor 0:

    B_sensor = params.B_per_sensor[0]

and therefore the corresponding sensor-0 SNR:

    SNR_0 = P / (B_sensor * N0)

If no bandwidth allocation is provided in the YAML, the split is equal among
all S sensors.
"""

import copy
import numpy as np

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.channel.SFCChannel import SFCChannel

from sfc.core.system_parameters import (
    build_derived_system_parameters,
    compute_benchmark_M_single_sensor,
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_rbcp_signal_representation_data(cfg):
    """
    Generate one representative-signal dataset comparing:
    - filtered signal
    - zero-mean signal
    - RbCP reconstruction
    - SFC reconstruction
    - Benchmark reconstruction

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    Returns
    -------
    dict
        Dictionary containing:
        - t
        - x_raw
        - x_filtered
        - x_zero_mean
        - x_rbcp
        - x_sfc
        - x_benchmark
        - ta
        - tb
        - ta_q
        - tb_q
        - params
        - mse_rbcp
        - mse_sfc
        - mse_benchmark
        - M_rbcp
        - M_benchmark
    """

    rng = np.random.default_rng(cfg["reproducibility"]["seed"])
    params = build_derived_system_parameters(cfg)

    # Figure uses one representative signal only.
    n_periods = 1
    N = cfg["signal"].get("N_override", params.N)
    tau = params.tau
    Tt = cfg["signal"]["Tt"]
    w0 = 2.0 * np.pi / tau

    t = np.arange(0.0, tau, Tt)
    n_time = len(t)

    print("[INFO] Running rbcp_signal_representation")
    print(f"[INFO] S = {params.S}")
    print(f"[INFO] P = {params.P}")
    print(f"[INFO] N0 = {params.N0}")
    print(f"[INFO] B = {params.B}")
    print(f"[INFO] R = {params.R}")
    print(f"[INFO] L = {params.L}")
    print(f"[INFO] SNR_total_dB = {params.SNR_dB}")
    print(f"[INFO] SNR_sensor_0_dB = {params.SNR_per_sensor_dB[0]}")
    print(f"[INFO] W = {params.W}")
    print(f"[INFO] tau = {params.tau}")
    print(f"[INFO] N = {N}")
    print(f"[INFO] M_RbCP = {params.M_rbcp}")
    print(f"[INFO] M_RbCP per sensor = {params.M_rbcp_per_sensor}")
    print(f"[INFO] B_per_sensor = {params.B_per_sensor}")
    print(f"[INFO] quantization_force_power_of_two = {params.quantization_force_power_of_two}")
    print(f"[INFO] quantization_rounding_mode = {params.quantization_rounding_mode}")

    # -------------------------------------------------------------------------
    # 1. Generate one representative signal
    # -------------------------------------------------------------------------
    x_raw = _generate_representative_signal(
        cfg=cfg,
        rng=rng,
        num_time_samples=n_time
    )

    # Keep tensor convention for trusted Fourier/phase cores:
    # shape = (time, periods, sensors)
    x_raw_3d = x_raw[:, None, None]

    # -------------------------------------------------------------------------
    # 2. Band-limit
    # -------------------------------------------------------------------------
    x_filtered_3d = _filter_signal_tensor(
        x_raw_3d,
        N=N,
        tau=tau,
        Tt=Tt
    )

    # -------------------------------------------------------------------------
    # 3. Peak-to-peak control
    # -------------------------------------------------------------------------
    x_filtered_3d = _apply_peak_to_peak_control(
        x_filtered_3d,
        cfg["signal"]["peak_to_peak"]
    )

    # -------------------------------------------------------------------------
    # 4. DC handling
    # -------------------------------------------------------------------------
    x_zero_mean_3d = _apply_dc_handling(
        x_filtered=x_filtered_3d,
        tau=tau,
        Tt=Tt,
        dc_enabled=cfg.get("dc", {}).get("enabled", False)
    )

    # -------------------------------------------------------------------------
    # 5. Fourier coefficients
    # -------------------------------------------------------------------------
    an, bn, x_used = _compute_fourier_coefficients(
        x_zero_mean=x_zero_mean_3d,
        tau=tau,
        N=N,
        Tt=Tt,
        normalize_dft=cfg["signal"]["normalize_dft"],
        normalization_target=cfg["signal"]["normalization_target"]
    )

    x_used_1d = x_used[:, 0, 0]

    # -------------------------------------------------------------------------
    # 6. ta/tb
    # -------------------------------------------------------------------------
    ta, tb = _compute_phase_coefficients(
        an=an,
        bn=bn,
        tau=tau,
        N=N,
        cfg=cfg,
        params=params,
        n_periods=n_periods
    )

    # Extract 1D phase vectors for the representative signal.
    ta_1d = ta[0, :, 0]
    tb_1d = tb[0, :, 0]

    # -------------------------------------------------------------------------
    # 7. RbCP branch (quantized ta/tb)
    # -------------------------------------------------------------------------
    ta_q, tb_q = quantize_ta_tb(
        ta_1d,
        tb_1d,
        w0,
        params.M_rbcp
    )

    x_rbcp = recover_signal(
        ta_q,
        tb_q,
        t,
        w0
    )
    mse_rbcp = float(np.mean((x_used_1d - x_rbcp) ** 2))

    # -------------------------------------------------------------------------
    # 8. SFC branch (NON-quantized ta/tb)
    # -------------------------------------------------------------------------
    x_sfc = np.full_like(x_used_1d, np.nan, dtype=float)
    mse_sfc = np.nan

    if cfg.get("mode", {}).get("run_sfc", True):
        x_sfc = _run_sfc_branch(
            ta=ta,
            tb=tb,
            cfg=cfg,
            params=params,
            N=N,
            t=t,
            sfc_enabled=True
        )
        mse_sfc = float(np.mean((x_used_1d - x_sfc) ** 2))

    # -------------------------------------------------------------------------
    # 9. Benchmark branch
    #    Uses sensor-0 bandwidth slice only.
    # -------------------------------------------------------------------------
    x_benchmark, M_benchmark = _run_benchmark_branch(
        x_ref=x_used_1d,
        t=t,
        cfg=cfg,
        params=params
    )
    mse_benchmark = float(np.mean((x_used_1d - x_benchmark) ** 2))

    return {
        "t": t,
        "x_raw": x_raw,
        "x_filtered": x_filtered_3d[:, 0, 0],
        "x_zero_mean": x_zero_mean_3d[:, 0, 0],
        "x_rbcp": x_rbcp,
        "x_sfc": x_sfc,
        "x_benchmark": x_benchmark,
        "ta": ta_1d,
        "tb": tb_1d,
        "ta_q": ta_q,
        "tb_q": tb_q,
        "params": params,
        "mse_rbcp": mse_rbcp,
        "mse_sfc": mse_sfc,
        "mse_benchmark": mse_benchmark,
        "M_rbcp": params.M_rbcp,
        "M_benchmark": M_benchmark,
    }


# =============================================================================
# SIGNAL GENERATION / PREPROCESSING
# =============================================================================

def _generate_representative_signal(cfg, rng, num_time_samples):
    """
    Generate one representative 1D signal.
    """

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        return rng.uniform(-1, 1, size=num_time_samples)

    if dist == "gaussian":
        return rng.normal(0, 1, size=num_time_samples)

    raise ValueError("Invalid distribution")


def _filter_signal_tensor(x_raw_3d, N, tau, Tt):
    """
    Band-limit the representative signal tensor using W_eff = 2N / tau.
    """

    W_eff = 2.0 * N / tau
    x_filtered = np.zeros_like(x_raw_3d)

    # shape = (time, periods, sensors=1)
    _, n_periods, S = x_raw_3d.shape

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw_3d[:, p, s],
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
    Tt,
    normalize_dft,
    normalization_target
):
    """
    Compute Fourier coefficients for the representative signal.
    """

    fourier_core = FourierCoefficientCore(
        T=tau,
        harmonics=N,
        sensor_nodes=1
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=normalize_dft,
        norm=normalization_target
    )

    return an, bn, x_used


def _compute_phase_coefficients(an, bn, tau, N, cfg, params, n_periods):
    """
    Compute ta/tb for the representative signal only.
    """

    phase_core = PhaseCoefficientCore(
        T=tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=1,
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
# SFC BRANCH
# =============================================================================

def _run_sfc_branch(ta, tb, cfg, params, N, t, sfc_enabled=True):
    """
    Run the representative-signal SFC branch.

    IMPORTANT
    ---------
    - The input is NON-quantized ta/tb.
    - Event IDs belong to the representative signal only (2N IDs).
    - Those IDs are assigned to sensor 0 in a figure-specific sensor_x_event.
    """

    if not sfc_enabled:
        return np.full_like(t, np.nan, dtype=float)

    n_periods = 1
    w0 = 2.0 * np.pi / params.tau

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=1,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    # Representative-signal events: shape based on 1 signal only.
    events = phase_core.ta_tb_to_events(ta, tb)

    # Build one SFC channel using a figure-specific sensor_x_event.
    sfc_channel = _build_representative_sfc_channel(cfg, params, N)

    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)
    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    x_sfc = recover_signal(
        ta_rec[0, :, 0],
        tb_rec[0, :, 0],
        t,
        w0
    )

    return x_sfc


def _build_representative_sfc_channel(cfg, params, N):
    """
    Build an SFCChannel for one representative signal.

    Event convention in this pipeline
    ---------------------------------
    Only the representative signal exists in the event matrix, so the number of
    event IDs is:

        2 * N

    But the physical system still has S sensors.

    We assign all event IDs to sensor 0 in a figure-specific sensor_x_event:

        shape = (S, 2N)

    with:
        row 0 -> ones
        rows 1..S-1 -> zeros
    """

    cfg_sfc = copy.deepcopy(cfg)

    if "channel" not in cfg_sfc:
        cfg_sfc["channel"] = {}

    sensor_x_event = np.zeros((params.S, 2 * N))
    sensor_x_event[0, :] = 1.0

    cfg_sfc["channel"]["sensor_x_event"] = sensor_x_event
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
        cfg_sfc["reproducibility"]["seed"] = cfg.get(
            "reproducibility", {}
        ).get(
            "seed",
            cfg.get("monte_carlo", {}).get("seed", 12345)
        )

    return SFCChannel(cfg_sfc)


# =============================================================================
# BENCHMARK BRANCH
# =============================================================================

def _run_benchmark_branch(x_ref, t, cfg, params):
    """
    Run Benchmark / Nyquist reconstruction for the representative signal.

    IMPORTANT
    ---------
    For this figure, the Benchmark uses the bandwidth slice of sensor 0 only:

        B_sensor = params.B_per_sensor[0]

    The number of bins M is computed from the physical sensor-0 budget:

        SNR_0 = P / (B_sensor * N0)

    The number of bins M is obtained from the core, using the current default
    quantization policy (free integer by default).
    """

    benchmark_cfg = cfg.get("benchmark", {})
    sampling_rate = benchmark_cfg.get("sampling_rate", params.W)
    effective_rate_factor = benchmark_cfg.get("effective_rate_factor", 1.0)

    B_sensor = params.B_per_sensor[0]

    M = compute_benchmark_M_single_sensor(
        tau=params.tau,
        B_sensor=B_sensor,
        P=params.P,
        N0=params.N0,
        sampling_rate=effective_rate_factor * sampling_rate,
        force_power_of_two=params.quantization_force_power_of_two,
        rounding_mode=params.quantization_rounding_mode
    )

    x_hat = _benchmark_sample_quantize_reconstruct(
        x_ref=x_ref,
        t=t,
        sampling_rate=sampling_rate,
        M=M
    )

    return x_hat, M


def _benchmark_sample_quantize_reconstruct(x_ref, t, sampling_rate, M):
    """
    Benchmark reconstruction using:
    1. uniform sampling
    2. uniform scalar quantization with M bins
    3. sinc reconstruction

    This implementation does NOT require M to be a power of 2.
    """

    tau = t[-1] + (t[1] - t[0])

    Ts = 1.0 / sampling_rate
    t_samples = np.arange(0.0, tau, Ts)

    # Sample the reference signal on the sampling grid.
    x_samples = np.interp(t_samples, t, x_ref)

    # Uniform quantization over the signal dynamic range.
    x_min = np.min(x_ref)
    x_max = np.max(x_ref)

    if np.isclose(x_max, x_min):
        return np.full_like(t, x_min, dtype=float)

    delta = (x_max - x_min) / M

    # Mid-rise / clipped uniform quantizer.
    idx = np.floor((x_samples - x_min) / delta)
    idx = np.clip(idx, 0, M - 1)

    xq = x_min + (idx + 0.5) * delta

    # Sinc reconstruction:
    # y(t) = sum_k xq[k] * sinc(fs * t - k)
    fs = sampling_rate
    k = np.arange(len(xq))
    x_hat = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        x_hat[i] = np.sum(xq * np.sinc(fs * ti - k))

    return x_hat
