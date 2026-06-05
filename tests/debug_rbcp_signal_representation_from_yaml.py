"""
tests/debug_rbcp_signal_representation_from_yaml.py

Debug runner for:
- RbCP
- SFC
- Nyquist Benchmark

This file contains the full simulation pipeline inside itself.

Key goals
---------
- simulate S distinct signals (S sensors)
- simulate more than one period
- keep one YAML-based entry point
- inspect the SFC stack in a controlled/debuggable way

IMPORTANT
---------
This file intentionally contains the whole pipeline instead of calling the
figure pipeline, so that multi-signal and multi-period debugging can be done
without ambiguity.

Data conventions
----------------
Signals are handled as tensors with shape:

    (time, periods, sensors)

For example:

    x_filtered.shape = (T_samples, n_periods, S)

The phase layer uses:

    ta.shape = (n_periods, N, S)
    tb.shape = (n_periods, N, S)

The SFC event matrix uses:

    events.shape = (event_slots_total, 2 * N * S)

Physical convention
-------------------
The default physical model is:

    P  = average available transmit power per sensor
    N0 = noise spectral-density / noise parameter

Then:

    SNR_total = P / (B * N0)

For communication-budget-based per-sensor methods, such as the Nyquist
Benchmark and RbCP:

    B_s = alpha_s * B
    SNR_s = P / (B_s * N0)

For SFC and RbCP_time, the time/event model uses the total B, not B_s.
"""

import copy
import yaml
import numpy as np
import matplotlib.pyplot as plt

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.nyquist import Nyquist
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.system_parameters import (
    build_derived_system_parameters,
)

from sfc.core.theory import (
    compute_benchmark_bits_per_sample_single_sensor,
)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def _generate_signals(cfg, rng, num_time_samples, n_periods, S):
    """
    Generate S distinct signals across multiple periods.

    Output shape:
        (num_time_samples, n_periods, S)
    """

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x_raw = rng.uniform(-1, 1, size=(num_time_samples, n_periods, S))

    elif dist == "gaussian":
        x_raw = rng.normal(0, 1, size=(num_time_samples, n_periods, S))

    else:
        raise ValueError("Invalid distribution")

    return x_raw


# =============================================================================
# BENCHMARK
# =============================================================================

def _benchmark_nyquist_multisignal(x_zero_mean, cfg, params, Tt):
    """
    Multi-signal, multi-period Nyquist benchmark.

    IMPORTANT
    ---------
    The benchmark communication budget uses each sensor's own bandwidth slice:

        B_sensor = params.B_per_sensor[s]

    and the corresponding per-sensor SNR:

        SNR_s = params.P / (B_sensor * params.N0)

    Input
    -----
    x_zero_mean : np.ndarray
        Shape:
            (time, periods, sensors)

    Returns
    -------
    np.ndarray
        Reconstructed benchmark signals with the same shape.
    """

    benchmark_cfg = cfg.get("benchmark", {})
    sampling_rate = benchmark_cfg.get("sampling_rate", params.W)
    effective_rate_factor = benchmark_cfg.get("effective_rate_factor", 1.0)
    effective_sampling_rate = effective_rate_factor * sampling_rate

    n_time, n_periods, S = x_zero_mean.shape
    x_benchmark = np.zeros_like(x_zero_mean)

    print(f"[INFO] Benchmark sampling_rate = {sampling_rate}")
    print(f"[INFO] Benchmark effective_rate_factor = {effective_rate_factor}")
    print(f"[INFO] Benchmark effective_sampling_rate = {effective_sampling_rate}")

    for s in range(S):
        B_sensor = float(params.B_per_sensor[s])
        SNR_sensor = float(params.P / (B_sensor * params.N0))
        SNR_sensor_dB = 10.0 * np.log10(max(SNR_sensor, np.finfo(float).tiny))

        bits_int = compute_benchmark_bits_per_sample_single_sensor(
            tau=params.tau,
            B_sensor=params.B_per_sensor[s],
            P=params.P,
            N0=params.N0,
            sampling_rate=sampling_rate,
            force_power_of_two=params.quantization_force_power_of_two,
            rounding_mode=params.quantization_rounding_mode,
        )

        print(
            f"[INFO] Benchmark sensor {s}: "
            f"B_sensor = {B_sensor:.6e}, "
            f"SNR_sensor = {SNR_sensor:.6e}, "
            f"SNR_sensor_dB = {SNR_sensor_dB:.3f}, "
            f"bits/sample = {bits_int}"
        )

        t = np.arange(0, params.tau, Tt)

        for p in range(n_periods):
            nyq_cfg = {
                "T": params.tau,
                "Tt": Tt,
                "sampling_rate": effective_sampling_rate,
                "sensor_nodes": 1,
                "bits_codeword": int(bits_int),
                "snr_dB": 100.0,
                "bandwidth": 1e6,
            }

            nyq = Nyquist(nyq_cfg)

            xs = nyq(
                x_zero_mean[:, p, s],
                t,
                quantize=False
            )

            xs_q = nyq.quantize(xs)

            x_benchmark[:, p, s] = nyq.recover_signal(xs_q)

    return x_benchmark


# =============================================================================
# SENSOR-EVENT ASSOCIATION
# =============================================================================

def _build_sensor_x_event(S, N):
    """
    Build the sensor-event association matrix.

    Event ordering assumed by the phase layer:
    for each sensor s:
        [ta events for N harmonics][tb events for N harmonics]

    Total number of event IDs:
        2 * N * S
    """

    num_event_ids = 2 * N * S
    sensor_x_event = np.zeros((S, num_event_ids))

    for s in range(S):
        start = 2 * s * N
        stop = 2 * (s + 1) * N
        sensor_x_event[s, start:stop] = 1.0

    return sensor_x_event


# =============================================================================
# SFC BRANCH
# =============================================================================

def _run_sfc_branch(ta, tb, cfg, params, t, n_periods):
    """
    Run the SFC branch for multiple periods and multiple sensors.

    Input
    -----
    ta, tb : np.ndarray
        Shapes:
            (n_periods, N, S)

    Returns
    -------
    tuple
        x_sfc : np.ndarray
            Shape:
                (time, periods, sensors)

        sfc_debug : dict
            Debug information
    """

    N = ta.shape[1]
    S = ta.shape[2]
    w0 = 2 * np.pi / params.tau

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=S,
        bandwidth=params.B,
        detect_errors=False,
        periods=n_periods,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    # ta/tb -> events
    events = phase_core.ta_tb_to_events(ta, tb)

    cfg_sfc = copy.deepcopy(cfg)

    if "channel" not in cfg_sfc:
        cfg_sfc["channel"] = {}

    cfg_sfc["channel"]["sensor_x_event"] = _build_sensor_x_event(S, N)

    cfg_sfc["channel"].setdefault("collision_mode", "sum")
    cfg_sfc["channel"].setdefault("type", "awgn")
    cfg_sfc["channel"].setdefault("detection_mode", "threshold")
    cfg_sfc["channel"].setdefault("score_threshold", params.L)

    if "threshold" not in cfg_sfc["channel"]:
        cfg_sfc["channel"].setdefault("threshold_factor", 0.5)

    sfc_channel = SFCChannel(cfg_sfc)

    out = sfc_channel(events, return_intermediates=True)
    events_est = out["events_est"]

    # events -> ta/tb
    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)

    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    # reconstruct per period/sensor
    n_time = len(t)
    x_sfc = np.zeros((n_time, n_periods, S))

    for p in range(n_periods):
        for s in range(S):
            x_sfc[:, p, s] = recover_signal(
                ta_rec[p, :, s],
                tb_rec[p, :, s],
                t,
                w0
            )

    sfc_debug = {
        "events": events,
        "events_est": events_est,
        "ta_rec": ta_rec,
        "tb_rec": tb_rec,
        "channel_intermediates": out,
    }

    return x_sfc, sfc_debug


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_debug_from_yaml(config_path):
    """
    Run the full multi-signal, multi-period debug experiment from YAML.
    """

    cfg = load_config(config_path)
    params = build_derived_system_parameters(cfg)

    rng = np.random.default_rng(cfg["reproducibility"]["seed"])

    n_periods = cfg["signal"].get("n_periods", 2)
    if n_periods <= 1:
        raise ValueError("This debug requires n_periods > 1")

    S = params.S
    Tt = cfg["signal"]["Tt"]
    t = np.arange(0, params.tau, Tt)

    print("[INFO] Running debug from YAML:", config_path)
    print(f"[INFO] S = {S}")
    print(f"[INFO] n_periods = {n_periods}")
    print(f"[INFO] N = {params.N}")
    print(f"[INFO] P = {params.P:.6e}")
    print(f"[INFO] N0 = {params.N0:.6e}")
    print(f"[INFO] B = {params.B:.6e}")
    print(f"[INFO] SNR_total = {params.SNR:.6e}")
    print(f"[INFO] SNR_total_dB = {params.SNR_dB:.3f}")
    print(f"[INFO] SNR_per_sensor = {params.SNR_per_sensor}")
    print(f"[INFO] SNR_per_sensor_dB = {params.SNR_per_sensor_dB}")
    print(f"[INFO] M_RbCP = {params.M_rbcp}")
    print(f"[INFO] M_RbCP per sensor = {params.M_rbcp_per_sensor}")
    print(f"[INFO] bandwidth_allocation = {params.bandwidth_allocation}")
    print(f"[INFO] B_per_sensor = {params.B_per_sensor}")

    # -------------------------------------------------------------------------
    # Optional diagnostics if available in the current DerivedSystemParameters
    # -------------------------------------------------------------------------
    if hasattr(params, "signal_level"):
        print(f"[INFO] signal_level = {params.signal_level:.6e}")

    if hasattr(params, "default_threshold"):
        print(f"[INFO] default_threshold = {params.default_threshold:.6e}")

    # -------------------------------------------------------------------------
    # 1. Generate S signals across multiple periods
    # -------------------------------------------------------------------------
    x_raw = _generate_signals(cfg, rng, len(t), n_periods, S)

    # -------------------------------------------------------------------------
    # 2. Band-limit each (period, sensor)
    # -------------------------------------------------------------------------
    W_eff = 2 * params.N / params.tau
    x_filtered = np.zeros_like(x_raw)

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw[:, p, s],
                W_eff,
                Tt,
                params.tau
            )

    # -------------------------------------------------------------------------
    # 3. Peak-to-peak control after filtering
    # -------------------------------------------------------------------------
    p2p_target = cfg["signal"]["peak_to_peak"]

    if p2p_target != 0:
        for p in range(n_periods):
            for s in range(S):
                current_p2p = np.max(x_filtered[:, p, s]) - np.min(x_filtered[:, p, s])
                if current_p2p != 0:
                    x_filtered[:, p, s] = x_filtered[:, p, s] * (p2p_target / current_p2p)

    # -------------------------------------------------------------------------
    # 4. DC handling
    # -------------------------------------------------------------------------
    dc_enabled = cfg.get("dc", {}).get("enabled", False)
    x_zero_mean = np.zeros_like(x_filtered)

    for p in range(n_periods):
        for s in range(S):
            if not dc_enabled:
                dc = Tt * np.sum(x_filtered[:, p, s]) / params.tau
                x_zero_mean[:, p, s] = x_filtered[:, p, s] - dc
            else:
                x_zero_mean[:, p, s] = x_filtered[:, p, s]

    # -------------------------------------------------------------------------
    # 5. Fourier coefficients for all periods and sensors
    # -------------------------------------------------------------------------
    fourier_core = FourierCoefficientCore(
        T=params.tau,
        harmonics=params.N,
        sensor_nodes=S
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=cfg["signal"]["normalize_dft"],
        norm=cfg["signal"]["normalization_target"]
    )

    print("[INFO] an.shape =", an.shape)
    print("[INFO] bn.shape =", bn.shape)
    print("[INFO] x_used.shape =", x_used.shape)

    # -------------------------------------------------------------------------
    # 6. ta/tb for all periods and sensors
    # -------------------------------------------------------------------------
    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=params.N,
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

    print("[INFO] ta.shape =", ta.shape)
    print("[INFO] tb.shape =", tb.shape)

    # -------------------------------------------------------------------------
    # 7. RbCP direct reconstruction
    # -------------------------------------------------------------------------
    ta_q, tb_q = quantize_ta_tb(ta, tb, 2 * np.pi / params.tau, params.M_rbcp)

    x_rbcp = np.zeros_like(x_used)

    for p in range(n_periods):
        for s in range(S):
            x_rbcp[:, p, s] = recover_signal(
                ta_q[p, :, s],
                tb_q[p, :, s],
                t,
                2 * np.pi / params.tau
            )

    # -------------------------------------------------------------------------
    # 8. SFC branch (multi-signal, multi-period)
    # -------------------------------------------------------------------------
    x_sfc, sfc_debug = _run_sfc_branch(
        ta=ta,   # IMPORTANT: non-quantized ta/tb
        tb=tb,
        cfg=cfg,
        params=params,
        t=t,
        n_periods=n_periods
    )

    # -------------------------------------------------------------------------
    # 9. Benchmark branch
    # -------------------------------------------------------------------------
    x_benchmark = _benchmark_nyquist_multisignal(
        x_zero_mean=x_zero_mean,
        cfg=cfg,
        params=params,
        Tt=Tt
    )

    data = {
        "t": t,
        "x_raw": x_raw,
        "x_filtered": x_filtered,
        "x_zero_mean": x_zero_mean,
        "x_used": x_used,
        "x_rbcp": x_rbcp,
        "x_sfc": x_sfc,
        "x_benchmark": x_benchmark,
        "ta": ta,
        "tb": tb,
        "ta_q": ta_q,
        "tb_q": tb_q,
        "sfc_debug": sfc_debug,
        "params": params,
        "cfg": cfg,
        "n_periods": n_periods,
    }

    print_debug_summary(data)
    plot_debug_results(data)

    return data


# =============================================================================
# DEBUG SUMMARY
# =============================================================================

def print_debug_summary(data):
    """
    Print per-period/per-sensor MSE diagnostics.
    """

    x_zero_mean = data["x_zero_mean"]
    x_rbcp = data["x_rbcp"]
    x_sfc = data["x_sfc"]
    x_benchmark = data["x_benchmark"]

    n_periods = x_zero_mean.shape[1]
    S = x_zero_mean.shape[2]

    print("\n[DEBUG SUMMARY]")

    mse_rbcp = np.zeros((n_periods, S))
    mse_sfc = np.zeros((n_periods, S))
    mse_benchmark = np.zeros((n_periods, S))

    for p in range(n_periods):
        for s in range(S):
            mse_rbcp[p, s] = np.mean((x_zero_mean[:, p, s] - x_rbcp[:, p, s]) ** 2)
            mse_sfc[p, s] = np.mean((x_zero_mean[:, p, s] - x_sfc[:, p, s]) ** 2)
            mse_benchmark[p, s] = np.mean((x_zero_mean[:, p, s] - x_benchmark[:, p, s]) ** 2)

    print("MSE RbCP:\n", mse_rbcp)
    print("MSE SFC:\n", mse_sfc)
    print("MSE Benchmark:\n", mse_benchmark)

    sfc_debug = data["sfc_debug"]
    events = sfc_debug["events"]
    events_est = sfc_debug["events_est"]

    print("\n[SFC EVENTS]")
    print("events.shape     =", events.shape)
    print("events_est.shape =", events_est.shape)
    print("events == events_est ?", np.array_equal(events, events_est))

    diff = np.where(events != events_est)
    print("num mismatches =", len(diff[0]))
    if len(diff[0]) > 0:
        print("mismatch rows =", diff[0][:20])
        print("mismatch cols =", diff[1][:20])


# =============================================================================
# PLOTTING
# =============================================================================

def plot_debug_results(data):
    """
    Plot one selected (period, sensor) view from the multi-signal experiment.
    """

    t = data["t"]
    x_filtered = data["x_filtered"]
    x_zero_mean = data["x_zero_mean"]
    x_rbcp = data["x_rbcp"]
    x_sfc = data["x_sfc"]
    x_benchmark = data["x_benchmark"]
    sfc_debug = data["sfc_debug"]
    cfg = data["cfg"]

    plot_cfg = cfg.get("plot", {})
    period_idx = plot_cfg.get("debug_period", 0)
    sensor_idx = plot_cfg.get("debug_sensor", 0)

    print(f"\n[PLOT] using period={period_idx}, sensor={sensor_idx}")

    # -------------------------------------------------------------------------
    # MAIN SIGNAL PLOT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(t, x_filtered[:, period_idx, sensor_idx], "-", label="Band-limited signal", linewidth=2)
    plt.plot(t, x_zero_mean[:, period_idx, sensor_idx], "-", label="Zero-mean signal", linewidth=2)
    plt.plot(t, x_rbcp[:, period_idx, sensor_idx], "--", label="RbCP reconstruction", linewidth=2)
    plt.plot(t, x_sfc[:, period_idx, sensor_idx], "-.", label="SFC reconstruction", linewidth=2)
    plt.plot(t, x_benchmark[:, period_idx, sensor_idx], ":", label="Nyquist Benchmark", linewidth=2)

    plt.xlabel(r"$t$")
    plt.ylabel("Amplitude")
    plt.title(f"Period {period_idx}, Sensor {sensor_idx}")
    plt.grid(True)
    plt.legend()
    plt.show(block=True)
    plt.close()

    # -------------------------------------------------------------------------
    # ERROR PLOT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 4))

    err_rbcp = x_zero_mean[:, period_idx, sensor_idx] - x_rbcp[:, period_idx, sensor_idx]
    err_sfc = x_zero_mean[:, period_idx, sensor_idx] - x_sfc[:, period_idx, sensor_idx]
    err_bench = x_zero_mean[:, period_idx, sensor_idx] - x_benchmark[:, period_idx, sensor_idx]

    plt.plot(t, err_rbcp, label="RbCP error", linewidth=2)
    plt.plot(t, err_sfc, label="SFC error", linewidth=2)
    plt.plot(t, err_bench, label="Benchmark error", linewidth=2)

    plt.xlabel(r"$t$")
    plt.ylabel("Error")
    plt.title(f"Errors | Period {period_idx}, Sensor {sensor_idx}")
    plt.grid(True)
    plt.legend()
    plt.show(block=True)
    plt.close()

    # -------------------------------------------------------------------------
    # CONSISTENCY CHECKS
    # -------------------------------------------------------------------------
    cfg_local = data["cfg"]

    B = cfg_local["system"]["B"]
    R = cfg_local["system"]["R"]
    L = cfg_local["system"]["L"]
    tau = cfg_local["signal"]["tau"]
    n_periods = cfg_local["signal"]["n_periods"]

    slot_duration = R / B
    slots_per_period = int(np.floor(tau / slot_duration))
    event_slots_total = slots_per_period * n_periods
    rx_slots_total = event_slots_total + L - 1

    print("\n[CONSISTENCY CHECKS]")
    print("slot_duration =", slot_duration)
    print("slots_per_period =", slots_per_period)
    print("event_slots_total =", event_slots_total)
    print("rx_slots_total =", rx_slots_total)

    print("events.shape =", sfc_debug["events"].shape)
    print("events_est.shape =", sfc_debug["events_est"].shape)
    print("superposed.shape =", sfc_debug["channel_intermediates"]["superposed"].shape)
    print("y.shape =", sfc_debug["channel_intermediates"]["y"].shape)

    assert sfc_debug["events"].shape[0] == event_slots_total, \
        "events.shape[0] is inconsistent with slots_per_period * n_periods"

    assert sfc_debug["events_est"].shape[0] == event_slots_total, \
        "events_est.shape[0] must match the number of possible event-start slots"

    assert sfc_debug["channel_intermediates"]["superposed"].shape[0] == rx_slots_total, \
        "superposed.shape[0] must match event_slots_total + L - 1"

    assert sfc_debug["channel_intermediates"]["y"].shape[0] == rx_slots_total, \
        "y.shape[0] must match event_slots_total + L - 1"

    # -------------------------------------------------------------------------
    # SFC INTERNAL PLOTS
    # -------------------------------------------------------------------------
    out = sfc_debug["channel_intermediates"]

    superposed = out["superposed"]
    y = out["y"]

    print("\n[PLOT CHECK]")
    print("superposed.shape =", superposed.shape)
    print("y.shape =", y.shape)

    assert len(superposed.shape) == 2, \
        "superposed must have shape (rx_slots_total, R)"

    assert len(y.shape) == 2, \
        "y must have shape (rx_slots_total, R)"

    assert superposed.shape[0] == rx_slots_total, \
        "superposed.shape[0] must be event_slots_total + L - 1"

    assert y.shape[0] == rx_slots_total, \
        "y.shape[0] must be event_slots_total + L - 1"

    # -------------------------------------------------------------------------
    # CHANNEL INPUT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.imshow(
        np.real(superposed),
        cmap="gray_r",
        aspect="auto",
        origin="lower"
    )
    plt.title("SFC Channel Input (final frame)")
    plt.xlabel("Sub-carrier index")
    plt.ylabel("Discrete simulation slot")
    plt.colorbar(label="Amplitude")
    plt.show(block=True)
    plt.close()

    # -------------------------------------------------------------------------
    # CHANNEL OUTPUT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.imshow(
        np.abs(y),
        cmap="viridis",
        aspect="auto",
        origin="lower"
    )
    plt.title("SFC Channel Output |y| (final frame)")
    plt.xlabel("Sub-carrier index")
    plt.ylabel("Discrete simulation slot")
    plt.colorbar(label="Magnitude")
    plt.show(block=True)
    plt.close()


if __name__ == "__main__":
    run_debug_from_yaml("tests/configs/debug_rbcp_signal_representation.yaml")
