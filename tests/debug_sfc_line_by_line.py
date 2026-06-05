"""
tests/debug_sfc_line_by_line.py

Debug/helper script for the SFC branch only.

Purpose
-------
Provide a single, saved Python file that executes the SFC path step by step:

    signal -> ta/tb -> events -> SFCChannel(events) -> events_est
           -> recovered ta/tb -> reconstructed x_sfc

This file is intended for use from the PyCharm Python Console or as a script.

IMPORTANT
---------
- The SFC branch is intentionally fed with NON-QUANTIZED ta/tb.
- The signal is filtered, optional peak-to-peak scaling is applied after
  filtering, DC can be removed, Fourier normalization is preserved, and then
  ta/tb are converted to events.
- The SFC stack is then applied.
- SFC uses the total system bandwidth B, not per-sensor bandwidth B_s.

Physical convention
-------------------
Preferred project convention:

    P  = average available transmit power per sensor
    N0 = noise spectral-density / noise parameter

Then:

    SNR_total = P / (B * N0)

This debug helper still accepts SNR_dB for convenience. If N0 is not explicitly
provided, N0 is derived from P, B, and SNR_dB:

    N0 = P / (B * SNR_total)

Recommended usage from PyCharm console
--------------------------------------
from tests.debug_sfc_line_by_line import run_sfc_line_by_line_debug

out = run_sfc_line_by_line_debug(
    S=1,
    P=1.0,
    N0=None,
    B=14.0,
    R=12,
    L=4,
    SNR_dB=-16.0,
    W=10.0,
    tau=1.0,
    Tt=0.01,
    peak_to_peak=8.0,
    distribution="uniform",
    channel_type="clean",
    threshold=None,
    threshold_factor=0.5,
    score_threshold=None,
    plot=True,
    seed=12345,
)

Returned dictionary keys
------------------------
- cfg_sfc
- t
- x_raw
- x_filtered
- x_zero_mean
- x_used
- ta
- tb
- events
- events_est
- ta_rec
- tb_rec
- x_sfc
- mse_sfc
- channel_intermediates
"""

import numpy as np
import matplotlib.pyplot as plt

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import calc_ta_tb, PhaseCoefficientCore
from sfc.core.reconstruction import recover_signal
from sfc.core.system_parameters import compute_M_rbcp
from sfc.core.theory import compute_N
from sfc.core.channel.SFCChannel import SFCChannel


def _snr_linear_from_db(SNR_dB):
    return 10.0 ** (float(SNR_dB) / 10.0)


def _snr_db_from_linear(SNR):
    return 10.0 * np.log10(max(float(SNR), np.finfo(float).tiny))


def _resolve_N0(P, B, SNR_dB, N0):
    """
    Resolve N0 for this standalone debug helper.

    Preferred:
        pass N0 explicitly.

    Backward-compatible:
        if N0 is None, derive it from P, B, and SNR_dB using total-band SNR:

            N0 = P / (B * SNR_total)
    """

    if P <= 0:
        raise ValueError("P must be positive.")

    if B <= 0:
        raise ValueError("B must be positive.")

    if N0 is not None:
        if N0 <= 0:
            raise ValueError("N0 must be positive when provided.")
        return float(N0)

    SNR_total = _snr_linear_from_db(SNR_dB)
    if SNR_total <= 0:
        raise ValueError("SNR_total must be positive.")

    return float(P / (B * SNR_total))


def _build_sensor_x_event(S, N):
    """
    Build the sensor-event association matrix.

    Event ordering:
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


def _prepare_frame_for_plot(arr):
    """
    Return a 2D array suitable for imshow.

    Supports both:
    - final-frame arrays with shape (rx_slots_total, R)
    - older/intermediate arrays with shape (event_slots_total, L, R)

    For 3D input, this helper flattens the first two axes only for debug
    visualization.
    """

    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        a, b, c = arr.shape
        return arr.reshape(a * b, c)

    raise ValueError(f"Expected 2D or 3D array for plotting, got shape={arr.shape}")


def run_sfc_line_by_line_debug(
    S=1,
    P=1.0,
    N0=None,
    B=14000,
    R=12,
    L=4,
    SNR_dB=0,
    W=10.0,
    tau=1.0,
    Tt=0.01,
    peak_to_peak=8.0,
    distribution="uniform",
    normalize_dft=True,
    normalization_target=3.9,
    dc_enabled=False,
    threshold_harmonics=0.001,
    collision_mode="sum",
    channel_type="clean",
    threshold=None,
    threshold_factor=0.5,
    detection_mode="threshold",
    score_threshold=None,
    plot=True,
    seed=12345,
):
    """
    Run the SFC flow step by step for one representative signal.

    Parameters
    ----------
    S, P, N0, B, R, L : system/channel parameters

    SNR_dB : float
        Total-band SNR in dB used only if N0 is None.

    W, tau, Tt : signal/time parameters

    peak_to_peak : float
        Target peak-to-peak value after filtering.
        If 0, keep the original filtered-signal peak-to-peak.

    distribution : {"uniform", "gaussian"}
        Raw-signal distribution.

    normalize_dft : bool
        Keep trusted Fourier normalization logic.

    normalization_target : float
        Target used by the trusted DFT normalization loop.

    dc_enabled : bool
        If False, remove DC before the phase pipeline.

    threshold_harmonics : float
        Threshold passed to the phase core.

    collision_mode, channel_type, threshold, threshold_factor,
    detection_mode, score_threshold
        SFC channel configuration.

    plot : bool
        If True, display diagnostic plots.

    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with all main intermediates and outputs.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    N0 = _resolve_N0(P=P, B=B, SNR_dB=SNR_dB, N0=N0)
    SNR_total = P / (B * N0)
    SNR_total_dB = _snr_db_from_linear(SNR_total)

    # ------------------------------------------------------------------
    # Derived theoretical parameters
    # ------------------------------------------------------------------
    N = compute_N(W, tau)

    # RbCP is not used by the SFC branch, but is useful as a diagnostic.
    # RbCP uses per-sensor B_s and SNR_s internally through P,N0.
    M_rbcp = compute_M_rbcp(
        S=S,
        W=W,
        tau=tau,
        B=B,
        P=P,
        N0=N0,
        bandwidth_allocation=None,
        force_power_of_two=False,
        rounding_mode="floor",
    )

    w0 = 2.0 * np.pi / tau
    n_vec = np.arange(1, N + 1)
    t = np.arange(0.0, tau, Tt)

    print(f"[INFO] P = {P:.6e}")
    print(f"[INFO] N0 = {N0:.6e}")
    print(f"[INFO] B = {B:.6e}")
    print(f"[INFO] SNR_total_dB = {SNR_total_dB:.6f}")
    print(f"[INFO] SNR_total = {SNR_total:.6e}")
    print(f"[INFO] Derived N = {N}")
    print(f"[INFO] Diagnostic M_RbCP = {M_rbcp}")
    print("[INFO] SFC uses total B, not B_s.")

    # ------------------------------------------------------------------
    # 1. Generate raw signal
    # ------------------------------------------------------------------
    if distribution == "uniform":
        x_raw = rng.uniform(-1, 1, len(t))
    elif distribution == "gaussian":
        x_raw = rng.normal(0, 1, len(t))
    else:
        raise ValueError("distribution must be 'uniform' or 'gaussian'")

    # ------------------------------------------------------------------
    # 2. Band-limit signal
    #    Use W_eff consistent with target N:
    #        N = floor(W_eff * tau / 2)
    # ------------------------------------------------------------------
    W_eff = 2.0 * N / tau
    x_filtered = filter_periodic(x_raw, W_eff, Tt, tau)

    # ------------------------------------------------------------------
    # 3. Optional peak-to-peak control after filtering
    # ------------------------------------------------------------------
    current_p2p = np.max(x_filtered) - np.min(x_filtered)
    if peak_to_peak != 0 and current_p2p != 0:
        x_filtered = x_filtered * (peak_to_peak / current_p2p)

    print(f"[INFO] Filtered peak-to-peak = {np.max(x_filtered) - np.min(x_filtered):.6e}")

    # ------------------------------------------------------------------
    # 4. DC handling
    # ------------------------------------------------------------------
    dc_value = Tt * np.sum(x_filtered) / tau
    if not dc_enabled:
        x_zero_mean = x_filtered - dc_value
    else:
        x_zero_mean = x_filtered.copy()

    print(f"[INFO] DC removed = {not dc_enabled}")
    print(f"[INFO] DC value = {dc_value:.6e}")

    # ------------------------------------------------------------------
    # 5. Fourier coefficients with trusted normalization logic
    # ------------------------------------------------------------------
    fourier_core = FourierCoefficientCore(
        T=tau,
        harmonics=N,
        sensor_nodes=1,
    )

    an, bn, x_used = fourier_core.calc_an_bn_dft(
        x_zero_mean,
        Tt,
        normalize=normalize_dft,
        norm=normalization_target,
    )

    x_used_1d = x_used[:, 0, 0]
    max_ab = np.max(an[0, :, 0] ** 2 + bn[0, :, 0] ** 2)
    print(f"[INFO] max(an^2 + bn^2) = {max_ab:.6e}")

    # ------------------------------------------------------------------
    # 6. ta/tb using ONLY the complex-log formula
    # ------------------------------------------------------------------
    ta, tb = calc_ta_tb(an[0, :, 0], bn[0, :, 0], n_vec, w0)
    ta = np.real(ta)
    tb = np.real(tb)

    print("[INFO] ta =", ta)
    print("[INFO] tb =", tb)
    print("[INFO] SFC branch uses NON-QUANTIZED ta/tb")

    # ------------------------------------------------------------------
    # 7. ta/tb -> events
    # ------------------------------------------------------------------
    phase_core = PhaseCoefficientCore(
        T=tau,
        harmonics=N,
        n_sub_symbol=L,
        resource=R,
        sensor_nodes=1,
        bandwidth=B,
        detect_errors=False,
        periods=1,
        threshold_harmonics=threshold_harmonics,
    )

    ta_3d = ta.reshape(1, N, 1)
    tb_3d = tb.reshape(1, N, 1)

    events = phase_core.ta_tb_to_events(ta_3d, tb_3d)
    num_event_ids = events.shape[1]

    print(f"[INFO] events.shape = {events.shape}")
    print(f"[INFO] sum(events) = {np.sum(events):.0f}")

    # ------------------------------------------------------------------
    # 8. Build local cfg for the SFC channel
    #    Representative-signal debug logic:
    #    assign all event IDs to sensor 0.
    # ------------------------------------------------------------------
    sensor_x_event = np.zeros((S, num_event_ids))
    sensor_x_event[0, :] = 1.0

    channel_cfg = {
        "sensor_x_event": sensor_x_event,
        "collision_mode": collision_mode,
        "type": channel_type,
        "detection_mode": detection_mode,
        "score_threshold": L if score_threshold is None else score_threshold,
    }

    if threshold is not None:
        channel_cfg["threshold"] = threshold
    else:
        channel_cfg["threshold_factor"] = threshold_factor

    cfg_sfc = {
        "system": {
            "S": S,
            "P": P,
            "N0": N0,
            "B": B,
            "R": R,
            "L": L,
            "SNR_dB": SNR_total_dB,
        },
        "signal": {
            "tau": tau,
            "W": W,
        },
        "channel": channel_cfg,
        "reproducibility": {
            "seed": seed,
        },
    }

    # ------------------------------------------------------------------
    # 9. Transmit through SFC
    # ------------------------------------------------------------------
    sfc_channel = SFCChannel(cfg_sfc)
    out = sfc_channel(events, return_intermediates=True)
    events_est = out["events_est"]

    print("[INFO] events original:")
    print(out["events"])
    print("[INFO] events estimated:")
    print(events_est)

    true_ids = np.where(out["events"][0] == 1)[0]
    est_ids = np.where(events_est[0] == 1)[0]
    print("[INFO] true_ids =", true_ids)
    print("[INFO] est_ids  =", est_ids)

    # ------------------------------------------------------------------
    # 10. events_est -> ta/tb_est
    # ------------------------------------------------------------------
    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)
    ta_rec = np.real(ta_rec[0, :, 0])
    tb_rec = np.real(tb_rec[0, :, 0])

    print("[INFO] ta recovered =", ta_rec)
    print("[INFO] tb recovered =", tb_rec)
    print(f"[INFO] max |ta - ta_rec| = {np.max(np.abs(ta - ta_rec)):.6e}")
    print(f"[INFO] max |tb - tb_rec| = {np.max(np.abs(tb - tb_rec)):.6e}")
    print("[INFO] 9999 in ta_rec =", np.any(ta_rec == 9999))
    print("[INFO] 9999 in tb_rec =", np.any(tb_rec == 9999))

    # ------------------------------------------------------------------
    # 11. Reconstruct SFC signal
    # ------------------------------------------------------------------
    x_sfc = recover_signal(ta_rec, tb_rec, t, w0)
    mse_sfc = float(np.mean((x_used_1d - x_sfc) ** 2))

    print(f"[INFO] MSE_sfc = {mse_sfc:.6e}")

    # ------------------------------------------------------------------
    # 12. Optional plots
    # ------------------------------------------------------------------
    if plot:
        # Signal comparison
        plt.figure(figsize=(10, 5))
        plt.plot(t, x_used_1d, label="x_used", linewidth=2)
        plt.plot(t, x_sfc, "--", label="x_sfc", linewidth=2)
        plt.xlabel(r"$t$")
        plt.ylabel("Amplitude")
        plt.title("SFC reconstruction from events")
        plt.grid(True)
        plt.legend()
        plt.show(block=True)
        plt.close()

        # Reconstruction error
        plt.figure(figsize=(10, 4))
        plt.plot(t, x_used_1d - x_sfc, color="red")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x_{\mathrm{used}} - x_{\mathrm{sfc}}$")
        plt.title("SFC reconstruction error")
        plt.grid(True)
        plt.show(block=True)
        plt.close()

        superposed_for_plot = _prepare_frame_for_plot(out["superposed"])
        y_for_plot = _prepare_frame_for_plot(out["y"])

        # Aggregate transmitted map
        plt.figure(figsize=(8, 4))
        plt.imshow(np.real(superposed_for_plot), cmap="gray_r", aspect="auto", origin="lower")
        plt.title("Superposed transmitted frame")
        plt.xlabel("Resource index")
        plt.ylabel("Discrete simulation slot")
        plt.colorbar(label="Amplitude")
        plt.show(block=True)
        plt.close()

        # Channel output magnitude
        plt.figure(figsize=(8, 4))
        plt.imshow(np.abs(y_for_plot), cmap="viridis", aspect="auto", origin="lower")
        plt.title("Channel output magnitude |y|")
        plt.xlabel("Resource index")
        plt.ylabel("Discrete simulation slot")
        plt.colorbar(label="Magnitude")
        plt.show(block=True)
        plt.close()

    return {
        "cfg_sfc": cfg_sfc,
        "t": t,
        "x_raw": x_raw,
        "x_filtered": x_filtered,
        "x_zero_mean": x_zero_mean,
        "x_used": x_used_1d,
        "ta": ta,
        "tb": tb,
        "events": events,
        "events_est": events_est,
        "ta_rec": ta_rec,
        "tb_rec": tb_rec,
        "x_sfc": x_sfc,
        "mse_sfc": mse_sfc,
        "channel_intermediates": out,
        "N0": N0,
        "SNR_total": SNR_total,
        "SNR_total_dB": SNR_total_dB,
        "M_rbcp_diagnostic": M_rbcp,
    }


if __name__ == "__main__":
    run_sfc_line_by_line_debug()
