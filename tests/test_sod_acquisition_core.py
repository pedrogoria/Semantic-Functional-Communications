"""
tests/test_sod_acquisition_core.py

Quick manual test for the Send-on-Delta (SoD) acquisition core.

Usage
-----
Python console:
    from scripts.test_sod_core import run_test_sod_core
    out = run_test_sod_core()

Terminal:
    python scripts/test_sod_core.py

What this test does
-------------------
- builds a simple sinusoidal test signal
- runs SoDAcquisitionCore
- prints diagnostics
- optionally plots:
    1. reference signal
    2. reconstructed signal
    3. SoD event locations
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from sfc.core.acquisition.sod import SoDAcquisitionCore


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test_sod_core(
        threshold=0.1,
        transmit_event_times=True,
        quantize_amplitudes=True,
        amplitude_bins=32,
        quantize_times=True,
        time_bins=128,
        reconstruction_mode="zero_order_hold",
        frequency_hz=3.0,
        tau=1.0,
        Tt=0.001,
        do_plot=True
):
    """
    Run a quick validation test for SoDAcquisitionCore.

    Parameters
    ----------
    threshold : float
        SoD threshold.

    transmit_event_times : bool
        If True, payload includes time + amplitude.
        If False, payload includes amplitude only.

    quantize_amplitudes : bool
        Whether to quantize SoD amplitudes.

    amplitude_bins : int
        Number of amplitude quantization bins.

    quantize_times : bool
        Whether to quantize SoD event times.

    time_bins : int
        Number of time quantization bins.

    reconstruction_mode : {"zero_order_hold", "linear"}
        SoD reconstruction rule.

    frequency_hz : float
        Frequency of the sinusoidal test signal.

    tau : float
        Signal duration.

    Tt : float
        Dense time-grid step.

    do_plot : bool
        If True, show plots.

    Returns
    -------
    dict
        Diagnostics from the test.
    """
    # -----------------------------------------------------------------
    # Build a simple 1-period / 1-sensor sinusoid
    # -----------------------------------------------------------------
    t = np.arange(0.0, tau, Tt, dtype=float)
    x_1d = np.sin(2.0 * np.pi * frequency_hz * t)
    x = x_1d[:, None, None]  # (time, periods, sensors)

    sod = SoDAcquisitionCore(
        threshold=threshold,
        transmit_event_times=transmit_event_times,
        quantize_amplitudes=quantize_amplitudes,
        amplitude_bins=amplitude_bins,
        quantize_times=quantize_times,
        time_bins=time_bins,
        reconstruction_mode=reconstruction_mode,
    )

    acq, rec = sod.acquire_and_reconstruct(x=x, t=t)

    key = "period_0_sensor_0"

    x_hat = rec.reconstructed_signal[:, 0, 0]
    mse = float(np.mean((x_1d - x_hat) ** 2))

    event_times = (
        acq.event_times_quantized[key]
        if acq.event_times_quantized is not None
        else acq.event_times[key]
    )

    event_values = (
        acq.event_values_quantized[key]
        if acq.event_values_quantized is not None
        else acq.event_values[key]
    )

    print("\n============================================================")
    print("[TEST] SoD acquisition core")
    print(f"[TEST] threshold = {threshold}")
    print(f"[TEST] transmit_event_times = {transmit_event_times}")
    print(f"[TEST] quantize_amplitudes = {quantize_amplitudes}")
    print(f"[TEST] amplitude_bins = {amplitude_bins}")
    print(f"[TEST] quantize_times = {quantize_times}")
    print(f"[TEST] time_bins = {time_bins}")
    print(f"[TEST] reconstruction_mode = {reconstruction_mode}")
    print(f"[TEST] frequency_hz = {frequency_hz}")
    print(f"[TEST] tau = {tau}")
    print(f"[TEST] Tt = {Tt}")
    print(f"[TEST] number of dense samples = {len(t)}")
    print(f"[TEST] event_count = {acq.event_count[key]}")
    print(f"[TEST] payload_bits = {acq.payload_bits[key]}")
    print(f"[TEST] mse = {mse:.8e}")

    if "bits_per_event" in acq.payload_metadata and key in acq.payload_metadata["bits_per_event"]:
        bits_info = acq.payload_metadata["bits_per_event"][key]
        print(f"[TEST] amplitude_bits_per_event = {bits_info['amplitude_bits']}")
        print(f"[TEST] time_bits_per_event = {bits_info['time_bits']}")
        print(f"[TEST] total_bits_per_event = {bits_info['total_bits']}")

    print(f"[TEST] first 10 event_times = {event_times[:10]}")
    print(f"[TEST] first 10 event_values = {event_values[:10]}")

    if do_plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # ------------------------------------------------------------
        # Reference + reconstructed signal
        # ------------------------------------------------------------
        axes[0].plot(t, x_1d, linewidth=2.0, label="Reference signal")
        axes[0].plot(t, x_hat, "--", linewidth=1.8, label="SoD reconstruction")
        axes[0].set_title("Send-on-Delta reconstruction")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        axes[0].legend()

        # ------------------------------------------------------------
        # Event locations
        # ------------------------------------------------------------
        axes[1].plot(t, x_1d, linewidth=1.5, label="Reference signal")
        axes[1].plot(
            event_times,
            event_values,
            "o",
            markersize=5,
            label="SoD events"
        )
        axes[1].set_title("SoD event locations")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show(block=True)

    return {
        "t": t,
        "x_ref": x_1d,
        "x_hat": x_hat,
        "acq": acq,
        "rec": rec,
        "event_times": event_times,
        "event_values": event_values,
        "mse": mse,
        "event_count": acq.event_count[key],
        "payload_bits": acq.payload_bits[key],
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    run_test_sod_core()

# from tests.test_sod_acquisition_core import run_test_sod_core
#
# out = run_test_sod_core(
#     threshold=0.1,
#     transmit_event_times=True,
#     quantize_amplitudes=True,
#     amplitude_bins=32,
#     quantize_times=True,
#     time_bins=128,
#     reconstruction_mode="zero_order_hold",
#     frequency_hz=3.0,
#     tau=1.0,
#     Tt=0.001,
#     do_plot=True,
# )
