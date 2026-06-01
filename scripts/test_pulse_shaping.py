"""
scripts/test_pulse_shaping.py supported pulse families:scripts/test_pulse_shaping.py
    * rect
    * rc
    * rrc
- applies:
    symbols -> upsample -> pulse shaping -> matched filter -> symbol sampling
- prints diagnostic values
- plots:
    1. pulse shape
    2. TX waveform
    3. matched-filter output with sampled points

Usage
-----
Python console:
    from scripts.test_pulse_shaping import run_test_pulse_shaping
    run_test_pulse_shaping()

Terminal:
    python scripts/test_pulse_shaping.py
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

from sfc.core.modulation.pulse_shaping import (
    generate_pulse,
    pulse_energy,
    pulse_peak,
    tx_rx_shaping_chain,
)


# =============================================================================
# BPSK TEST SYMBOLS
# =============================================================================

def generate_bpsk_symbols(n_symbols=16, seed=42):
    """
    Generate a simple BPSK symbol sequence in {-1, +1}.

    Parameters
    ----------
    n_symbols : int
        Number of symbols.

    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        1D BPSK symbol sequence.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_symbols)
    symbols = 2 * bits - 1
    return symbols.astype(float)


# =============================================================================
# SINGLE PULSE TEST
# =============================================================================

def run_single_pulse_test(
    pulse_kind="rrc",
    n_symbols=16,
    samples_per_symbol=8,
    span_symbols=8,
    rolloff=0.25,
    seed=42,
    do_plot=True
):
    """
    Run one quick validation test for a selected pulse family.

    Parameters
    ----------
    pulse_kind : {"rect", "rc", "rrc"}
        Pulse family.

    n_symbols : int
        Number of symbols.

    samples_per_symbol : int
        Samples per symbol.

    span_symbols : int
        Pulse span in symbols (used by rc/rrc).

    rolloff : float
        Roll-off factor for rc/rrc.

    seed : int
        Random seed for the BPSK symbols.

    do_plot : bool
        If True, show plots.

    Returns
    -------
    dict
        Diagnostics from the test.
    """

    symbols = generate_bpsk_symbols(
        n_symbols=n_symbols,
        seed=seed
    )

    pulse = generate_pulse(
        pulse=pulse_kind,
        samples_per_symbol=samples_per_symbol,
        span_symbols=span_symbols,
        rolloff=rolloff,
        gain=1.0,
        normalize_energy=True
    )

    out = tx_rx_shaping_chain(
        symbols=symbols,
        pulse_kind=pulse_kind,
        samples_per_symbol=samples_per_symbol,
        span_symbols=span_symbols,
        rolloff=rolloff,
        gain=1.0,
        normalize_energy=True,
        tx_mode="full",
        mf_mode="full"
    )

    tx = out["tx"]
    y_mf = out["y_mf"]
    samples = out["samples"]

    mse = np.mean((symbols - np.real(samples)) ** 2)

    print("\n============================================================")
    print(f"[TEST] pulse_kind = {pulse_kind}")
    print(f"[TEST] n_symbols = {n_symbols}")
    print(f"[TEST] samples_per_symbol = {samples_per_symbol}")
    print(f"[TEST] span_symbols = {span_symbols}")
    print(f"[TEST] rolloff = {rolloff}")
    print(f"[TEST] pulse length = {len(pulse)}")
    print(f"[TEST] pulse energy = {pulse_energy(pulse):.8f}")
    print(f"[TEST] pulse peak = {pulse_peak(pulse):.8f}")
    print(f"[TEST] first transmitted symbols = {symbols[:8]}")
    print(f"[TEST] first recovered samples = {np.real(samples[:8])}")
    print(f"[TEST] mse(symbols, recovered_samples) = {mse:.8e}")

    if do_plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        # ------------------------------------------------------------
        # Pulse
        # ------------------------------------------------------------
        axes[0].plot(pulse, linewidth=2)
        axes[0].set_title(f"Pulse shape: {pulse_kind}")
        axes[0].set_xlabel("Sample index")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)

        # ------------------------------------------------------------
        # TX waveform
        # ------------------------------------------------------------
        axes[1].plot(tx, linewidth=1.5)
        axes[1].set_title("Pulse-shaped transmit waveform")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True)

        # ------------------------------------------------------------
        # Matched-filter output + symbol-rate samples
        # ------------------------------------------------------------
        axes[2].plot(y_mf, linewidth=1.2, label="Matched-filter output")

        delay = len(pulse) - 1
        sample_idx = delay + np.arange(len(samples)) * samples_per_symbol
        sample_idx = sample_idx[sample_idx < len(y_mf)]

        axes[2].plot(
            sample_idx,
            np.real(samples[:len(sample_idx)]),
            "o",
            label="Recovered symbol-rate samples"
        )

        axes[2].set_title("Matched-filter output and sampled symbols")
        axes[2].set_xlabel("Sample index")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.show(block=True)

    return {
        "pulse_kind": pulse_kind,
        "symbols": symbols,
        "pulse": pulse,
        "tx": tx,
        "y_mf": y_mf,
        "samples": samples,
        "mse": mse,
    }


# =============================================================================
# RUN ALL QUICK TESTS
# =============================================================================

def run_test_pulse_shaping():
    """
    Run a quick validation over all currently supported pulse families.
    """

    results = []

    for pulse_kind in ["rect", "rc", "rrc"]:
        result = run_single_pulse_test(
            pulse_kind=pulse_kind,
            n_symbols=16,
            samples_per_symbol=8,
            span_symbols=8,
            rolloff=0.25,
            seed=42,
            do_plot=True
        )
        results.append(result)

    print("\n==================== SUMMARY ====================")
    for result in results:
        print(
            f"{result['pulse_kind']:>4s} | "
            f"MSE = {result['mse']:.8e} | "
            f"pulse_energy = {pulse_energy(result['pulse']):.8f}"
        )

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    run_test_pulse_shaping()
