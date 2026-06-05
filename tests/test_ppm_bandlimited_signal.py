"""
tests/test_ppm_bandlimited_signal.py

Quick test for analog-like PPM using a band-limited random signal generated
with the same filtering logic used by the SFC experiments.

This script is a standalone PPM sanity/debug test.

Physical convention
-------------------
Preferred project convention:

    P  = average available transmit power per sensor
    N0 = noise spectral-density / noise parameter

For a PPM/FDMA-style single-sensor branch:

    B_sensor = alpha_s * B
    SNR_s = P / (B_sensor * N0)

This script supports two physical regimes:

1. fixed_P_and_N0
   - P fixed
   - N0 fixed
   - B_sensor fixed
   - SNR_s is derived as:
         SNR_s = P / (B_sensor * N0)

2. fixed_sensor_SNR_and_N0
   - SNR_s fixed
   - N0 fixed
   - B_sensor fixed
   - P is derived as:
         P = SNR_s * B_sensor * N0

Correct PPM timing interpretation
---------------------------------
- The message is sampled at fs_msg.
- Each sample occupies exactly one transmission interval / slot:

      Ts = 1 / fs_msg

- Ts is therefore the quantization / transmission interval of one sample.
- The pulse itself has its own duration:

      Tp < Ts

- The sample value is mapped to the pulse position inside the slot.

Test chain
----------
1. Generate a random signal.
2. Band-limit it using sfc.core.filters.filter_periodic(...).
3. Optionally adjust peak-to-peak.
4. Sample the band-limited signal at fs_msg.
5. Define one transmission interval per sample:
       Ts = 1 / fs_msg
6. Map each sampled amplitude to a pulse position inside its slot.
7. Normalize TX waveform power to P.
8. Add AWGN using:
       SNR_s = P / (B_sensor * N0)
9. Demodulate by detecting the pulse position inside each slot.
10. Recover the sampled amplitudes.
11. Reconstruct the continuous-time signal by interpolation.
12. Plot:
    - message signal
    - PPM modulated signal
    - PPM received signal
    - demodulated samples
    - PSD of TX and RX
    - reconstruction comparison

Usage
-----
Python console:
    from tests.test_ppm_bandlimited_signal import run_test_ppm_bandlimited_signal
    out = run_test_ppm_bandlimited_signal()

Terminal:
    python tests/test_ppm_bandlimited_signal.py
"""

from __future__ import annotations

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

from sfc.core.filters import filter_periodic


# =============================================================================
# PHYSICAL HELPERS
# =============================================================================

def snr_linear_from_db(snr_db: float) -> float:
    """
    Convert SNR from dB to linear scale.
    """
    return 10.0 ** (float(snr_db) / 10.0)


def snr_db_from_linear(snr_linear: float) -> float:
    """
    Convert SNR from linear scale to dB.
    """
    return 10.0 * np.log10(max(float(snr_linear), np.finfo(float).tiny))


def resolve_power_noise_model(
    power_model: str,
    P: float | None,
    N0: float,
    B_sensor: float,
    snr_sensor_db: float | None,
) -> tuple[float, float, float, float]:
    """
    Resolve (P, N0, SNR_sensor_linear, SNR_sensor_dB).

    Parameters
    ----------
    power_model : str
        Supported values:
        - "fixed_P_and_N0"
        - "fixed_sensor_SNR_and_N0"

    P : float | None
        Per-sensor transmit power.

    N0 : float
        Noise spectral-density / noise parameter.

    B_sensor : float
        Sensor bandwidth slice.

    snr_sensor_db : float | None
        Per-sensor SNR in dB, used only in fixed_sensor_SNR_and_N0 mode.

    Returns
    -------
    tuple
        (P_resolved, N0_resolved, SNR_sensor_linear, SNR_sensor_dB)
    """

    if B_sensor <= 0:
        raise ValueError("B_sensor must be positive.")

    if N0 <= 0:
        raise ValueError("N0 must be positive.")

    if power_model == "fixed_P_and_N0":
        if P is None:
            raise ValueError("power_model='fixed_P_and_N0' requires P.")

        if P <= 0:
            raise ValueError("P must be positive.")

        snr_sensor_linear = P / (B_sensor * N0)
        snr_sensor_db_resolved = snr_db_from_linear(snr_sensor_linear)

        return float(P), float(N0), float(snr_sensor_linear), float(snr_sensor_db_resolved)

    if power_model in {
        "fixed_sensor_SNR_and_N0",
        "fixed_SNR_per_sensor_and_N0",
        "fixed_per_sensor_SNR_and_N0",
    }:
        if snr_sensor_db is None:
            raise ValueError(
                f"power_model='{power_model}' requires snr_sensor_db."
            )

        snr_sensor_linear = snr_linear_from_db(snr_sensor_db)
        P_resolved = snr_sensor_linear * B_sensor * N0

        return float(P_resolved), float(N0), float(snr_sensor_linear), float(snr_sensor_db)

    raise ValueError(
        f"Unsupported power_model='{power_model}'. "
        "Use 'fixed_P_and_N0' or 'fixed_sensor_SNR_and_N0'."
    )


# =============================================================================
# BASIC HELPERS
# =============================================================================

def awgn_from_physical_snr(
    signal: np.ndarray,
    snr_sensor_linear: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add AWGN to a real-valued signal using the target per-sensor SNR.

    The actual noise variance is chosen relative to the measured waveform power:

        noise_power = signal_power / SNR_s

    Parameters
    ----------
    signal : np.ndarray
        Real-valued waveform.

    snr_sensor_linear : float
        Target per-sensor SNR in linear scale.

    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Noisy waveform.
    """

    signal = np.asarray(signal, dtype=float)

    if snr_sensor_linear <= 0:
        raise ValueError("snr_sensor_linear must be positive.")

    power_signal = np.mean(signal ** 2)

    if np.isclose(power_signal, 0.0):
        return np.array(signal, copy=True)

    power_noise = power_signal / snr_sensor_linear
    noise = rng.normal(0.0, np.sqrt(power_noise), size=signal.shape)

    return signal + noise


def normalize_waveform_power(
    x: np.ndarray,
    target_power: float
) -> np.ndarray:
    """
    Normalize waveform average power to target_power.

    Parameters
    ----------
    x : np.ndarray
        Input waveform.

    target_power : float
        Desired average power.

    Returns
    -------
    np.ndarray
        Power-normalized waveform.
    """

    if target_power <= 0:
        raise ValueError("target_power must be positive.")

    x = np.asarray(x, dtype=float)
    current_power = np.mean(x ** 2)

    if np.isclose(current_power, 0.0):
        return np.array(x, copy=True)

    return x * np.sqrt(target_power / current_power)


def periodogram_psd(x, fs):
    """
    Simple one-sided periodogram PSD estimate.

    Parameters
    ----------
    x : np.ndarray
        Real signal.

    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    tuple
        (f, Pxx_dB)
    """

    x = np.asarray(x, dtype=float)
    n = len(x)

    w = np.hanning(n)
    xw = x * w

    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1.0 / fs)

    U = np.sum(w ** 2)
    Pxx = (np.abs(X) ** 2) / (fs * U)

    if n > 1:
        Pxx[1:-1] *= 2.0

    Pxx_dB = 10.0 * np.log10(Pxx + 1e-20)
    return f, Pxx_dB


# =============================================================================
# BAND-LIMITED SIGNAL GENERATION
# =============================================================================

def generate_bandlimited_random_signal(
    duration=4.0,
    Tt=0.001,
    W=10.4,
    distribution="uniform",
    peak_to_peak=1.0,
    remove_dc=False,
    seed=42
):
    """
    Generate a band-limited random signal using the same periodic low-pass filter
    logic used by the SFC code.

    Parameters
    ----------
    duration : float
        Signal duration in seconds.

    Tt : float
        Time step of the high-resolution continuous-time representation.

    W : float
        Target signal bandwidth.

    distribution : {"uniform", "gaussian"}
        Raw random-signal distribution before filtering.

    peak_to_peak : float
        Target peak-to-peak after filtering. If 0, no scaling is applied.

    remove_dc : bool
        If True, explicitly remove the mean after filtering.

    seed : int
        Random seed.

    Returns
    -------
    tuple
        (t, x_raw, x_filtered)
    """

    rng = np.random.default_rng(seed)

    t = np.arange(0.0, duration, Tt)

    if distribution == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=len(t))
    elif distribution == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=len(t))
    else:
        raise ValueError("distribution must be 'uniform' or 'gaussian'")

    x_filtered = filter_periodic(x_raw, W, Tt, duration)

    if peak_to_peak != 0:
        current_p2p = np.max(x_filtered) - np.min(x_filtered)
        if current_p2p != 0:
            x_filtered = x_filtered * (peak_to_peak / current_p2p)

    if remove_dc:
        x_filtered = x_filtered - np.mean(x_filtered)

    return t, x_raw, x_filtered


# =============================================================================
# SAMPLING
# =============================================================================

def sample_signal_uniform(t, x, fs_msg):
    """
    Uniformly sample a continuous-time reference waveform.

    Parameters
    ----------
    t : np.ndarray
        Dense reference time grid.

    x : np.ndarray
        Dense reference waveform.

    fs_msg : float
        Message sampling frequency in Hz.

    Returns
    -------
    tuple
        (t_samp, x_samp, Ts)
    """

    if fs_msg <= 0:
        raise ValueError("fs_msg must be positive.")

    Ts = 1.0 / fs_msg

    t_samp = np.arange(0.0, t[-1] + (t[1] - t[0]) / 2.0, Ts)
    t_samp = t_samp[t_samp <= t[-1] + 1e-12]

    x_samp = np.interp(t_samp, t, x)

    return t_samp, x_samp, Ts


# =============================================================================
# PPM MODULATION / DEMODULATION
# =============================================================================

def ppm_modulate_samples(
    x_samp,
    fs_msg,
    fs_tx,
    pulse_width=None,
    pulse_width_ratio=0.15,
    x_min_ref=None,
    x_max_ref=None,
    pulse_amplitude=1.0
):
    """
    Encode sampled amplitudes into PPM pulse positions.

    Timing interpretation
    ---------------------
    One sample -> one slot.

    Slot duration:

        Ts = 1 / fs_msg

    The pulse width is independent of Ts, but must satisfy:

        0 < Tp < Ts

    If pulse_width is None:

        Tp = pulse_width_ratio * Ts

    The sample is normalized to [0, 1] using:

        a = (x - x_min_ref) / (x_max_ref - x_min_ref)

    Then mapped to pulse delay:

        delay = a * (Ts - Tp)

    Returns
    -------
    dict
        {
            "t_tx": ...,
            "x_ppm": ...,
            "slot_edges": ...,
            "delays": ...,
            "x_min_ref": ...,
            "x_max_ref": ...,
            "Ts": ...,
            "Tp": ...,
            "sps_slot": ...,
            "pw_samp": ...
        }
    """

    x_samp = np.asarray(x_samp, dtype=float)

    if fs_msg <= 0:
        raise ValueError("fs_msg must be positive.")

    if fs_tx <= 0:
        raise ValueError("fs_tx must be positive.")

    Ts = 1.0 / fs_msg

    if pulse_width is None:
        pulse_width = pulse_width_ratio * Ts

    if pulse_width <= 0:
        raise ValueError("pulse_width must be positive")

    if pulse_width >= Ts:
        raise ValueError("pulse_width must be smaller than the slot duration Ts")

    sps_slot = int(round(Ts * fs_tx))
    pw_samp = max(1, int(round(pulse_width * fs_tx)))

    if sps_slot < 2:
        raise ValueError(
            "fs_tx is too small relative to fs_msg. "
            "Need at least two samples per PPM slot."
        )

    if pw_samp >= sps_slot:
        raise ValueError(
            "pulse_width is too large after discretization. "
            "Need pw_samp < sps_slot."
        )

    n_slots = len(x_samp)
    x_len = n_slots * sps_slot
    x_ppm = np.zeros(x_len, dtype=float)

    if x_min_ref is None:
        x_min_ref = float(np.min(x_samp))

    if x_max_ref is None:
        x_max_ref = float(np.max(x_samp))

    if np.isclose(x_max_ref, x_min_ref):
        a_norm = np.zeros_like(x_samp)
    else:
        a_norm = (x_samp - x_min_ref) / (x_max_ref - x_min_ref)

    a_norm = np.clip(a_norm, 0.0, 1.0)

    max_delay = Ts - pulse_width
    delays = a_norm * max_delay

    for k, delay_k in enumerate(delays):
        start_slot = k * sps_slot
        delay_samp = int(round(delay_k * fs_tx))

        start_pulse = start_slot + delay_samp
        end_pulse = min(start_pulse + pw_samp, start_slot + sps_slot)

        x_ppm[start_pulse:end_pulse] = pulse_amplitude

    t_tx = np.arange(x_len) / fs_tx
    slot_edges = np.arange(n_slots + 1) * Ts

    return {
        "t_tx": t_tx,
        "x_ppm": x_ppm,
        "slot_edges": slot_edges,
        "delays": delays,
        "x_min_ref": x_min_ref,
        "x_max_ref": x_max_ref,
        "Ts": Ts,
        "Tp": pulse_width,
        "sps_slot": sps_slot,
        "pw_samp": pw_samp,
    }


def ppm_demodulate_samples(
    y_rx,
    fs_msg,
    fs_tx,
    pulse_width,
    x_min_ref,
    x_max_ref
):
    """
    Demodulate PPM by estimating the pulse start position inside each slot.

    Detection rule
    --------------
    For each slot:
    - apply a rectangular matched detector using a valid moving sum
    - find the pulse-start index that maximizes the detector output
    - map the estimated delay back to amplitude

    Returns
    -------
    dict
        {
            "x_hat": ...,
            "delay_hat": ...,
            "slot_peak_values": ...
        }
    """

    y_rx = np.asarray(y_rx, dtype=float)

    if fs_msg <= 0:
        raise ValueError("fs_msg must be positive.")

    if fs_tx <= 0:
        raise ValueError("fs_tx must be positive.")

    Ts = 1.0 / fs_msg
    sps_slot = int(round(Ts * fs_tx))
    pw_samp = max(1, int(round(pulse_width * fs_tx)))
    max_delay = Ts - pulse_width

    if sps_slot < 2:
        raise ValueError("Invalid sps_slot. Increase fs_tx or decrease fs_msg.")

    if pw_samp >= sps_slot:
        raise ValueError("Invalid pw_samp. Pulse width is too large.")

    n_slots = len(y_rx) // sps_slot

    x_hat = np.zeros(n_slots, dtype=float)
    delay_hat = np.zeros(n_slots, dtype=float)
    slot_peak_values = np.zeros(n_slots, dtype=float)

    kernel = np.ones(pw_samp, dtype=float)

    for k in range(n_slots):
        start = k * sps_slot
        stop = start + sps_slot

        slot = y_rx[start:stop]

        # Valid matched-filter output. Index corresponds to pulse start.
        metric = np.convolve(slot, kernel, mode="valid")

        idx_peak = int(np.argmax(metric))
        slot_peak_values[k] = metric[idx_peak]

        delay = idx_peak / fs_tx
        delay = np.clip(delay, 0.0, max_delay)

        delay_hat[k] = delay

        if np.isclose(max_delay, 0.0):
            a_hat = 0.0
        else:
            a_hat = delay / max_delay

        a_hat = np.clip(a_hat, 0.0, 1.0)
        x_hat[k] = x_min_ref + a_hat * (x_max_ref - x_min_ref)

    return {
        "x_hat": x_hat,
        "delay_hat": delay_hat,
        "slot_peak_values": slot_peak_values,
    }


# =============================================================================
# RECONSTRUCTION
# =============================================================================

def reconstruct_from_samples(t_ref, t_samp, x_samp_hat):
    """
    Reconstruct a dense waveform from sample values using linear interpolation.
    """
    return np.interp(t_ref, t_samp, x_samp_hat)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_time_domain_chain(
    t,
    x,
    t_samp,
    x_samp,
    t_tx,
    x_ppm,
    y_rx,
    x_samp_hat,
    slot_edges,
    time_xlim=None
):
    """
    Plot the 4-panel time-domain figure.
    """

    fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=False)

    axes[0].plot(t, x, linewidth=1.0)
    axes[0].plot(t_samp, x_samp, "x", markersize=4)
    axes[0].set_title("Message Signal")
    axes[0].set_xlim(time_xlim if time_xlim is not None else (t[0], t[-1]))
    axes[0].grid(True)

    axes[1].plot(t_tx, x_ppm, linewidth=1.0)
    for edge in slot_edges[:-1]:
        axes[1].axvline(edge, color="red", alpha=0.35, linewidth=1.2)
    axes[1].set_title("PPM Modulated Signal")
    axes[1].set_xlim(time_xlim if time_xlim is not None else (t_tx[0], t_tx[-1]))
    axes[1].grid(True)

    axes[2].plot(t_tx, y_rx, linewidth=1.0)
    for edge in slot_edges[:-1]:
        axes[2].axvline(edge, color="red", alpha=0.35, linewidth=1.2)
    axes[2].set_title("PPM Received Signal")
    axes[2].set_xlim(time_xlim if time_xlim is not None else (t_tx[0], t_tx[-1]))
    axes[2].grid(True)

    axes[3].plot(
        t_samp,
        x_samp,
        "o",
        label="Original",
        markersize=5,
        markerfacecolor="none"
    )
    axes[3].plot(
        t_samp[:len(x_samp_hat)],
        x_samp_hat,
        "x",
        label="Recovered",
        markersize=5
    )
    axes[3].set_title("PPM Demod")
    axes[3].set_xlim(time_xlim if time_xlim is not None else (t_samp[0], t_samp[-1]))
    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()
    plt.show(block=True)


def plot_psd(tx, rx, fs_tx, fmax=None):
    """
    Plot TX and RX PSDs in two stacked panels.
    """

    f_tx, Ptx = periodogram_psd(tx, fs_tx)
    f_rx, Prx = periodogram_psd(rx, fs_tx)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(f_tx, Ptx, linewidth=1.0)
    axes[0].set_title("Power Spectral Density (PSD) - PPM TX")
    axes[0].set_ylabel("Power / Frequency (dB/Hz)")
    axes[0].grid(True)

    axes[1].plot(f_rx, Prx, color="red", linewidth=1.0)
    axes[1].set_title("Power Spectral Density (PSD) - PPM RX")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power / Frequency (dB/Hz)")
    axes[1].grid(True)

    if fmax is not None:
        axes[0].set_xlim(0.0, fmax)
        axes[1].set_xlim(0.0, fmax)

    plt.tight_layout()
    plt.show(block=True)


def plot_reconstruction(t, x, x_hat, time_xlim=None):
    """
    Plot original vs reconstructed waveform.
    """

    plt.figure(figsize=(16, 7))
    plt.plot(t, x, linewidth=1.5, label="Original")
    plt.plot(t, x_hat, "--", linewidth=1.2, label="Recovered")
    plt.title("Reconstruction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.xlim(time_xlim if time_xlim is not None else (t[0], t[-1]))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test_ppm_bandlimited_signal(
    duration=1.0,
    Tt=0.001,
    W=10.0,
    distribution="gaussian",
    peak_to_peak=12.0,
    remove_dc=False,
    fs_msg=20.0,
    fs_tx=1000.0,
    pulse_width=None,
    pulse_width_ratio=0.15,
    pulse_amplitude=1.0,
    power_model="fixed_P_and_N0",
    P=1.0,
    N0=1e-3,
    B_sensor=1000.0,
    snr_sensor_db=20.0,
    seed=47,
    plot_fmax=500.0
):
    """
    Run the complete PPM band-limited signal test.

    Parameters
    ----------
    duration : float
        Signal duration in seconds.

    Tt : float
        Time step for dense waveform generation.

    W : float
        Bandwidth of the band-limited random signal.

    distribution : {"uniform", "gaussian"}
        Random source distribution.

    peak_to_peak : float
        Target peak-to-peak after filtering.

    remove_dc : bool
        Whether to remove DC after filtering.

    fs_msg : float
        Message sampling frequency in Hz.
        One slot per sample:
            Ts = 1 / fs_msg

    fs_tx : float
        Discrete-time waveform sampling frequency for PPM TX/RX.

    pulse_width : float | None
        Pulse duration Tp in seconds. If None, pulse_width_ratio is used.

    pulse_width_ratio : float
        Used only if pulse_width is None:
            Tp = pulse_width_ratio * Ts

    pulse_amplitude : float
        PPM pulse amplitude before power normalization.

    power_model : str
        "fixed_P_and_N0" or "fixed_sensor_SNR_and_N0".

    P : float | None
        Per-sensor power. Required for fixed_P_and_N0.

    N0 : float
        Noise parameter.

    B_sensor : float
        Sensor bandwidth slice.

    snr_sensor_db : float
        Per-sensor SNR in dB. Used for fixed_sensor_SNR_and_N0.

    seed : int
        Random seed.

    plot_fmax : float
        Maximum frequency shown in the PSD plots.

    Returns
    -------
    dict
        Main intermediate objects and diagnostics.
    """

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    # 0. Resolve physical budget
    # -----------------------------------------------------------------
    P_resolved, N0_resolved, snr_sensor_linear, snr_sensor_db_resolved = (
        resolve_power_noise_model(
            power_model=power_model,
            P=P,
            N0=N0,
            B_sensor=B_sensor,
            snr_sensor_db=snr_sensor_db
        )
    )

    # -----------------------------------------------------------------
    # 1. Generate band-limited random signal
    # -----------------------------------------------------------------
    t, x_raw, x = generate_bandlimited_random_signal(
        duration=duration,
        Tt=Tt,
        W=W,
        distribution=distribution,
        peak_to_peak=peak_to_peak,
        remove_dc=remove_dc,
        seed=seed
    )

    # -----------------------------------------------------------------
    # 2. Sample the message signal
    # -----------------------------------------------------------------
    t_samp, x_samp, Ts = sample_signal_uniform(
        t=t,
        x=x,
        fs_msg=fs_msg
    )

    # -----------------------------------------------------------------
    # 3. PPM modulation
    # -----------------------------------------------------------------
    ppm_tx = ppm_modulate_samples(
        x_samp=x_samp,
        fs_msg=fs_msg,
        fs_tx=fs_tx,
        pulse_width=pulse_width,
        pulse_width_ratio=pulse_width_ratio,
        x_min_ref=float(np.min(x_samp)),
        x_max_ref=float(np.max(x_samp)),
        pulse_amplitude=pulse_amplitude
    )

    t_tx = ppm_tx["t_tx"]
    x_ppm_raw = ppm_tx["x_ppm"]
    slot_edges = ppm_tx["slot_edges"]
    Tp = ppm_tx["Tp"]

    # -----------------------------------------------------------------
    # 4. Normalize TX waveform power to P
    # -----------------------------------------------------------------
    x_ppm = normalize_waveform_power(
        x=x_ppm_raw,
        target_power=P_resolved
    )

    tx_power = float(np.mean(x_ppm ** 2))

    # -----------------------------------------------------------------
    # 5. AWGN using per-sensor SNR
    # -----------------------------------------------------------------
    y_rx = awgn_from_physical_snr(
        signal=x_ppm,
        snr_sensor_linear=snr_sensor_linear,
        rng=rng
    )

    # -----------------------------------------------------------------
    # 6. Demodulation
    # -----------------------------------------------------------------
    ppm_rx = ppm_demodulate_samples(
        y_rx=y_rx,
        fs_msg=fs_msg,
        fs_tx=fs_tx,
        pulse_width=Tp,
        x_min_ref=ppm_tx["x_min_ref"],
        x_max_ref=ppm_tx["x_max_ref"]
    )

    x_samp_hat = ppm_rx["x_hat"]

    # -----------------------------------------------------------------
    # 7. Dense-time reconstruction
    # -----------------------------------------------------------------
    x_hat = reconstruct_from_samples(
        t_ref=t,
        t_samp=t_samp[:len(x_samp_hat)],
        x_samp_hat=x_samp_hat
    )

    # -----------------------------------------------------------------
    # 8. Diagnostics
    # -----------------------------------------------------------------
    mse_samples = float(np.mean((x_samp[:len(x_samp_hat)] - x_samp_hat) ** 2))
    mse_reconstruction = float(np.mean((x - x_hat) ** 2))

    print("\n============================================================")
    print("[PPM TEST]")
    print(f"duration = {duration}")
    print(f"Tt = {Tt}")
    print(f"W = {W}")
    print(f"distribution = {distribution}")
    print(f"peak_to_peak = {peak_to_peak}")
    print(f"remove_dc = {remove_dc}")
    print(f"fs_msg = {fs_msg}")
    print(f"fs_tx = {fs_tx}")
    print(f"Ts = {Ts}")
    print(f"Tp = {Tp}")
    print(f"pulse_width_ratio = {Tp / Ts:.6f}")
    print(f"pulse_amplitude_raw = {pulse_amplitude}")
    print(f"power_model = {power_model}")
    print(f"P_resolved = {P_resolved:.8e}")
    print(f"N0_resolved = {N0_resolved:.8e}")
    print(f"B_sensor = {B_sensor:.8e}")
    print(f"SNR_sensor_linear = {snr_sensor_linear:.8e}")
    print(f"SNR_sensor_dB = {snr_sensor_db_resolved:.8f}")
    print(f"tx_power_after_normalization = {tx_power:.8e}")
    print(f"num_message_samples = {len(x_samp)}")
    print(f"mse_samples = {mse_samples:.8e}")
    print(f"mse_reconstruction = {mse_reconstruction:.8e}")

    # -----------------------------------------------------------------
    # 9. Plots
    # -----------------------------------------------------------------
    time_xlim = (0.0, duration)

    plot_time_domain_chain(
        t=t,
        x=x,
        t_samp=t_samp,
        x_samp=x_samp,
        t_tx=t_tx,
        x_ppm=x_ppm,
        y_rx=y_rx,
        x_samp_hat=x_samp_hat,
        slot_edges=slot_edges,
        time_xlim=time_xlim
    )

    plot_psd(
        tx=x_ppm,
        rx=y_rx,
        fs_tx=fs_tx,
        fmax=plot_fmax
    )

    plot_reconstruction(
        t=t,
        x=x,
        x_hat=x_hat,
        time_xlim=time_xlim
    )

    return {
        "t": t,
        "x_raw": x_raw,
        "x": x,
        "t_samp": t_samp,
        "x_samp": x_samp,
        "t_tx": t_tx,
        "x_ppm_raw": x_ppm_raw,
        "x_ppm": x_ppm,
        "y_rx": y_rx,
        "x_samp_hat": x_samp_hat,
        "x_hat": x_hat,
        "Ts": Ts,
        "Tp": Tp,
        "slot_edges": slot_edges,
        "mse_samples": mse_samples,
        "mse_reconstruction": mse_reconstruction,
        "P": P_resolved,
        "N0": N0_resolved,
        "B_sensor": B_sensor,
        "SNR_sensor_linear": snr_sensor_linear,
        "SNR_sensor_dB": snr_sensor_db_resolved,
        "tx_power": tx_power,
        "ppm_tx": ppm_tx,
        "ppm_rx": ppm_rx,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    run_test_ppm_bandlimited_signal()
