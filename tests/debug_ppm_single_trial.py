"""
tests/debug_ppm_single_trial.py

Debug / visualization script for a single PPM trial using the same modeling
choices adopted by the fair comparison pipeline.

This script:
- loads the comparison YAML
- selects one B value
- generates one common band-limited source realization with the same SFC-style logic
- selects one sensor
- runs the PPM + FDMA branch for that sensor
- plots:
    1. Message Signal (original domain)
    2. Message Signal (normalized PPM domain)
    3. PPM Modulated Signal
    4. PPM Received Signal
    5. PPM Demod (original domain)
    6. Continuous-Time Reconstruction

Important conventions
---------------------
1. Ts = 1 / fs_msg is the transmission / quantization interval of one sample.
2. The pulse duration Tp is independent from Ts, but Tp < Ts.
3. When the PPM branch reconstructs the continuous-time signal, it uses:
       periodic_replicas = 10
   i.e. replicas from -10*tau to +10*tau,
   total of 21 periodic blocks in the truncated periodic sinc reconstruction.

Usage (Python console - PyCharm)
--------------------------------
from tests.debug_ppm_single_trial import debug_ppm_single_trial

out = debug_ppm_single_trial(
    config_path="experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml",
    B_value=2500,
    sensor_idx=0,
    seed=47,
    do_plot=True
)

Terminal
--------
python tests/debug_ppm_single_trial.py --config experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml --B 2500 --sensor 0
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# ---------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sfc.core.filters import filter_periodic
from sfc.core.system_parameters import build_derived_system_parameters
from sfc.core.modulation.ppm import PPMCore
from sfc.core.mac.fdma import FDMACore


# =============================================================================
# GLOBAL RECONSTRUCTION CHOICE
# =============================================================================

PERIODIC_REPLICAS = 10


# =============================================================================
# CONFIG / POWER MODEL
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_power_model_inplace(cfg: dict):
    """
    Resolve the selected physical-budget regime in-place.

    Supported modes
    ---------------
    1. fixed_P_and_N0
       - keep P fixed
       - keep N0 fixed
       - derive SNR_dB(B)

    2. fixed_P_and_SNR
       - keep P fixed
       - keep SNR fixed
       - derive N0(B)
    """
    mode = cfg.get("power_model", {}).get("mode", "fixed_P_and_N0")

    if mode == "fixed_P_and_N0":
        cfg["system"]["SNR_dB"] = derive_snr_db_from_fixed_P_and_N0(cfg)
        return

    if mode == "fixed_P_and_SNR":
        cfg["system"]["N0"] = derive_n0_from_fixed_P_and_snr(cfg)
        return

    raise ValueError(
        f"Unsupported power_model.mode: {mode}. "
        f"Supported modes are: 'fixed_P_and_N0', 'fixed_P_and_SNR'."
    )


def derive_snr_db_from_fixed_P_and_N0(cfg: dict) -> float:
    """
    Derive SNR_dB(B) from fixed P and fixed N0 using:
        SNR(B) = P / (B * N0)
    """
    P = cfg["system"]["P"]
    B = cfg["system"]["B"]
    N0 = cfg["system"]["N0"]

    snr = P / (B * N0)
    return 10.0 * np.log10(snr)


def derive_n0_from_fixed_P_and_snr(cfg: dict) -> float:
    """
    Derive N0(B) from fixed P and fixed SNR using:
        N0(B) = P / (B * SNR)
    """
    P = cfg["system"]["P"]
    B = cfg["system"]["B"]
    SNR_dB = cfg["system"]["SNR_dB"]

    snr = 10.0 ** (SNR_dB / 10.0)
    return P / (B * snr)


# =============================================================================
# COMMON SOURCE GENERATION (SAME LOGIC AS FAIR PIPELINE)
# =============================================================================

def generate_common_bandlimited_signals(cfg: dict, rng: np.random.Generator, params, N: int):
    """
    Generate the common source signals using the same logic as the comparison pipeline.

    Returns
    -------
    dict
        {
            "t": ...,
            "Tt": ...,
            "x_ref": ...   # shape: (time, periods=1, sensors)
        }
    """

    tau = params.tau
    Tt = cfg["signal"]["Tt"]
    S = params.S
    n_periods = 1

    t = np.arange(0.0, tau, Tt)
    n_time = len(t)

    dist = cfg["signal"]["distribution"]

    if dist == "uniform":
        x_raw = rng.uniform(-1.0, 1.0, size=(n_time, n_periods, S))
    elif dist == "gaussian":
        x_raw = rng.normal(0.0, 1.0, size=(n_time, n_periods, S))
    else:
        raise ValueError("Invalid distribution")

    W_eff = 2.0 * N / tau
    x_filtered = np.zeros_like(x_raw)

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw[:, p, s],
                W_eff,
                Tt,
                tau
            )

    peak_to_peak = cfg["signal"]["peak_to_peak"]
    if peak_to_peak != 0:
        for p in range(n_periods):
            for s in range(S):
                current_p2p = np.max(x_filtered[:, p, s]) - np.min(x_filtered[:, p, s])
                if current_p2p != 0:
                    x_filtered[:, p, s] *= peak_to_peak / current_p2p

    dc_enabled = cfg.get("dc", {}).get("enabled", False)
    x_zero_mean = np.zeros_like(x_filtered)

    for p in range(n_periods):
        for s in range(S):
            if not dc_enabled:
                dc = Tt * np.sum(x_filtered[:, p, s]) / tau
                x_zero_mean[:, p, s] = x_filtered[:, p, s] - dc
            else:
                x_zero_mean[:, p, s] = x_filtered[:, p, s]

    return {
        "t": t,
        "Tt": Tt,
        "x_ref": x_zero_mean,
    }


# =============================================================================
# FDMA-AWARE PER-SENSOR AWGN
# =============================================================================

def add_awgn_single_sensor_fdma(
    x: np.ndarray,
    P_per_sensor: float,
    B_sensor: float,
    N0: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """
    Add AWGN to one sensor waveform under the FDMA interpretation.

    Interpretation
    --------------
    For the chosen sensor:
        SNR_s = P_per_sensor / (B_sensor * N0)

    The actual noise variance is chosen to be consistent with the measured
    waveform power and the target SNR_s.

    Parameters
    ----------
    x : np.ndarray
        Shape:
            (time, periods=1, sensors=1)

    Returns
    -------
    tuple
        (y_noisy, snr_sensor_linear)
    """

    if B_sensor <= 0:
        raise ValueError("B_sensor must be > 0")

    snr_s = P_per_sensor / (B_sensor * N0)

    ps = np.mean(x[:, :, 0] ** 2)
    if np.isclose(ps, 0.0):
        return np.array(x, copy=True), snr_s

    pn = ps / snr_s
    noise = rng.normal(0.0, np.sqrt(pn), size=x[:, :, 0].shape)

    y = np.array(x, copy=True)
    y[:, :, 0] = x[:, :, 0] + noise

    return y, snr_s


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def compute_original_samples(t: np.ndarray, x_ref_sensor: np.ndarray, symbol_times: np.ndarray) -> np.ndarray:
    """
    Compute samples directly from the original-domain signal.

    This is what should be plotted against the original waveform.
    """
    return np.interp(symbol_times, t, x_ref_sensor)


def normalize_original_samples_for_plot(
    original_samples: np.ndarray,
    normalization_state
) -> np.ndarray:
    """
    Map original-domain samples into the same normalized PPM domain used internally
    by PPMCore, for visualization purposes only.
    """

    if normalization_state is None or not normalization_state.enabled:
        return original_samples

    x_min = np.asarray(normalization_state.metadata["x_min"], dtype=float)
    x_max = np.asarray(normalization_state.metadata["x_max"], dtype=float)
    eps_margin = float(normalization_state.eps_margin)

    # single period / single sensor in this debug script
    xmin = float(x_min[0, 0])
    xmax = float(x_max[0, 0])

    if np.isclose(xmax, xmin):
        return np.full_like(original_samples, 0.5, dtype=float)

    x01 = (original_samples - xmin) / (xmax - xmin)
    return eps_margin + (1.0 - 2.0 * eps_margin) * x01


def build_normalized_reference_curve(
    x_ref_sensor: np.ndarray,
    normalization_state
) -> np.ndarray:
    """
    Build the normalized version of the original waveform so it can be compared
    honestly against the internally normalized PPM samples.
    """

    if normalization_state is None or not normalization_state.enabled:
        return x_ref_sensor

    x_min = np.asarray(normalization_state.metadata["x_min"], dtype=float)
    x_max = np.asarray(normalization_state.metadata["x_max"], dtype=float)
    eps_margin = float(normalization_state.eps_margin)

    xmin = float(x_min[0, 0])
    xmax = float(x_max[0, 0])

    if np.isclose(xmax, xmin):
        return np.full_like(x_ref_sensor, 0.5, dtype=float)

    x01 = (x_ref_sensor - xmin) / (xmax - xmin)
    return eps_margin + (1.0 - 2.0 * eps_margin) * x01


# =============================================================================
# PLOTTING
# =============================================================================

def plot_ppm_single_trial(
    t: np.ndarray,
    x_ref_sensor: np.ndarray,
    x_ref_sensor_norm: np.ndarray,
    symbol_times: np.ndarray,
    original_samples: np.ndarray,
    sampled_message_norm: np.ndarray,
    pulse_positions: np.ndarray,
    tx_waveform: np.ndarray,
    rx_waveform: np.ndarray,
    recovered_samples: np.ndarray,
    recovered_continuous: np.ndarray,
):
    """
    Plot the main debug views for one PPM trial.

    Important
    ---------
    This plotting function keeps domains separate:
    - original signal domain
    - internal normalized PPM domain
    """

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=False)

    # -----------------------------------------------------------------
    # 1. Message signal (original domain)
    # -----------------------------------------------------------------
    axes[0].plot(t, x_ref_sensor, linewidth=1.2, label="Original signal")
    axes[0].plot(
        symbol_times,
        original_samples,
        "x",
        markersize=5,
        label="Original-domain samples"
    )
    axes[0].set_title("Message Signal (original domain)")
    axes[0].grid(True)
    axes[0].legend()

    # -----------------------------------------------------------------
    # 2. Message signal (normalized PPM domain)
    # -----------------------------------------------------------------
    axes[1].plot(t, x_ref_sensor_norm, linewidth=1.2, label="Normalized signal")
    axes[1].plot(
        symbol_times,
        sampled_message_norm,
        "x",
        markersize=5,
        label="Normalized PPM samples"
    )
    axes[1].set_title("Message Signal (normalized PPM domain)")
    axes[1].grid(True)
    axes[1].legend()

    # -----------------------------------------------------------------
    # 3. PPM TX
    # -----------------------------------------------------------------
    axes[2].plot(t, tx_waveform, linewidth=1.2, label="PPM TX")
    for ts in symbol_times:
        axes[2].axvline(ts, color="red", alpha=0.25, linewidth=1.0)
    axes[2].plot(
        pulse_positions,
        np.interp(pulse_positions, t, tx_waveform),
        "o",
        markersize=3,
        label="Pulse positions"
    )
    axes[2].set_title("PPM Modulated Signal")
    axes[2].grid(True)
    axes[2].legend()

    # -----------------------------------------------------------------
    # 4. PPM RX
    # -----------------------------------------------------------------
    axes[3].plot(t, rx_waveform, linewidth=1.0, label="PPM RX")
    for ts in symbol_times:
        axes[3].axvline(ts, color="red", alpha=0.25, linewidth=1.0)
    axes[3].set_title("PPM Received Signal")
    axes[3].grid(True)
    axes[3].legend()

    # -----------------------------------------------------------------
    # 5. PPM Demod (original domain)
    # -----------------------------------------------------------------
    axes[4].plot(
        symbol_times,
        original_samples,
        "o",
        markersize=5,
        markerfacecolor="none",
        label="Original-domain samples"
    )
    axes[4].plot(
        symbol_times,
        recovered_samples,
        "x",
        markersize=5,
        label="Recovered samples"
    )
    axes[4].set_title("PPM Demod (original domain)")
    axes[4].grid(True)
    axes[4].legend()

    plt.tight_layout()
    plt.show(block=True)

    # -----------------------------------------------------------------
    # Continuous-time reconstruction
    # -----------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(t, x_ref_sensor, linewidth=1.3, label="Original")
    plt.plot(t, recovered_continuous, "--", linewidth=1.2, label="Recovered")
    plt.title("Continuous-Time Reconstruction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


# =============================================================================
# MAIN DEBUG FUNCTION
# =============================================================================

def debug_ppm_single_trial(
    config_path: str,
    B_value: float,
    sensor_idx: int = 0,
    seed: int = 47,
    do_plot: bool = True
):
    """
    Run and visualize one PPM trial consistent with the fair-comparison setup.

    Parameters
    ----------
    config_path : str
        Path to the comparison YAML.

    B_value : float
        Selected B value for the debug run.

    sensor_idx : int, optional
        Sensor index to visualize.

    seed : int, optional
        Random seed.

    do_plot : bool, optional
        If True, show plots.

    Returns
    -------
    dict
        Debug objects and diagnostics.
    """

    cfg = load_config(config_path)
    cfg_B = copy.deepcopy(cfg)
    cfg_B["system"]["B"] = float(B_value)

    apply_power_model_inplace(cfg_B)
    params = build_derived_system_parameters(cfg_B)

    N = cfg_B["signal"].get("N_override", params.N)

    if sensor_idx < 0 or sensor_idx >= params.S:
        raise ValueError(f"sensor_idx must satisfy 0 <= sensor_idx < {params.S}")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1) Common source generation
    # ------------------------------------------------------------
    common = generate_common_bandlimited_signals(
        cfg=cfg_B,
        rng=rng,
        params=params,
        N=N
    )

    t = common["t"]
    x_ref = common["x_ref"]                 # shape: (time, 1, S)
    x_sensor = x_ref[:, :, sensor_idx:sensor_idx + 1]   # shape: (time, 1, 1)

    # ------------------------------------------------------------
    # 2) PPM settings
    # ------------------------------------------------------------
    ppm_cfg = cfg_B.get("ppm", {})
    fdma_cfg = cfg_B.get("fdma", {})

    fs_msg = ppm_cfg.get("fs_msg", params.W)
    pulse_width = ppm_cfg.get("pulse_width", 0.1 / fs_msg)
    pulse_type = ppm_cfg.get("pulse_type", "raised_cosine")
    rolloff = ppm_cfg.get("rolloff", 0.99)
    span = ppm_cfg.get("span", 12)
    eps_margin = ppm_cfg.get("eps_margin", 1e-3)

    ppm_core = PPMCore(
        fc=fs_msg,
        pulse_width=pulse_width,
        rec_pulse=ppm_cfg.get("rec_pulse", 0.0),
        pulse_type=pulse_type,
        rolloff=rolloff,
        span=span,
        eps_margin=eps_margin,
        interp_mode=ppm_cfg.get("interp_mode", "linear"),
        periodic_replicas=PERIODIC_REPLICAS,
        clip_recovered_to_unit_interval=ppm_cfg.get(
            "clip_recovered_to_unit_interval",
            True
        ),
    )

    # ------------------------------------------------------------
    # 3) FDMA fairness object (single debug sensor uses its B slice)
    # ------------------------------------------------------------
    fdma_core = FDMACore(
        S=params.S,
        B_total=params.B,
        P_per_sensor=params.P,
        tau=params.tau,
        bandwidth_allocation=params.bandwidth_allocation,
        normalize_sensor_power=fdma_cfg.get("normalize_sensor_power", True),
        return_nonorthogonal_sum_preview=False,
        frequency_axis_centered_at_zero=fdma_cfg.get(
            "frequency_axis_centered_at_zero",
            True
        ),
    )

    B_sensor = float(fdma_core.get_bandwidth_per_sensor()[sensor_idx])

    # ------------------------------------------------------------
    # 4) PPM modulation
    # ------------------------------------------------------------
    mod_result = ppm_core.modulate(x_sensor, t)

    tx = mod_result.tx_waveform  # shape: (time, 1, 1)

    # Optionally normalize waveform power to the per-sensor budget P
    if fdma_cfg.get("normalize_sensor_power", True):
        ps = np.mean(tx[:, :, 0] ** 2)
        if not np.isclose(ps, 0.0):
            scale = np.sqrt(params.P / ps)
            tx = tx * scale

    # ------------------------------------------------------------
    # 5) AWGN under FDMA interpretation
    # ------------------------------------------------------------
    rx, snr_sensor_linear = add_awgn_single_sensor_fdma(
        x=tx,
        P_per_sensor=params.P,
        B_sensor=B_sensor,
        N0=cfg_B["system"]["N0"],
        rng=rng
    )

    # ------------------------------------------------------------
    # 6) PPM demodulation / reconstruction
    # ------------------------------------------------------------
    demod_result = ppm_core.demodulate(
        y=rx,
        t=t,
        modulation_result=mod_result,
        reconstruct_continuous=True
    )

    # ------------------------------------------------------------
    # 7) Build CORRECT debug-domain objects
    # ------------------------------------------------------------
    symbol_times = mod_result.symbol_times
    pulse_positions = mod_result.aux["pulse_positions"][:, 0, 0]

    # These are INTERNAL normalized samples used by the PPM mapping
    sampled_message_norm = mod_result.sampled_message[:, 0, 0]

    # These are ORIGINAL-domain samples, computed consistently from x_ref
    x_ref_sensor = x_sensor[:, 0, 0]
    original_samples = compute_original_samples(
        t=t,
        x_ref_sensor=x_ref_sensor,
        symbol_times=symbol_times
    )

    # Recovered samples from demodulation are already in the ORIGINAL domain
    recovered_samples = demod_result.recovered_samples[:, 0, 0]
    recovered_continuous = demod_result.recovered_continuous[:, 0, 0]

    # Normalized reference curve (for honest comparison in normalized domain)
    x_ref_sensor_norm = build_normalized_reference_curve(
        x_ref_sensor=x_ref_sensor,
        normalization_state=mod_result.normalization_state
    )

    # Optional: recompute normalized samples directly from original domain
    # so one can cross-check consistency against mod_result.sampled_message
    original_samples_norm = normalize_original_samples_for_plot(
        original_samples=original_samples,
        normalization_state=mod_result.normalization_state
    )

    mse_samples_original = float(np.mean((original_samples - recovered_samples) ** 2))
    mse_samples_normalized_consistency = float(
        np.mean((sampled_message_norm - original_samples_norm) ** 2)
    )
    mse_continuous = float(np.mean((x_ref_sensor - recovered_continuous) ** 2))

    print("\n[DEBUG PPM SINGLE TRIAL]")
    print(f"config_path = {config_path}")
    print(f"B_value = {B_value}")
    print(f"sensor_idx = {sensor_idx}")
    print(f"seed = {seed}")
    print(f"power_model.mode = {cfg_B.get('power_model', {}).get('mode', 'fixed_P_and_N0')}")
    print(f"S = {params.S}")
    print(f"P = {params.P}")
    print(f"B_total = {params.B}")
    print(f"B_sensor = {B_sensor}")
    print(f"N0_derived = {cfg_B['system']['N0']}")
    print(f"SNR_dB_derived = {cfg_B['system']['SNR_dB']}")
    print(f"SNR_sensor_linear = {snr_sensor_linear}")
    print(f"tau = {params.tau}")
    print(f"W = {params.W}")
    print(f"fs_msg = {fs_msg}")
    print(f"pulse_width = {pulse_width}")
    print(f"pulse_type = {pulse_type}")
    print(f"rolloff = {rolloff}")
    print(f"span = {span}")
    print(f"periodic_replicas = {PERIODIC_REPLICAS}")
    print(f"mse_samples_original = {mse_samples_original:.8e}")
    print(f"mse_continuous = {mse_continuous:.8e}")
    print(f"normalized_sample_consistency = {mse_samples_normalized_consistency:.8e}")

    if do_plot:
        plot_ppm_single_trial(
            t=t,
            x_ref_sensor=x_ref_sensor,
            x_ref_sensor_norm=x_ref_sensor_norm,
            symbol_times=symbol_times,
            original_samples=original_samples,
            sampled_message_norm=sampled_message_norm,
            pulse_positions=pulse_positions,
            tx_waveform=tx[:, 0, 0],
            rx_waveform=rx[:, 0, 0],
            recovered_samples=recovered_samples,
            recovered_continuous=recovered_continuous,
        )

    return {
        "config": cfg_B,
        "t": t,
        "x_ref_sensor": x_ref_sensor,
        "x_ref_sensor_norm": x_ref_sensor_norm,
        "symbol_times": symbol_times,
        "original_samples": original_samples,
        "sampled_message_norm": sampled_message_norm,
        "original_samples_norm": original_samples_norm,
        "pulse_positions": pulse_positions,
        "tx_waveform": tx[:, 0, 0],
        "rx_waveform": rx[:, 0, 0],
        "recovered_samples": recovered_samples,
        "recovered_continuous": recovered_continuous,
        "mse_samples_original": mse_samples_original,
        "mse_continuous": mse_continuous,
        "normalized_sample_consistency": mse_samples_normalized_consistency,
        "B_sensor": B_sensor,
        "SNR_sensor_linear": snr_sensor_linear,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--B", type=float, required=True)
    parser.add_argument("--sensor", type=int, default=0)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    debug_ppm_single_trial(
        config_path=args.config,
        B_value=args.B,
        sensor_idx=args.sensor,
        seed=args.seed,
        do_plot=True
    )


if __name__ == "__main__":
    main()

# from tests.debug_ppm_single_trial import debug_ppm_single_trial
#
# out = debug_ppm_single_trial(
#     config_path="experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml",
#     B_value=2500,
#     sensor_idx=0,
#     seed=47,
#     do_plot=True
# )
