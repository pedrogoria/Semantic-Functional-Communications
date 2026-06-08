"""
tests/debug_sfc_single_trial.py

Debug / visualization script for a single SFC trial using the same modeling
choices adopted by the fair comparison pipeline.

This script:
- loads the comparison YAML
- selects one B value
- resolves the physical budget consistently with the project convention
- generates one common band-limited source realization with the same SFC-style logic
- computes the native SFC chain
- optionally computes the SFC + SED chain
- visualizes one selected sensor

Main debug views
----------------
1. Message Signal
2. SFC Reconstruction
3. SFC + SED Reconstruction, optional
4. Compact diagnostics for event-domain processing

Physical convention
-------------------
Preferred/default mode:

    P and N0 are provided in the YAML.

Then:

    SNR_total = P / (B * N0)

SFC uses the total bandwidth B, not per-sensor bandwidth B_s.

Optional fixed per-sensor SNR mode:

    If power_model.mode is "fixed_sensor_SNR_and_N0",
    the YAML SNR_dB is interpreted as SNR_s and P is derived from:

        P = SNR_s * B_s * N0

    With a scalar P shared by all sensors, this mode requires uniform
    bandwidth allocation.

Usage, Python console / PyCharm
-------------------------------
from tests.debug_sfc_single_trial import debug_sfc_single_trial

out = debug_sfc_single_trial(
    config_path="experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml",
    B_value=2500,
    sensor_idx=0,
    seed=47,
    use_sed=True,
    do_plot=True
)

Terminal
--------
python tests/debug_sfc_single_trial.py --config experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml --B 2500 --sensor 0 --use-sed
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
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import PhaseCoefficientCore
from sfc.core.reconstruction import recover_signal
from sfc.core.channel.SFCChannel import SFCChannel
from sfc.core.system_parameters import build_derived_system_parameters
from sfc.core.semantic_error_detection import detect_semantic_errors


# =============================================================================
# CONFIG / POWER MODEL
# =============================================================================

def load_config(path: str) -> dict:
    """
    Load YAML configuration file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _snr_linear_from_db(snr_db: float) -> float:
    """
    Convert SNR from dB to linear scale.
    """
    return 10.0 ** (float(snr_db) / 10.0)


def _snr_db_from_linear(snr: float) -> float:
    """
    Convert SNR from linear scale to dB.
    """
    return 10.0 * np.log10(max(float(snr), np.finfo(float).tiny))


def _resolve_bandwidth_allocation_from_cfg(cfg: dict) -> np.ndarray:
    """
    Resolve and validate bandwidth allocation from cfg["system"].

    If no bandwidth allocation is provided, an equal split among S sensors is used.
    """
    system_cfg = cfg.get("system", {})
    S = int(system_cfg.get("S", system_cfg.get("sensor_nodes", 1)))

    allocation = system_cfg.get("bandwidth_allocation", None)

    if allocation is None:
        return np.ones(S, dtype=float) / S

    allocation = np.asarray(allocation, dtype=float).reshape(-1)

    if len(allocation) != S:
        raise ValueError(
            f"bandwidth_allocation length mismatch: len={len(allocation)} but S={S}"
        )

    if np.any(allocation < 0):
        raise ValueError("bandwidth_allocation must be nonnegative.")

    if not np.isclose(np.sum(allocation), 1.0):
        raise ValueError(
            f"bandwidth_allocation must sum to 1. Current sum={np.sum(allocation)}"
        )

    return allocation


def apply_power_model_inplace(cfg: dict):
    """
    Resolve the selected physical-budget regime in-place.

    Supported modes
    ---------------
    1. fixed_P_and_N0
       Preferred/default mode.
       - keep P fixed
       - keep N0 fixed
       - SNR is derived by build_derived_system_parameters(...)
       - this function only updates system.SNR_dB as total-band metadata

    2. fixed_sensor_SNR_and_N0
       Optional mode.
       - keep per-sensor SNR_dB fixed
       - keep N0 fixed
       - derive scalar P from:
             P = SNR_s * B_s * N0

       This requires uniform bandwidth allocation when using one scalar P.

    Deprecated / unsupported
    ------------------------
    fixed_P_and_SNR is intentionally not supported here because the current
    project convention is:
    - P and N0 are primary;
    - SNR is derived;
    - SNR_dB is used only when deriving P.
    """

    mode = cfg.get("power_model", {}).get("mode", "fixed_P_and_N0")
    system_cfg = cfg.setdefault("system", {})

    B = float(system_cfg["B"])
    P = system_cfg.get("P", None)
    N0 = system_cfg.get("N0", None)

    if mode == "fixed_P_and_N0":
        if P is None:
            raise ValueError(
                "power_model.mode='fixed_P_and_N0' requires system.P to be provided."
            )
        if N0 is None:
            raise ValueError(
                "power_model.mode='fixed_P_and_N0' requires system.N0 to be provided."
            )

        P = float(P)
        N0 = float(N0)

        snr_total = P / (B * N0)
        system_cfg["SNR_dB"] = _snr_db_from_linear(snr_total)
        return

    if mode in {
        "fixed_sensor_SNR_and_N0",
        "fixed_SNR_per_sensor_and_N0",
        "fixed_per_sensor_SNR_and_N0",
    }:
        if N0 is None:
            raise ValueError(
                f"power_model.mode='{mode}' requires system.N0 to be provided."
            )

        if system_cfg.get("SNR_dB", None) is None:
            raise ValueError(
                f"power_model.mode='{mode}' requires system.SNR_dB to be provided."
            )

        allocation = _resolve_bandwidth_allocation_from_cfg(cfg)
        S = len(allocation)
        uniform_allocation = np.ones(S, dtype=float) / S

        if not np.allclose(allocation, uniform_allocation):
            raise ValueError(
                "Fixed per-sensor SNR with scalar P requires uniform bandwidth allocation. "
                "For nonuniform allocation, use fixed P,N0 or introduce sensor-dependent P_s."
            )

        snr_sensor = _snr_linear_from_db(system_cfg["SNR_dB"])
        B_sensor = B * allocation[0]
        system_cfg["P"] = snr_sensor * B_sensor * float(N0)
        return

    if mode == "fixed_P_and_SNR":
        raise ValueError(
            "power_model.mode='fixed_P_and_SNR' is deprecated in this project convention. "
            "Use either fixed_P_and_N0 or fixed_sensor_SNR_and_N0."
        )

    raise ValueError(
        f"Unsupported power_model.mode: {mode}. "
        "Supported modes are: 'fixed_P_and_N0' and 'fixed_sensor_SNR_and_N0'."
    )


# =============================================================================
# COMMON SOURCE GENERATION
# =============================================================================

def generate_common_bandlimited_signals(
    cfg: dict,
    rng: np.random.Generator,
    params,
    N: int
):
    """
    Generate the common source signals using the same logic as the comparison
    pipeline.

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

    x_filtered = np.zeros_like(x_raw)

    for p in range(n_periods):
        for s in range(S):
            x_filtered[:, p, s] = filter_periodic(
                x_raw[:, p, s],
                params.W,
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
# SHARED FOURIER / PHASE HELPERS
# =============================================================================

def compute_ta_tb_from_reference(cfg, params, x_ref, Tt, N):
    """
    Compute ta/tb from the common source reference.
    """

    fourier_core = FourierCoefficientCore(
        T=params.tau,
        harmonics=N,
        sensor_nodes=params.S
    )

    an, bn, _ = fourier_core.calc_an_bn_dft(
        x_ref,
        Tt,
        normalize=cfg["signal"]["normalize_dft"],
        norm=cfg["signal"]["normalization_target"]
    )

    phase_core = PhaseCoefficientCore(
        T=params.tau,
        harmonics=N,
        n_sub_symbol=params.L,
        resource=params.R,
        sensor_nodes=params.S,
        bandwidth=params.B,
        detect_errors=False,
        periods=1,
        threshold_harmonics=cfg["signal"].get("threshold_harmonics", 0.001)
    )

    ta, tb = phase_core.calc_ta_tb(an, bn)

    return np.real(ta), np.real(tb), phase_core


def build_sensor_x_event(S, N):
    """
    Build the sensor-event association matrix.

    Event ordering:
    for each sensor s:
        [ta events for N harmonics][tb events for N harmonics]
    """

    num_event_ids = 2 * N * S
    sensor_x_event = np.zeros((S, num_event_ids))

    for s in range(S):
        start = 2 * s * N
        stop = 2 * (s + 1) * N
        sensor_x_event[s, start:stop] = 1.0

    return sensor_x_event


def build_sfc_channel_for_B(cfg_B, N, S):
    """
    Build one native SFCChannel instance for the current B-point.
    """

    cfg_sfc = copy.deepcopy(cfg_B)

    if "channel" not in cfg_sfc:
        cfg_sfc["channel"] = {}

    cfg_sfc["channel"]["sensor_x_event"] = build_sensor_x_event(S, N)
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
        cfg_sfc["reproducibility"]["seed"] = cfg_B.get(
            "reproducibility", {}
        ).get(
            "seed",
            cfg_B.get("monte_carlo", {}).get("seed", 12345)
        )

    return SFCChannel(cfg_sfc)


def extract_sed_outputs(sed_result):
    """
    Extract (corrected_events_est, period_valid_mask) from the SED result.

    Supports dict-like or dataclass-like objects.
    """

    if isinstance(sed_result, dict):
        corrected_events_est = sed_result["corrected_events_est"]
        period_valid_mask = np.asarray(sed_result["period_valid_mask"], dtype=bool)
        return corrected_events_est, period_valid_mask

    corrected_events_est = getattr(sed_result, "corrected_events_est")
    period_valid_mask = np.asarray(getattr(sed_result, "period_valid_mask"), dtype=bool)

    return corrected_events_est, period_valid_mask


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sfc_single_trial(
    t: np.ndarray,
    x_ref_sensor: np.ndarray,
    x_rec_sfc_sensor: np.ndarray,
    x_rec_sfc_sed_sensor: np.ndarray | None,
    events: np.ndarray,
    events_est: np.ndarray,
    corrected_events_est: np.ndarray | None,
    sensor_idx: int,
    valid_sfc_sed_trial: bool,
    use_sed: bool,
):
    """
    Plot the main debug views for one SFC trial.
    """

    show_sed_panel = use_sed

    if show_sed_panel:
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # -----------------------------------------------------------------
    # 1. Message / SFC reconstruction
    # -----------------------------------------------------------------
    axes[0].plot(t, x_ref_sensor, linewidth=1.2, label="Original")
    axes[0].plot(t, x_rec_sfc_sensor, "--", linewidth=1.2, label="SFC reconstruction")
    axes[0].set_title(f"Message Signal vs SFC Reconstruction (sensor {sensor_idx})")
    axes[0].grid(True)
    axes[0].legend()

    # -----------------------------------------------------------------
    # 2. SFC + SED reconstruction, optional
    # -----------------------------------------------------------------
    offset = 1

    if show_sed_panel:
        axes[1].plot(t, x_ref_sensor, linewidth=1.2, label="Original")

        if x_rec_sfc_sed_sensor is not None:
            axes[1].plot(
                t,
                x_rec_sfc_sed_sensor,
                "--",
                linewidth=1.2,
                label="SFC + SED reconstruction"
            )

        title = f"SFC + SED Reconstruction (sensor {sensor_idx})"
        if not valid_sfc_sed_trial:
            title += " [INVALID / DISCARDED]"

        axes[1].set_title(title)
        axes[1].grid(True)
        axes[1].legend()
        offset = 2

    # -----------------------------------------------------------------
    # 3. Events sent
    # -----------------------------------------------------------------
    axes[offset].imshow(
        events.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    axes[offset].set_title("Event matrix (transmitted)")
    axes[offset].set_xlabel("Event slot")
    axes[offset].set_ylabel("Event index")

    # -----------------------------------------------------------------
    # 4. Events received / corrected
    # -----------------------------------------------------------------
    ax_last = axes[offset + 1]

    if show_sed_panel and corrected_events_est is not None:
        img = corrected_events_est
        title = "Event matrix after SED"
    else:
        img = events_est
        title = "Event matrix (received)"

    ax_last.imshow(
        img.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    ax_last.set_title(title)
    ax_last.set_xlabel("Event slot")
    ax_last.set_ylabel("Event index")

    plt.tight_layout()
    plt.show(block=True)


# =============================================================================
# MAIN DEBUG FUNCTION
# =============================================================================

def debug_sfc_single_trial(
    config_path: str,
    B_value: float,
    sensor_idx: int = 0,
    seed: int = 47,
    use_sed: bool = True,
    do_plot: bool = True
):
    """
    Run and visualize one SFC trial consistent with the fair-comparison setup.
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
    Tt = common["Tt"]
    x_ref = common["x_ref"]  # shape: (time, 1, S)

    # ------------------------------------------------------------
    # 2) Shared Fourier / ta-tb
    # ------------------------------------------------------------
    ta, tb, phase_core = compute_ta_tb_from_reference(
        cfg=cfg_B,
        params=params,
        x_ref=x_ref,
        Tt=Tt,
        N=N
    )

    # ------------------------------------------------------------
    # 3) Native SFC channel
    # ------------------------------------------------------------
    sfc_channel = build_sfc_channel_for_B(cfg_B, N, params.S)

    events = phase_core.ta_tb_to_events(ta, tb)
    out = sfc_channel(events)
    events_est = out["events_est"] if isinstance(out, dict) else out

    # ------------------------------------------------------------
    # 4) Native SFC without SED
    # ------------------------------------------------------------
    ta_rec, tb_rec = phase_core.event_to_ta_tb(events_est)
    ta_rec = np.real(ta_rec)
    tb_rec = np.real(tb_rec)

    w0 = 2.0 * np.pi / params.tau

    x_rec_sfc = np.zeros_like(x_ref)
    for s in range(params.S):
        x_rec_sfc[:, 0, s] = recover_signal(
            ta_rec[0, :, s],
            tb_rec[0, :, s],
            t,
            w0
        )

    mse_sfc = float(
        np.mean((x_ref[:, 0, sensor_idx] - x_rec_sfc[:, 0, sensor_idx]) ** 2)
    )

    # ------------------------------------------------------------
    # 5) Native SFC + SED, optional
    # ------------------------------------------------------------
    x_rec_sfc_sed = None
    mse_sfc_sed = np.nan
    valid_fraction_sfc_sed = np.nan
    is_valid_sfc_sed_trial = False
    corrected_events_est = None
    period_valid_mask = None

    if use_sed:
        period_slots = events_est.shape[0]
        sed_cfg = cfg_B.get("sed", {})
        sensor_x_event = build_sensor_x_event(params.S, N)

        sed_result = detect_semantic_errors(
            events_est=events_est,
            period_slots=period_slots,
            N=N,
            sensor_x_event=sensor_x_event,
            discard_invalid_periods=sed_cfg.get("discard_invalid_periods", True)
        )

        corrected_events_est, period_valid_mask = extract_sed_outputs(sed_result)

        valid_fraction_sfc_sed = float(np.mean(period_valid_mask))
        is_valid_sfc_sed_trial = bool(period_valid_mask[0])

        if is_valid_sfc_sed_trial:
            ta_rec_sed, tb_rec_sed = phase_core.event_to_ta_tb(corrected_events_est)
            ta_rec_sed = np.real(ta_rec_sed)
            tb_rec_sed = np.real(tb_rec_sed)

            x_rec_sfc_sed = np.zeros_like(x_ref)
            for s in range(params.S):
                x_rec_sfc_sed[:, 0, s] = recover_signal(
                    ta_rec_sed[0, :, s],
                    tb_rec_sed[0, :, s],
                    t,
                    w0
                )

            mse_sfc_sed = float(
                np.mean((x_ref[:, 0, sensor_idx] - x_rec_sfc_sed[:, 0, sensor_idx]) ** 2)
            )
        else:
            x_rec_sfc_sed = None

    print("\n[DEBUG SFC SINGLE TRIAL]")
    print(f"config_path = {config_path}")
    print(f"B_value = {B_value}")
    print(f"sensor_idx = {sensor_idx}")
    print(f"seed = {seed}")
    print(f"use_sed = {use_sed}")
    print(f"power_model.mode = {cfg_B.get('power_model', {}).get('mode', 'fixed_P_and_N0')}")
    print(f"S = {params.S}")
    print(f"P = {params.P}")
    print(f"N0 = {params.N0}")
    print(f"B_total = {params.B}")
    print(f"SNR_total = {params.SNR}")
    print(f"SNR_total_dB = {params.SNR_dB}")
    print(f"SNR_per_sensor = {params.SNR_per_sensor}")
    print(f"SNR_per_sensor_dB = {params.SNR_per_sensor_dB}")
    print(f"tau = {params.tau}")
    print(f"W = {params.W}")
    print(f"N = {N}")
    print(f"R = {params.R}")
    print(f"L = {params.L}")
    print(f"mse_sfc = {mse_sfc:.8e}")

    if use_sed:
        if np.isnan(mse_sfc_sed):
            print("mse_sfc_sed = nan (invalid / discarded trial)")
        else:
            print(f"mse_sfc_sed = {mse_sfc_sed:.8e}")
        print(f"valid_fraction_sfc_sed = {valid_fraction_sfc_sed}")
        print(f"is_valid_sfc_sed_trial = {is_valid_sfc_sed_trial}")
        print(f"period_valid_mask = {period_valid_mask}")

    if do_plot:
        plot_sfc_single_trial(
            t=t,
            x_ref_sensor=x_ref[:, 0, sensor_idx],
            x_rec_sfc_sensor=x_rec_sfc[:, 0, sensor_idx],
            x_rec_sfc_sed_sensor=None if x_rec_sfc_sed is None else x_rec_sfc_sed[:, 0, sensor_idx],
            events=events,
            events_est=events_est,
            corrected_events_est=corrected_events_est,
            sensor_idx=sensor_idx,
            valid_sfc_sed_trial=is_valid_sfc_sed_trial,
            use_sed=use_sed,
        )

    return {
        "config": cfg_B,
        "params": params,
        "t": t,
        "x_ref": x_ref,
        "ta": ta,
        "tb": tb,
        "events": events,
        "events_est": events_est,
        "corrected_events_est": corrected_events_est,
        "period_valid_mask": period_valid_mask,
        "x_rec_sfc": x_rec_sfc,
        "x_rec_sfc_sed": x_rec_sfc_sed,
        "mse_sfc": mse_sfc,
        "mse_sfc_sed": mse_sfc_sed,
        "valid_fraction_sfc_sed": valid_fraction_sfc_sed,
        "is_valid_sfc_sed_trial": is_valid_sfc_sed_trial,
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
    parser.add_argument("--use-sed", action="store_true")
    args = parser.parse_args()

    debug_sfc_single_trial(
        config_path=args.config,
        B_value=args.B,
        sensor_idx=args.sensor,
        seed=args.seed,
        use_sed=args.use_sed,
        do_plot=True
    )


if __name__ == "__main__":
    main()


# from tests.debug_sfc_single_trial import debug_sfc_single_trial
#
# out = debug_sfc_single_trial(
#     config_path="experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml",
#     B_value=2500,
#     sensor_idx=0,
#     seed=47,
#     use_sed=False,
#     do_plot=True
# )
