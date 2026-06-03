"""
sfc/core/system_parameters.py

System-parameter builder and compatibility wrappers.

Purpose
-------
This module centralizes the construction of derived system parameters used by
simulation pipelines.

Main design rule
----------------
The physical channel model should be parameterized primarily by:

    P  = average available transmit power per sensor
    N0 = noise spectral-density / noise parameter

Then SNR is derived, not manually imposed, except in the special case where the
configuration intentionally asks to derive P from a fixed per-sensor SNR.

Default physical model
----------------------
By default, simulations should provide:

    system:
      P:  ...
      N0: ...

and the builder computes:

    SNR_total = P / (B * N0)

For methods that use per-sensor bandwidth slices:

    B_s = alpha_s * B
    SNR_s = P / (B_s * N0)

Fixed per-sensor SNR mode
-------------------------
If system.P is null and system.SNR_dB is provided, this file interprets SNR_dB
as the desired per-sensor SNR. Then P is derived from:

    SNR_s = P / (B_s * N0)

For a scalar P shared by all sensors, this is only consistent for uniform
bandwidth allocation. For nonuniform allocation, either use fixed P,N0 or extend
the model to sensor-dependent powers P_s.

Compatibility
-------------
This module also provides wrappers for:

- compute_M_rbcp(...)
- compute_M_rbcp_per_sensor(...)
- compute_benchmark_M_per_sensor(...)

The preferred usage is with P and N0.

Legacy usage with SNR is still accepted, but SNR is interpreted as already being
the appropriate SNR for the corresponding sensor channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from sfc.core.theory import (
    compute_N as theory_compute_N,
    compute_snr_linear as theory_compute_snr_linear,
    compute_N0 as theory_compute_N0,
    compute_sensor_bandwidths as theory_compute_sensor_bandwidths,
    compute_M_rbcp_per_sensor as theory_compute_M_rbcp_per_sensor,
    compute_M_rbcp_single_sensor as theory_compute_M_rbcp_single_sensor,
    compute_benchmark_M_per_sensor as theory_compute_benchmark_M_per_sensor,
    compute_benchmark_M_single_sensor as theory_compute_benchmark_M_single_sensor,
    compute_M_time as theory_compute_M_time,
    compute_slot_duration as theory_compute_slot_duration,
    compute_slots_per_period as theory_compute_slots_per_period,
    compute_event_slots_total as theory_compute_event_slots_total,
    compute_rx_slots_total as theory_compute_rx_slots_total,
    compute_total_energy as theory_compute_total_energy,
    compute_symbol_energy as theory_compute_symbol_energy,
    compute_signal_level as theory_compute_signal_level,
    compute_default_detection_threshold as theory_compute_default_detection_threshold,
)


# =============================================================================
# DATA CONTAINER
# =============================================================================

@dataclass
class DerivedSystemParameters:
    """
    Container for derived system parameters.
    """

    S: int
    B: float
    P: float
    N0: float

    E_tot: float
    E_s: float
    signal_level: float
    default_threshold: float

    SNR: float
    SNR_dB: float
    SNR_per_sensor: np.ndarray
    SNR_per_sensor_dB: np.ndarray

    W: float
    tau: float
    N: int

    R: int
    L: int

    bandwidth_allocation: np.ndarray
    B_per_sensor: np.ndarray

    M_time: int
    slot_duration: float
    slots_per_period: int
    event_slots_total: int
    rx_slots_total: int

    M_rbcp_per_sensor: np.ndarray
    M_rbcp: int

    quantization_force_power_of_two: bool
    quantization_rounding_mode: str


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def _as_optional_float(value):
    """
    Convert value to float unless value is None.
    """
    if value is None:
        return None
    return float(value)


def _safe_log10(x):
    """
    Safe log10 for positive scalar/array.
    """
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, np.finfo(float).tiny))


# =============================================================================
# POWER / NOISE RESOLUTION
# =============================================================================

def resolve_power_noise_model(
    cfg: Dict[str, Any],
    *,
    B: float,
    S: int,
    bandwidth_allocation: np.ndarray,
) -> Tuple[float, float]:
    """
    Resolve P and N0 from the configuration.

    Preferred case
    --------------
    If both P and N0 are provided:

        P  = cfg["system"]["P"]
        N0 = cfg["system"]["N0"]

    then these are used directly.

    Fixed per-sensor SNR case
    -------------------------
    If P is None and N0 is provided, then system.SNR_dB must be provided and is
    interpreted as the desired per-sensor SNR.

    Then:

        SNR_s = P / (B_s * N0)

    For scalar P, this is only consistent when bandwidth allocation is uniform.
    """

    system_cfg = cfg.get("system", {})

    P = _as_optional_float(system_cfg.get("P", None))
    N0 = _as_optional_float(system_cfg.get("N0", None))
    SNR_dB = _as_optional_float(system_cfg.get("SNR_dB", None))

    if B <= 0:
        raise ValueError("system.B must be positive.")

    if S < 1:
        raise ValueError("system.S must be >= 1.")

    if P is not None and P <= 0:
        raise ValueError("system.P must be positive when provided.")

    if N0 is not None and N0 <= 0:
        raise ValueError("system.N0 must be positive when provided.")

    # -------------------------------------------------------------------------
    # Preferred mode: P and N0 are explicitly given.
    # -------------------------------------------------------------------------
    if P is not None and N0 is not None:
        return float(P), float(N0)

    # -------------------------------------------------------------------------
    # Fixed per-sensor SNR mode: derive P from SNR_dB and N0.
    # -------------------------------------------------------------------------
    if P is None and N0 is not None and SNR_dB is not None:
        allocation = np.asarray(bandwidth_allocation, dtype=float).reshape(-1)
        uniform_allocation = np.ones(S, dtype=float) / S

        if not np.allclose(allocation, uniform_allocation):
            raise ValueError(
                "Cannot derive a scalar P from fixed per-sensor SNR_dB with "
                "nonuniform bandwidth allocation. Use fixed P,N0 or introduce "
                "sensor-dependent powers P_s."
            )

        SNR_sensor = theory_compute_snr_linear(SNR_dB)
        B_sensor = B * allocation[0]
        P_derived = SNR_sensor * B_sensor * N0

        return float(P_derived), float(N0)

    # -------------------------------------------------------------------------
    # Legacy inverse mode: derive N0 from P and SNR_dB using total B.
    # Not the preferred default, but kept for backward compatibility.
    # -------------------------------------------------------------------------
    if P is not None and N0 is None and SNR_dB is not None:
        SNR_total = theory_compute_snr_linear(SNR_dB)
        N0_derived = theory_compute_N0(P=P, B=B, SNR=SNR_total)

        return float(P), float(N0_derived)

    raise ValueError(
        "Invalid system power/noise configuration. Expected either:\n"
        "  - system.P and system.N0, or\n"
        "  - system.P = null, system.N0, and system.SNR_dB for fixed "
        "per-sensor SNR mode, or\n"
        "  - system.P, system.N0 = null, and system.SNR_dB for legacy "
        "N0 derivation."
    )


# =============================================================================
# COMPATIBILITY WRAPPERS AROUND THEORY.PY
# =============================================================================

def compute_M_rbcp_per_sensor(
    S: int,
    W: float,
    tau: float,
    B: float,
    P: Optional[float] = None,
    N0: Optional[float] = None,
    SNR: Optional[float] = None,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> np.ndarray:
    """
    Compute per-sensor feasible M_RbCP values.

    Preferred usage:
        use P and N0.

    Legacy usage:
        use SNR, interpreted as already being the correct sensor-channel SNR.
    """

    if P is not None and N0 is not None:
        return theory_compute_M_rbcp_per_sensor(
            S=S,
            W=W,
            tau=tau,
            B=B,
            P=P,
            N0=N0,
            bandwidth_allocation=bandwidth_allocation,
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode,
        )

    if SNR is None:
        raise ValueError("compute_M_rbcp_per_sensor requires either (P,N0) or SNR.")

    _, B_per_sensor = theory_compute_sensor_bandwidths(
        B=B,
        S=S,
        bandwidth_allocation=bandwidth_allocation,
    )

    return np.array([
        theory_compute_M_rbcp_single_sensor(
            W=W,
            tau=tau,
            B_sensor=B_s,
            P=1.0,
            N0=1.0 / (B_s * SNR),
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode,
        )
        for B_s in B_per_sensor
    ], dtype=int)


def compute_M_rbcp(
    S: int,
    W: float,
    tau: float,
    B: float,
    P: Optional[float] = None,
    N0: Optional[float] = None,
    SNR: Optional[float] = None,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> int:
    """
    Compute common feasible M_RbCP across sensors:

        M_RbCP = min_s M_RbCP,s
    """

    M_vec = compute_M_rbcp_per_sensor(
        S=S,
        W=W,
        tau=tau,
        B=B,
        P=P,
        N0=N0,
        SNR=SNR,
        bandwidth_allocation=bandwidth_allocation,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode,
    )

    return int(np.min(M_vec))


def compute_benchmark_M_per_sensor(
    S: int,
    tau: float,
    B: float,
    sampling_rate: float,
    P: Optional[float] = None,
    N0: Optional[float] = None,
    SNR: Optional[float] = None,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> np.ndarray:
    """
    Compute per-sensor feasible Benchmark quantization bins M.

    Preferred usage:
        use P and N0.

    Legacy usage:
        use SNR, interpreted as already being the correct sensor-channel SNR.
    """

    if P is not None and N0 is not None:
        return theory_compute_benchmark_M_per_sensor(
            S=S,
            tau=tau,
            B=B,
            P=P,
            N0=N0,
            sampling_rate=sampling_rate,
            bandwidth_allocation=bandwidth_allocation,
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode,
        )

    if SNR is None:
        raise ValueError(
            "compute_benchmark_M_per_sensor requires either (P,N0) or SNR."
        )

    _, B_per_sensor = theory_compute_sensor_bandwidths(
        B=B,
        S=S,
        bandwidth_allocation=bandwidth_allocation,
    )

    return np.array([
        theory_compute_benchmark_M_single_sensor(
            tau=tau,
            B_sensor=B_s,
            P=1.0,
            N0=1.0 / (B_s * SNR),
            sampling_rate=sampling_rate,
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode,
        )
        for B_s in B_per_sensor
    ], dtype=int)


def compute_benchmark_M_single_sensor(
    tau: float,
    B_sensor: float,
    sampling_rate: float,
    P: Optional[float] = None,
    N0: Optional[float] = None,
    SNR: Optional[float] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> int:
    """
    Compute feasible Benchmark M for a single sensor.

    Preferred usage:
        use P and N0.

    Legacy usage:
        use SNR, interpreted as the SNR for this sensor channel.
    """

    if P is not None and N0 is not None:
        return theory_compute_benchmark_M_single_sensor(
            tau=tau,
            B_sensor=B_sensor,
            P=P,
            N0=N0,
            sampling_rate=sampling_rate,
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode,
        )

    if SNR is None:
        raise ValueError(
            "compute_benchmark_M_single_sensor requires either (P,N0) or SNR."
        )

    return theory_compute_benchmark_M_single_sensor(
        tau=tau,
        B_sensor=B_sensor,
        P=1.0,
        N0=1.0 / (B_sensor * SNR),
        sampling_rate=sampling_rate,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode,
    )


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_derived_system_parameters(cfg: Dict[str, Any]) -> DerivedSystemParameters:
    """
    Build derived system parameters from a parsed YAML configuration.
    """

    system_cfg = cfg.get("system", {})
    signal_cfg = cfg.get("signal", {})
    quant_cfg = cfg.get("quantization", {})

    # -------------------------------------------------------------------------
    # Basic system parameters
    # -------------------------------------------------------------------------
    S = int(system_cfg.get("S", system_cfg.get("sensor_nodes", 1)))
    B = float(system_cfg["B"])

    W = float(signal_cfg.get("W", system_cfg.get("W", 1.0)))
    tau = float(signal_cfg.get("tau", signal_cfg.get("T", system_cfg.get("tau", 1.0))))

    R = int(system_cfg.get("R", 1))
    L = int(system_cfg.get("L", 1))

    n_periods = int(
        signal_cfg.get(
            "n_periods",
            system_cfg.get(
                "n_periods",
                cfg.get("simulation", {}).get("n_periods", 1)
            )
        )
    )

    # -------------------------------------------------------------------------
    # Bandwidth allocation
    # -------------------------------------------------------------------------
    raw_allocation = system_cfg.get("bandwidth_allocation", None)

    bandwidth_allocation, B_per_sensor = theory_compute_sensor_bandwidths(
        B=B,
        S=S,
        bandwidth_allocation=raw_allocation,
    )

    # -------------------------------------------------------------------------
    # Resolve physical power/noise model
    # -------------------------------------------------------------------------
    P, N0 = resolve_power_noise_model(
        cfg=cfg,
        B=B,
        S=S,
        bandwidth_allocation=bandwidth_allocation,
    )

    # -------------------------------------------------------------------------
    # SNR quantities
    # -------------------------------------------------------------------------
    SNR_total = P / (B * N0)
    SNR_total_dB = float(_safe_log10(SNR_total))

    SNR_per_sensor = P / (B_per_sensor * N0)
    SNR_per_sensor_dB = _safe_log10(SNR_per_sensor)

    # -------------------------------------------------------------------------
    # Energy / amplitude quantities used by SFC physical channel and detector
    # -------------------------------------------------------------------------
    threshold_factor = float(
        cfg.get("channel", {}).get("threshold_factor", 0.5)
    )

    E_tot = theory_compute_total_energy(
        P=P,
        tau=tau,
    )

    E_s = theory_compute_symbol_energy(
        P=P,
        tau=tau,
        L=L,
    )

    signal_level = theory_compute_signal_level(
        P=P,
        tau=tau,
        L=L,
    )

    default_threshold = theory_compute_default_detection_threshold(
        P=P,
        tau=tau,
        L=L,
        threshold_factor=threshold_factor,
    )

    # -------------------------------------------------------------------------
    # Harmonics
    # -------------------------------------------------------------------------
    if "N_override" in signal_cfg and signal_cfg["N_override"] is not None:
        N = int(signal_cfg["N_override"])
    else:
        N = theory_compute_N(W=W, tau=tau)

    # -------------------------------------------------------------------------
    # Quantization policy
    # -------------------------------------------------------------------------
    quantization_force_power_of_two = bool(
        quant_cfg.get("force_power_of_two", False)
    )
    quantization_rounding_mode = quant_cfg.get("rounding_mode", "floor")

    # -------------------------------------------------------------------------
    # SFC time quantities use total B, not B_s.
    # -------------------------------------------------------------------------
    M_time = theory_compute_M_time(tau=tau, B=B, R=R)
    slot_duration = theory_compute_slot_duration(B=B, R=R)
    slots_per_period = theory_compute_slots_per_period(tau=tau, B=B, R=R)
    event_slots_total = theory_compute_event_slots_total(
        tau=tau,
        B=B,
        R=R,
        n_periods=n_periods,
    )
    rx_slots_total = theory_compute_rx_slots_total(
        tau=tau,
        B=B,
        R=R,
        n_periods=n_periods,
        L=L,
    )

    # -------------------------------------------------------------------------
    # RbCP M values use per-sensor B_s and SNR_s computed from P,N0.
    # -------------------------------------------------------------------------
    M_rbcp_per_sensor = compute_M_rbcp_per_sensor(
        S=S,
        W=W,
        tau=tau,
        B=B,
        P=P,
        N0=N0,
        bandwidth_allocation=bandwidth_allocation,
        force_power_of_two=quantization_force_power_of_two,
        rounding_mode=quantization_rounding_mode,
    )

    M_rbcp = int(np.min(M_rbcp_per_sensor))

    return DerivedSystemParameters(
        S=S,
        B=B,
        P=P,
        N0=N0,
        E_tot=E_tot,
        E_s=E_s,
        signal_level=signal_level,
        default_threshold=default_threshold,
        SNR=SNR_total,
        SNR_dB=SNR_total_dB,
        SNR_per_sensor=SNR_per_sensor,
        SNR_per_sensor_dB=SNR_per_sensor_dB,
        W=W,
        tau=tau,
        N=N,
        R=R,
        L=L,
        bandwidth_allocation=bandwidth_allocation,
        B_per_sensor=B_per_sensor,
        M_time=M_time,
        slot_duration=slot_duration,
        slots_per_period=slots_per_period,
        event_slots_total=event_slots_total,
        rx_slots_total=rx_slots_total,
        M_rbcp_per_sensor=M_rbcp_per_sensor,
        M_rbcp=M_rbcp,
        quantization_force_power_of_two=quantization_force_power_of_two,
        quantization_rounding_mode=quantization_rounding_mode,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DerivedSystemParameters",
    "build_derived_system_parameters",
    "resolve_power_noise_model",

    "compute_M_rbcp_per_sensor",
    "compute_M_rbcp",
    "compute_benchmark_M_per_sensor",
    "compute_benchmark_M_single_sensor",
]
