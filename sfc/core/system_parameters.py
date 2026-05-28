"""
sfc/core/system_parameters.py

Builder and compatibility layer for system-level derived parameters.

Design rule
-----------
- Closed-form theoretical formulas live in:
      sfc/core/theory.py
- This file is responsible for:
    1. Packaging primary + derived quantities into a frozen dataclass
    2. Building the central parameter object from cfg
    3. Preserving backward compatibility through lightweight wrappers

This avoids duplication of physics / communication formulas across:
- physical_channel.py
- detection.py
- pipelines
- debug scripts
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from sfc.core.theory import (
    compute_N as theory_compute_N,
    compute_capacity as theory_compute_capacity,
    compute_snr_linear as theory_compute_snr_linear,
    compute_M_rbcp as theory_compute_M_rbcp,
    compute_M_rbcp_per_sensor as theory_compute_M_rbcp_per_sensor,
    compute_M_time as theory_compute_M_time,
    compute_N0 as theory_compute_N0,
    compute_total_energy as theory_compute_total_energy,
    compute_symbol_energy as theory_compute_symbol_energy,
    compute_signal_level as theory_compute_signal_level,
    compute_default_detection_threshold as theory_compute_default_detection_threshold,
    compute_bandwidth_allocation as theory_compute_bandwidth_allocation,
    compute_sensor_bandwidths as theory_compute_sensor_bandwidths,
)


# =============================================================================
# DERIVED SYSTEM PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class DerivedSystemParameters:
    """
    Immutable container for system-level primary + derived parameters.

    Primary parameters
    ------------------
    S : int
        Number of sensors/signals.

    P : float
        Average transmit power per sensor.

    B : float
        Total system bandwidth.

    R : int
        Number of resources / subcarriers.

    L : int
        Number of temporal resources / symbols per cycle.

    SNR_dB : float
        Signal-to-noise ratio in dB.

    W : float
        Signal bandwidth.

    tau : float
        Cycle / frame duration.

    Derived parameters
    ------------------
    SNR : float
        Linear-scale SNR.

    N : int
        Number of harmonics.

    M_rbcp : int
        Common feasible number of RbCP bins.

    M_rbcp_per_sensor : np.ndarray
        Feasible RbCP bins per sensor.

    M_time : int
        Number of temporal resources for SFC:
            M_time = floor(tau * B / R)

    N0 : float
        Noise parameter derived from:
            SNR = P / (B * N0)

    E_tot : float
        Total available energy per cycle:
            E_tot = P * tau

    E_s : float
        Energy per transmitted symbol/resource:
            E_s = (P * tau) / L

    signal_level : float
        Expected matched-filter signal amplitude:
            sqrt(E_s)

    default_threshold : float
        Default detector threshold:
            threshold_factor * signal_level

    bandwidth_allocation : np.ndarray
        Fraction of total bandwidth assigned to each sensor.

    B_per_sensor : np.ndarray
        Effective bandwidth assigned to each sensor.
    """

    # primary inputs
    S: int
    P: float
    B: float
    R: int
    L: int
    SNR_dB: float
    W: float
    tau: float

    # derived quantities
    SNR: float
    N: int
    M_rbcp: int
    M_rbcp_per_sensor: np.ndarray
    M_time: int
    N0: float
    E_tot: float
    E_s: float
    signal_level: float
    default_threshold: float
    bandwidth_allocation: np.ndarray
    B_per_sensor: np.ndarray


# =============================================================================
# CENTRAL BUILDER
# =============================================================================

def build_derived_system_parameters(cfg, threshold_factor: Optional[float] = None):
    """
    Build the full set of derived system parameters from the configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with:
            cfg["system"]["S"]
            cfg["system"]["P"]
            cfg["system"]["B"]
            cfg["system"]["R"]
            cfg["system"]["L"]
            cfg["system"]["SNR_dB"]
            cfg["signal"]["W"]
            cfg["signal"]["tau"]

        Optional:
            cfg["system"]["bandwidth_allocation"]
            cfg["channel"]["threshold_factor"]

    threshold_factor : float or None, optional
        Factor used to derive the default detector threshold.

        If None, it is read from:
            cfg["channel"]["threshold_factor"]
        and defaults to 0.5 if absent.

    Returns
    -------
    DerivedSystemParameters
        Frozen dataclass containing all primary and derived parameters.
    """

    # -------------------------------------------------------------------------
    # Primary parameters
    # -------------------------------------------------------------------------
    S = cfg["system"]["S"]
    P = cfg["system"]["P"]
    B = cfg["system"]["B"]
    R = cfg["system"]["R"]
    L = cfg["system"]["L"]
    SNR_dB = cfg["system"]["SNR_dB"]

    W = cfg["signal"]["W"]
    tau = cfg["signal"]["tau"]

    bandwidth_allocation = cfg["system"].get("bandwidth_allocation", None)

    # -------------------------------------------------------------------------
    # Threshold factor source
    # -------------------------------------------------------------------------
    if threshold_factor is None:
        threshold_factor = cfg.get("channel", {}).get("threshold_factor", 0.5)

    # -------------------------------------------------------------------------
    # Derived quantities from theory.py
    # -------------------------------------------------------------------------
    SNR = theory_compute_snr_linear(SNR_dB)

    N = theory_compute_N(W, tau)

    alloc, B_per_sensor = theory_compute_sensor_bandwidths(
        B=B,
        S=S,
        bandwidth_allocation=bandwidth_allocation
    )

    M_rbcp_per_sensor = theory_compute_M_rbcp_per_sensor(
        S=S,
        W=W,
        tau=tau,
        B=B,
        SNR=SNR,
        bandwidth_allocation=bandwidth_allocation
    )

    M_rbcp = theory_compute_M_rbcp(
        S=S,
        W=W,
        tau=tau,
        B=B,
        SNR=SNR,
        bandwidth_allocation=bandwidth_allocation
    )

    M_time = theory_compute_M_time(tau, B, R)

    # IMPORTANT:
    # As specified in the project, N0 uses the TOTAL system bandwidth B.
    N0 = theory_compute_N0(P, B, SNR)

    E_tot = theory_compute_total_energy(P, tau)
    E_s = theory_compute_symbol_energy(P, tau, L)
    signal_level = theory_compute_signal_level(P, tau, L)

    default_threshold = theory_compute_default_detection_threshold(
        P=P,
        tau=tau,
        L=L,
        threshold_factor=threshold_factor,
    )

    return DerivedSystemParameters(
        S=S,
        P=P,
        B=B,
        R=R,
        L=L,
        SNR_dB=SNR_dB,
        W=W,
        tau=tau,
        SNR=SNR,
        N=N,
        M_rbcp=M_rbcp,
        M_rbcp_per_sensor=M_rbcp_per_sensor,
        M_time=M_time,
        N0=N0,
        E_tot=E_tot,
        E_s=E_s,
        signal_level=signal_level,
        default_threshold=default_threshold,
        bandwidth_allocation=alloc,
        B_per_sensor=B_per_sensor,
    )


# =============================================================================
# BACKWARD-COMPATIBILITY WRAPPERS
# =============================================================================
#
# These wrappers preserve old imports such as:
#   from sfc.core.system_parameters import compute_N, compute_M_rbcp, ...
#
# Internally, they simply delegate to theory.py.
# =============================================================================

def compute_N(W: float, tau: float) -> int:
    """
    Backward-compatible wrapper for theory.compute_N().
    """
    return theory_compute_N(W, tau)


def compute_capacity(B: float, SNR: float) -> float:
    """
    Backward-compatible wrapper for theory.compute_capacity().
    """
    return theory_compute_capacity(B, SNR)


def compute_snr_linear(SNR_dB: float) -> float:
    """
    Backward-compatible wrapper for theory.compute_snr_linear().
    """
    return theory_compute_snr_linear(SNR_dB)


def compute_bandwidth_allocation(
    S: int,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Backward-compatible wrapper for theory.compute_bandwidth_allocation().
    """
    return theory_compute_bandwidth_allocation(S, bandwidth_allocation)


def compute_sensor_bandwidths(
    B: float,
    S: int,
    bandwidth_allocation: Optional[Sequence[float]] = None
):
    """
    Backward-compatible wrapper for theory.compute_sensor_bandwidths().
    """
    return theory_compute_sensor_bandwidths(B, S, bandwidth_allocation)


def compute_M_rbcp(
    S: int,
    W: float,
    tau: float,
    B: float,
    SNR: float,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> int:
    """
    Backward-compatible wrapper for theory.compute_M_rbcp().
    """
    return theory_compute_M_rbcp(
        S=S,
        W=W,
        tau=tau,
        B=B,
        SNR=SNR,
        bandwidth_allocation=bandwidth_allocation
    )


def compute_M_rbcp_per_sensor(
    S: int,
    W: float,
    tau: float,
    B: float,
    SNR: float,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Backward-compatible wrapper for theory.compute_M_rbcp_per_sensor().
    """
    return theory_compute_M_rbcp_per_sensor(
        S=S,
        W=W,
        tau=tau,
        B=B,
        SNR=SNR,
        bandwidth_allocation=bandwidth_allocation
    )


def compute_M_time(tau: float, B: float, R: int) -> int:
    """
    Backward-compatible wrapper for theory.compute_M_time().
    """
    return theory_compute_M_time(tau, B, R)


def compute_N0(P: float, B: float, SNR: float) -> float:
    """
    Backward-compatible wrapper for theory.compute_N0().
    """
    return theory_compute_N0(P, B, SNR)


def compute_total_energy(P: float, tau: float) -> float:
    """
    Backward-compatible wrapper for theory.compute_total_energy().
    """
    return theory_compute_total_energy(P, tau)


def compute_symbol_energy(P: float, tau: float, L: int) -> float:
    """
    Backward-compatible wrapper for theory.compute_symbol_energy().
    """
    return theory_compute_symbol_energy(P, tau, L)


def compute_signal_level(P: float, tau: float, L: int) -> float:
    """
    Backward-compatible wrapper for theory.compute_signal_level().
    """
    return theory_compute_signal_level(P, tau, L)


def compute_default_detection_threshold(
    P: float,
    tau: float,
    L: int,
    threshold_factor: float = 0.5
) -> float:
    """
    Backward-compatible wrapper for theory.compute_default_detection_threshold().
    """
    return theory_compute_default_detection_threshold(P, tau, L, threshold_factor)


# =============================================================================
# OPTIONAL EXPORT LIST
# =============================================================================

__all__ = [
    "DerivedSystemParameters",
    "build_derived_system_parameters",

    # backward-compatible wrappers
    "compute_N",
    "compute_capacity",
    "compute_snr_linear",
    "compute_bandwidth_allocation",
    "compute_sensor_bandwidths",
    "compute_M_rbcp",
    "compute_M_rbcp_per_sensor",
    "compute_M_time",
    "compute_N0",
    "compute_total_energy",
    "compute_symbol_energy",
    "compute_signal_level",
    "compute_default_detection_threshold",
]
