"""
sfc/core/system_parameters.py

Core module for deriving system parameters based on the physical model.

This file centralizes:
- number of harmonics (N)
- channel capacity
- number of RbCP bins (M_RbCP)
- number of temporal resources for SFC (M_time)
- physical quantities derived from:
    P, B, R, L, SNR_dB, W, tau

Design goal
-----------
Avoid duplicated physical calculations across:
- physical_channel.py
- detection.py
- figure pipelines
- future SFC modules

Single source of truth
----------------------
All physical/system-derived quantities should be computed here and then
consumed by the other modules.
"""

import numpy as np
from dataclasses import dataclass


# =============================================================================
# DERIVED SYSTEM PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class DerivedSystemParameters:
    """
    Immutable container for system-level derived parameters.

    Primary parameters
    ------------------
    S : int
        Number of sensors/signals.

    P : float
        Average transmit power per sensor.

    B : float
        Channel bandwidth.

    R : int
        Number of resources / subcarriers.

    L : int
        Number of temporal resources / symbols per cycle.

    SNR_dB : float
        Signal-to-noise ratio in dB, as stored in the cfg.

    W : float
        Signal bandwidth.

    tau : float
        Duration of one cycle / frame.

    Derived parameters
    ------------------
    SNR : float
        Linear-scale SNR.

    N : int
        Number of harmonics.

    M_rbcp : int
        Number of quantization bins for the RbCP representation.

    M_time : int
        Number of temporal resources for SFC:
            M_time = tau * B / R

    N0 : float
        Noise parameter derived from:
            SNR = P / (B * N0)

    E_tot : float
        Total available energy per cycle:
            E_tot = P * tau

    E_s : float
        Energy per transmitted symbol/resource:
            E_s = E_tot / L = (P * tau) / L

    signal_level : float
        Expected amplitude scale at the matched-filter output:
            sqrt(E_s)

    default_threshold : float
        Default detection threshold derived from:
            threshold_factor * signal_level
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
    M_time: int
    N0: float
    E_tot: float
    E_s: float
    signal_level: float
    default_threshold: float


# =============================================================================
# N (NUMBER OF HARMONICS)
# =============================================================================

def compute_N(W, tau):
    """
    Compute the number of harmonics.

        N = floor(W * tau / 2)

    Parameters
    ----------
    W : float
        Signal bandwidth.

    tau : float
        Duration of observation frame / cycle.

    Returns
    -------
    int
        Number of harmonics.
    """
    return int(np.floor((W * tau) / 2))


# =============================================================================
# CHANNEL CAPACITY
# =============================================================================

def compute_capacity(B, SNR):
    """
    Compute the Shannon channel capacity.

        C = B * log2(1 + SNR)

    Parameters
    ----------
    B : float
        Channel bandwidth.

    SNR : float
        Signal-to-noise ratio in linear scale.

    Returns
    -------
    float
        Capacity in bits per second.
    """
    return B * np.log2(1 + SNR)


# =============================================================================
# RBCP BINS
# =============================================================================

def compute_M_rbcp(S, W, tau, B, SNR):
    """
    Compute the number of quantization bins for RbCP.

    Derived from the communication constraint:
        2 * N * S * log2(M_RbCP) <= tau * B * log2(1 + SNR)

    Therefore:
        bits_per_symbol = [tau * B * log2(1 + SNR)] / [2 * N * S]

        M_RbCP = 2 ^ floor(bits_per_symbol)

    Parameters
    ----------
    S : int
        Number of sensors/signals.

    W : float
        Signal bandwidth.

    tau : float
        Frame duration.

    B : float
        Channel bandwidth.

    SNR : float
        Signal-to-noise ratio in linear scale.

    Returns
    -------
    int
        Number of bins (power of 2).
    """

    N = compute_N(W, tau)

    capacity_bits = tau * compute_capacity(B, SNR)

    bits_per_symbol = capacity_bits / (2 * N * S)

    bits_per_symbol = max(bits_per_symbol, 1e-12)
    bits_int = int(np.floor(bits_per_symbol))

    return int(2 ** bits_int)


# =============================================================================
# SFC TEMPORAL RESOURCES (NOT BENCHMARK)
# =============================================================================

def compute_M_time(tau, B, R):
    """
    Compute the temporal resources for SFC.

        M_time = tau * B / R

    Parameters
    ----------
    tau : float
        Frame duration.

    B : float
        Channel bandwidth.

    R : int
        Number of resources / subcarriers.

    Returns
    -------
    int
        Number of temporal resources for SFC.
    """
    return int(np.floor((tau * B) / R))


# =============================================================================
# SNR CONVERSION
# =============================================================================

def compute_snr_linear(SNR_dB):
    """
    Convert SNR from dB to linear scale.

    Parameters
    ----------
    SNR_dB : float
        SNR in dB.

    Returns
    -------
    float
        SNR in linear scale.
    """
    return 10 ** (SNR_dB / 10.0)


# =============================================================================
# NOISE PARAMETER
# =============================================================================

def compute_N0(P, B, SNR):
    """
    Compute the noise parameter N0 from:

        SNR = P / (B * N0)

    Therefore:

        N0 = P / (B * SNR)

    Parameters
    ----------
    P : float
        Average transmit power.

    B : float
        Channel bandwidth.

    SNR : float
        Signal-to-noise ratio in linear scale.

    Returns
    -------
    float
        Noise parameter N0.
    """
    return P / (B * SNR)


# =============================================================================
# ENERGY
# =============================================================================

def compute_total_energy(P, tau):
    """
    Compute total available energy per cycle:

        E_tot = P * tau

    Parameters
    ----------
    P : float
        Average transmit power.

    tau : float
        Cycle / frame duration.

    Returns
    -------
    float
        Total available energy per cycle.
    """
    return P * tau


def compute_symbol_energy(P, tau, L):
    """
    Compute the average energy per transmitted symbol/resource.

    Under the current modeling assumption, the total available energy over one
    cycle is uniformly distributed across the L transmitted symbols/resources:

        E_s = (P * tau) / L

    Parameters
    ----------
    P : float
        Average transmit power.

    tau : float
        Cycle / frame duration.

    L : int
        Number of temporal symbols/resources per cycle.

    Returns
    -------
    float
        Energy per symbol/resource.
    """
    return (P * tau) / L


def compute_signal_level(P, tau, L):
    """
    Compute the expected matched-filter output amplitude scale:

        signal_level = sqrt(E_s)

    where:
        E_s = (P * tau) / L

    Parameters
    ----------
    P : float
        Average transmit power.

    tau : float
        Cycle / frame duration.

    L : int
        Number of symbols/resources per cycle.

    Returns
    -------
    float
        Expected signal amplitude level.
    """
    E_s = compute_symbol_energy(P, tau, L)
    return np.sqrt(E_s)


def compute_default_detection_threshold(P, tau, L, threshold_factor=0.5):
    """
    Compute the default detection threshold.

        threshold = threshold_factor * sqrt(E_s)

    where:
        E_s = (P * tau) / L

    Parameters
    ----------
    P : float
        Average transmit power.

    tau : float
        Cycle / frame duration.

    L : int
        Number of symbols/resources per cycle.

    threshold_factor : float, optional
        Multiplicative factor applied to the expected signal level.

    Returns
    -------
    float
        Default detection threshold.
    """
    return threshold_factor * compute_signal_level(P, tau, L)


# =============================================================================
# FULL BUILDER
# =============================================================================

def build_derived_system_parameters(cfg, threshold_factor=None):
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

    threshold_factor : float, optional
        Factor used to derive the default detector threshold.

    Returns
    -------
    DerivedSystemParameters
        Frozen dataclass containing all primary and derived parameters.
    """

    # -------------------------------------------------------------------------
    # Primary parameters from cfg
    # -------------------------------------------------------------------------
    S = cfg["system"]["S"]
    P = cfg["system"]["P"]
    B = cfg["system"]["B"]
    R = cfg["system"]["R"]
    L = cfg["system"]["L"]
    SNR_dB = cfg["system"]["SNR_dB"]

    W = cfg["signal"]["W"]
    tau = cfg["signal"]["tau"]

    if threshold_factor is None:
        threshold_factor = cfg.get("channel", {}).get("threshold_factor", 0.5)

    # -------------------------------------------------------------------------
    # Derived
    # -------------------------------------------------------------------------
    SNR = compute_snr_linear(SNR_dB)

    N = compute_N(W, tau)
    M_rbcp = compute_M_rbcp(S, W, tau, B, SNR)
    M_time = compute_M_time(tau, B, R)

    N0 = compute_N0(P, B, SNR)

    E_tot = compute_total_energy(P, tau)
    E_s = compute_symbol_energy(P, tau, L)
    signal_level = compute_signal_level(P, tau, L)

    default_threshold = compute_default_detection_threshold(
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
        M_time=M_time,
        N0=N0,
        E_tot=E_tot,
        E_s=E_s,
        signal_level=signal_level,
        default_threshold=default_threshold,
    )
