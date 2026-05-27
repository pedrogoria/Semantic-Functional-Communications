"""
sfc/core/system_parameters.py

Core module for deriving system parameters based on physical model.

Defines:

- number of harmonics (N)
- number of RbCP bins (M_RbCP)
- number of temporal samples (M_time, for SFC future models)

All derivations strictly follow the manuscript assumptions.
"""

import numpy as np


# =============================================================================
# N (NUMBER OF HARMONICS)
# =============================================================================

def compute_N(W, tau):
    """
    Compute number of harmonics.

    N = floor(W * tau / 2)

    Parameters
    ----------
    W : float
        Signal bandwidth.

    tau : float
        Duration of observation frame.

    Returns
    -------
    int
    """
    return int(np.floor((W * tau) / 2))


# =============================================================================
# CHANNEL CAPACITY
# =============================================================================

def compute_capacity(B, SNR):
    """
    Shannon capacity.

    C = B log2(1 + SNR)

    Parameters
    ----------
    B : float
        Channel bandwidth.

    SNR : float
        Signal-to-noise ratio (linear scale).

    Returns
    -------
    float
        Capacity in bits per second.
    """
    return B * np.log2(1 + SNR)


# =============================================================================
# RBCP BINS (EQ. 20)
# =============================================================================

def compute_M_rbcp(S, W, tau, B, SNR):
    """
    Compute number of quantization bins for RbCP (Eq. 20).

    Derived from:
        2 N S log2(M_RbCP) <= tau * B log2(1 + SNR)

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
        Signal-to-noise ratio (linear).

    Returns
    -------
    int
        Number of bins (power of 2).
    """

    N = compute_N(W, tau)

    capacity_bits = tau * compute_capacity(B, SNR)

    bits_per_symbol = capacity_bits / (2 * N * S)

    # Convert to power of two
    bits_per_symbol = max(bits_per_symbol, 1e-12)

    bits_int = int(np.floor(bits_per_symbol))

    return int(2 ** bits_int)


# =============================================================================
# SFC TEMPORAL RESOURCES (NOT BENCHMARK)
# =============================================================================

def compute_M_time(tau, B, R):
    """
    Compute temporal resources for SFC (NOT benchmark).

    M_time = tau * B / R

    Parameters
    ----------
    tau : float
        Frame duration.

    B : float
        Channel bandwidth.

    R : int
        Number of subcarriers/resources.

    Returns
    -------
    int
    """
    return int(np.floor((tau * B) / R))
