"""
rbcp.py

Core mathematical functions for RbCP (Representation by Cosine Phase)

This module implements the theoretical expressions used in Figure 1.

All functions are deterministic and stateless.

Author: SFC Project
"""

import numpy as np


def compute_Q(M):
    """
    Compute Q parameter as defined in the manuscript.

    Q = (M / (2*pi)) * sin(pi / M)

    Parameters
    ----------
    M : int
        Number of quantization bins

    Returns
    -------
    float
        Q parameter
    """
    return (M / (2 * np.pi)) * np.sin(np.pi / M)


def mse_upper_bound(N, Q):
    """
    Upper bound from Lemma 4.

    MSE <= 4N * (1/2 - Q) * (3/2 - Q)

    Parameters
    ----------
    N : int
        Number of harmonics
    Q : float
        Q parameter

    Returns
    -------
    float
        Upper bound MSE
    """
    return 4 * N * (0.5 - Q) * (1.5 - Q)


def mse_star(N, Q):
    """
    Analytical approximation (Proposition 2).

    MSE* = 2N * (1 - 2Q)

    Parameters
    ----------
    N : int
        Number of harmonics
    Q : float
        Q parameter

    Returns
    -------
    float
        MSE star
    """
    return 2 * N * (1 - 2 * Q)
