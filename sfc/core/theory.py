"""
sfc/core/theory.py

Trusted analytical/theoretical formulas extracted from the manuscript.

This module centralizes deterministic, stateless formulas used for:
- Benchmark / Nyquist theoretical calculations
- RbCP theoretical calculations
- Time-domain reporting theoretical calculations
- SFC overlap upper bound
- Latency expressions
- Capacity-induced quantization-bin relations

IMPORTANT
---------
This file is intended to contain ONLY closed-form or direct analytical
expressions from the manuscript.

Allowed contents:
- deterministic mathematical formulas
- simple helper functions to combine manuscript equations
- numerically stable implementations of the exact same formulas

Forbidden contents:
- Monte Carlo simulation logic
- file I/O
- plotting
- experiment orchestration
- any approximation not explicitly documented

Design choice
-------------
Functions are used instead of classes because these formulas are stateless,
deterministic, and easier to audit in a functional style.
"""

from __future__ import annotations

import math
import numpy as np


# =============================================================================
# Basic channel/capacity formulas
# =============================================================================

def shannon_capacity(B: float, P: float, N0: float) -> float:
    """
    Compute the Shannon capacity used in the manuscript:

        C = B * log2(1 + P / (B * N0))

    Parameters
    ----------
    B : float
        Channel bandwidth.

    P : float
        Average transmit power.

    N0 : float
        One-sided noise spectral density parameter as used in the manuscript.

    Returns
    -------
    float
        Shannon capacity in bits per second.
    """

    return B * np.log2(1.0 + P / (B * N0))


def benchmark_rate(W: float, M: float, S: int = 1) -> float:
    """
    Benchmark/Nyquist rate:

        R_BA = W * S * log2(M)

    Parameters
    ----------
    W : float
        Signal bandwidth parameter used in the manuscript.

    M : float
        Number of quantization bins for the Benchmark approach.

    S : int, optional
        Number of users/sensors.

    Returns
    -------
    float
        Required rate in bits per second.
    """

    return W * S * np.log2(M)


def rbcp_rate(M_rbcp: float, N: int, tau: float, S: int = 1) -> float:
    """
    RbCP reporting rate:

        R_RbCP = (2 * N * S / tau) * log2(M_RbCP)

    Parameters
    ----------
    M_rbcp : float
        Number of bins used to quantize the phase parameters.

    N : int
        Number of harmonics.

    tau : float
        Observation window / signal period.

    S : int, optional
        Number of users/sensors.

    Returns
    -------
    float
        Required rate in bits per second.
    """

    return (2.0 * N * S / tau) * np.log2(M_rbcp)


def benchmark_max_bins(B: float, P: float, N0: float, W: float, S: int = 1) -> float:
    """
    Maximum Benchmark/Nyquist number of bins under the manuscript capacity model.

    From:
        W * S * log2(M) <= B * log2(1 + P / (B * N0))

    Therefore:
        M <= (1 + P / (B * N0)) ** (B / (W * S))

    Parameters
    ----------
    B : float
        Channel bandwidth.

    P : float
        Average transmit power.

    N0 : float
        Noise spectral density parameter.

    W : float
        Signal bandwidth parameter.

    S : int, optional
        Number of users/sensors.

    Returns
    -------
    float
        Maximum admissible number of bins for the Benchmark approach.
    """

    return (1.0 + P / (B * N0)) ** (B / (W * S))


def rbcp_max_bins(B: float, P: float, N0: float, N: int, tau: float, S: int = 1) -> float:
    """
    Maximum RbCP number of bins under the manuscript capacity model.

    From:
        (2 * N * S / tau) * log2(M_RbCP) <= B * log2(1 + P / (B * N0))

    Therefore:
        M_RbCP <= (1 + P / (B * N0)) ** (tau * B / (2 * N * S))

    Parameters
    ----------
    B : float
        Channel bandwidth.

    P : float
        Average transmit power.

    N0 : float
        Noise spectral density parameter.

    N : int
        Number of harmonics.

    tau : float
        Observation window / signal period.

    S : int, optional
        Number of users/sensors.

    Returns
    -------
    float
        Maximum admissible number of bins for RbCP.
    """

    return (1.0 + P / (B * N0)) ** (tau * B / (2.0 * N * S))


# =============================================================================
# Figure-1 / Figure-2 related formulas
# =============================================================================

def compute_q(M: float) -> float:
    """
    Compute the Q factor used in the manuscript:

        Q = (M / (2*pi)) * sin(pi / M)

    Parameters
    ----------
    M : float
        Number of quantization bins.

    Returns
    -------
    float
        Q factor.
    """

    return (M / (2.0 * np.pi)) * np.sin(np.pi / M)


def benchmark_mse(M: float) -> float:
    """
    Benchmark/Nyquist normalized MSE:

        MSE_{x,y} = 1 / (12 * M^2)

    Parameters
    ----------
    M : float
        Number of quantization bins.

    Returns
    -------
    float
        Normalized MSE of the Benchmark approach.
    """

    return 1.0 / (12.0 * M**2)


def rbcp_mse_general(N: int, Q: float, I) -> float:
    """
    General RbCP MSE from the manuscript (Lemma 3 / Eq. 12):

        MSE_RbCP = 2N - sum_n [4Q - (2Q - 1)^2 * I_n]

    Parameters
    ----------
    N : int
        Number of harmonics.

    Q : float
        Q factor defined by the manuscript.

    I : float or array-like
        Integral term from the joint distribution of t_a and t_b.
        If scalar, the same value is used for all harmonics.
        If array-like, it must contain one value per harmonic.

    Returns
    -------
    float
        Theoretical RbCP MSE.
    """

    I_arr = np.asarray(I, dtype=float)

    if I_arr.ndim == 0:
        I_arr = np.full(N, float(I_arr))

    if I_arr.shape[0] != N:
        raise ValueError("I must be scalar or have length N.")

    return 2.0 * N - np.sum(4.0 * Q - (2.0 * Q - 1.0) ** 2 * I_arr)


def rbcp_mse_lower_bound(N: int, Q: float) -> float:
    """
    Lower bound from Lemma 4:

        (MSE_RbCP / N) >= 1 - 4Q^2

    Therefore:
        MSE_RbCP >= N * (1 - 4Q^2)

    Parameters
    ----------
    N : int
        Number of harmonics.

    Q : float
        Q factor.

    Returns
    -------
    float
        Lower bound on MSE_RbCP.
    """

    return N * (1.0 - 4.0 * Q**2)


def rbcp_mse_upper_bound(N: int, Q: float) -> float:
    """
    Upper bound from Lemma 4:

        (MSE_RbCP / N) <= 4 * (1/2 - Q) * (3/2 - Q)

    Therefore:
        MSE_RbCP <= 4N * (1/2 - Q) * (3/2 - Q)

    Parameters
    ----------
    N : int
        Number of harmonics.

    Q : float
        Q factor.

    Returns
    -------
    float
        Upper bound on MSE_RbCP.
    """

    return 4.0 * N * (0.5 - Q) * (1.5 - Q)


def rbcp_mse_star(N: int, Q: float) -> float:
    """
    Proposition 2:

        MSE*_RbCP = 2N * (1 - 2Q)

    Parameters
    ----------
    N : int
        Number of harmonics.

    Q : float
        Q factor.

    Returns
    -------
    float
        Analytical MSE* under the manuscript uniformity assumptions.
    """

    return 2.0 * N * (1.0 - 2.0 * Q)


# =============================================================================
# Bin relations used in the manuscript
# =============================================================================

def xi_from_bandwidth(W: float, w0: float) -> float:
    """
    Compute the xi term used in Eq. (17):

        xi = pi*W / w0 - floor(pi*W / w0)

    Parameters
    ----------
    W : float
        Signal bandwidth parameter.

    w0 : float
        Fundamental angular frequency.

    Returns
    -------
    float
        Fractional spectral remainder xi in [0, 1).
    """

    return np.pi * W / w0 - np.floor(np.pi * W / w0)


def rbcp_bins_from_benchmark(M: float, W: float, w0: float, tau: float | None = None) -> float:
    """
    Compute M_RbCP from M using the manuscript relation (Eq. 17).

    The manuscript gives:
        M_RbCP = M * (tau * W) / (2N)

    Using tau = 2*pi / w0 and N = floor(pi*W / w0), the equivalent expression is:
        M_RbCP = M * (pi*W) / (pi*W - xi*w0)

    Parameters
    ----------
    M : float
        Benchmark number of bins.

    W : float
        Signal bandwidth parameter.

    w0 : float
        Fundamental angular frequency.

    tau : float or None, optional
        Signal period / observation window. If None, tau is inferred as 2*pi / w0.

    Returns
    -------
    float
        Manuscript-equivalent number of RbCP bins.
    """

    if tau is None:
        tau = 2.0 * np.pi / w0

    N = int(np.floor(np.pi * W / w0))
    return M * tau * W / (2.0 * N)


def time_bins(tau: float, B: float, R: float) -> float:
    """
    Number of time bins used by the Time abstraction:

        M_time = tau * B / R

    Parameters
    ----------
    tau : float
        Signal period / observation window.

    B : float
        Channel bandwidth.

    R : float
        Number of channel uses per symbol (as defined in the manuscript).

    Returns
    -------
    float
        Number of time bins.
    """

    return tau * B / R


def time_quantization_interval(n: int, w0: float, tau: float, B: float, R: float) -> float:
    """
    Time-domain quantization interval from the manuscript:

        l_time = 2*pi*R / (n * w0 * tau * B)

    Parameters
    ----------
    n : int
        Harmonic index.

    w0 : float
        Fundamental angular frequency.

    tau : float
        Signal period / observation window.

    B : float
        Channel bandwidth.

    R : float
        Number of channel uses per symbol.

    Returns
    -------
    float
        Time-domain quantization interval.
    """

    return 2.0 * np.pi * R / (n * w0 * tau * B)


# =============================================================================
# SFC overlap / duplicate-event upper bound
# =============================================================================

def sfc_duplicate_probability_upper_bound(m_time: int, N: int, S: int) -> float:
    """
    Lemma 5 upper bound for the probability of receiving duplicate values.

    The manuscript gives:
        epsilon <= 1 - M_time! / ((M_time - 2NS)! * M_time^(2NS))

    This implementation is mathematically equivalent, but computed through
    log-gamma for numerical stability.

    Parameters
    ----------
    m_time : int
        Number of time bins (M_time).

    N : int
        Number of harmonics.

    S : int
        Number of sensors/users.

    Returns
    -------
    float
        Upper bound on epsilon.

    Notes
    -----
    If m_time < 2*N*S, the exact factorial expression is not defined because
    (M_time - 2NS)! would be negative. In that regime, the bound effectively
    saturates to 1 for practical purposes, since collision-free placement of
    all events is impossible.
    """

    total_events = 2 * N * S

    if m_time < total_events:
        return 1.0

    log_num = math.lgamma(m_time + 1)
    log_den = math.lgamma(m_time - total_events + 1) + total_events * math.log(m_time)

    epsilon0 = math.exp(log_num - log_den)
    return 1.0 - epsilon0


# =============================================================================
# Latency formulas
# =============================================================================

def rbcp_total_latency(tau: float, N: int, M_rbcp: float, C: float) -> float:
    """
    Total latency of the RbCP approach (Eq. 24):

        L_total_RbCP = tau + (2N * log2(M_RbCP)) / C

    Parameters
    ----------
    tau : float
        Observation window / sensing latency.

    N : int
        Number of harmonics.

    M_rbcp : float
        Number of RbCP quantization bins.

    C : float
        Channel capacity.

    Returns
    -------
    float
        Total RbCP latency.
    """

    return tau + (2.0 * N * np.log2(M_rbcp)) / C


def benchmark_total_latency(W: float, tau: float, M: float, C: float) -> float:
    """
    Total latency of the Benchmark approach (Eq. 25):

        L_total_BA = (W * tau * log2(M)) / C + 1/W

    Parameters
    ----------
    W : float
        Signal bandwidth parameter.

    tau : float
        Observation window / signal period.

    M : float
        Number of Benchmark quantization bins.

    C : float
        Channel capacity.

    Returns
    -------
    float
        Total Benchmark latency.
    """

    return (W * tau * np.log2(M)) / C + 1.0 / W


def rbcp_has_lower_latency_than_benchmark(
    tau: float,
    N: int,
    M_rbcp: float,
    W: float,
    M: float,
    C: float
) -> bool:
    """
    Evaluate the latency inequality discussed in the manuscript:

        tau + (2N * log2(M_RbCP))/C < (W*tau*log2(M))/C + 1/W

    Parameters
    ----------
    tau : float
        Observation window / signal period.

    N : int
        Number of harmonics.

    M_rbcp : float
        RbCP number of bins.

    W : float
        Signal bandwidth parameter.

    M : float
        Benchmark number of bins.

    C : float
        Channel capacity.

    Returns
    -------
    bool
        True if the RbCP latency is lower than the Benchmark latency.
    """

    return rbcp_total_latency(tau, N, M_rbcp, C) < benchmark_total_latency(W, tau, M, C)