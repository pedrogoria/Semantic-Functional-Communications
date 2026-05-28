"""
sfc/core/theory.py

Central repository for closed-form theoretical relations used across the project.

This module centralizes formulas from the manuscript, including:
- number of harmonics N
- Shannon capacity
- RbCP Q-factor
- MSE bounds / approximations for RbCP
- relations between M and M_RbCP
- common / per-sensor feasible M_RbCP under bandwidth sharing
- benchmark bits-per-sample under capacity constraints
- SFC time-slot relations
- duplicate-reception upper bound (Lemma 5)
- physical helper formulas (SNR, N0, energy, thresholds)

Design rule
-----------
Pipelines, channel modules, debug scripts, and builders should import the
theoretical relations from this file instead of reimplementing formulas locally.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# BASIC SIGNAL / CHANNEL THEORY
# =============================================================================

def compute_N(W: float, tau: float) -> int:
    """
    Number of harmonics.

        N = floor(W * tau / 2)

    Parameters
    ----------
    W : float
        Signal bandwidth.
    tau : float
        Cycle / frame duration.

    Returns
    -------
    int
        Number of harmonics.
    """
    return int(np.floor((W * tau) / 2))


def compute_snr_linear(SNR_dB: float) -> float:
    """
    Convert SNR from dB to linear scale.
    """
    return 10 ** (SNR_dB / 10.0)


def compute_capacity(B: float, SNR: float) -> float:
    """
    Shannon capacity.

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
# RbCP THEORY (LEMMA 4 / PROPOSITION 2)
# =============================================================================

def compute_q(M_rbcp: int) -> float:
    """
    Compute the Q term used in Lemma 3 / Lemma 4 / Proposition 2:

        Q = (M_RbCP / (2*pi)) * sin(pi / M_RbCP)

    Parameters
    ----------
    M_rbcp : int
        Number of quantization bins for the RbCP representation.

    Returns
    -------
    float
        Q factor.
    """
    return (M_rbcp / (2.0 * np.pi)) * np.sin(np.pi / M_rbcp)


def rbcp_mse_upper_bound(N: int, Q: float) -> float:
    """
    Lemma 4 upper bound for MSE_RbCP:

        MSE_RbCP / N <= 4 * (1/2 - Q) * (3/2 - Q)

    Therefore:

        MSE_upper = 4 * N * (1/2 - Q) * (3/2 - Q)

    Parameters
    ----------
    N : int
        Number of harmonics.
    Q : float
        Q factor.

    Returns
    -------
    float
        Upper bound on the RbCP MSE.
    """
    return 4.0 * N * (0.5 - Q) * (1.5 - Q)


def rbcp_mse_lower_bound(N: int, Q: float) -> float:
    """
    Lemma 4 lower bound for MSE_RbCP:

        MSE_RbCP / N >= 1 - 4Q^2

    Therefore:

        MSE_lower = N * (1 - 4Q^2)

    Parameters
    ----------
    N : int
        Number of harmonics.
    Q : float
        Q factor.

    Returns
    -------
    float
        Lower bound on the RbCP MSE.
    """
    return N * (1.0 - 4.0 * Q * Q)


def rbcp_mse_star(N: int, Q: float) -> float:
    """
    Proposition 2:

        MSE*_RbCP = 2N (1 - 2Q)

    Parameters
    ----------
    N : int
        Number of harmonics.
    Q : float
        Q factor.

    Returns
    -------
    float
        MSE*_RbCP.
    """
    return 2.0 * N * (1.0 - 2.0 * Q)


# =============================================================================
# RELATION BETWEEN M AND M_RbCP (EQ. 17)
# =============================================================================

def compute_M_rbcp_from_M(M: float, W: float, tau: float) -> float:
    """
    Equation (17):

        M_RbCP = M * tau * W / (2N)

    with:
        N = floor(W * tau / 2)

    This is equivalent to the manuscript relation:
        M_RbCP = M * pi*W / (pi*W - xi*w0)

    Parameters
    ----------
    M : float
        Number of Benchmark quantization bins.
    W : float
        Signal bandwidth.
    tau : float
        Frame duration.

    Returns
    -------
    float
        Corresponding M_RbCP value.
    """
    N = compute_N(W, tau)
    return M * tau * W / (2.0 * N)


def compute_M_from_M_rbcp(M_rbcp: float, W: float, tau: float) -> float:
    """
    Inverse of Equation (17):

        M = M_RbCP * (2N) / (tau * W)

    Parameters
    ----------
    M_rbcp : float
        Number of RbCP bins.
    W : float
        Signal bandwidth.
    tau : float
        Frame duration.

    Returns
    -------
    float
        Corresponding Benchmark M value.
    """
    N = compute_N(W, tau)
    return M_rbcp * (2.0 * N) / (tau * W)


# =============================================================================
# BANDWIDTH SHARING AMONG SENSORS
# =============================================================================

def compute_bandwidth_allocation(
    S: int,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Compute / validate the bandwidth-allocation vector.

    Rules
    -----
    - if None: equal split among S sensors
    - otherwise:
        * length must be S
        * entries must be nonnegative
        * sum must be 1

    Returns
    -------
    np.ndarray
        Allocation vector of length S.
    """
    if bandwidth_allocation is None:
        return np.ones(S, dtype=float) / S

    alloc = np.asarray(bandwidth_allocation, dtype=float)

    if alloc.ndim != 1:
        raise ValueError("bandwidth_allocation must be a 1D vector")

    if len(alloc) != S:
        raise ValueError(
            f"bandwidth_allocation must have length S={S}, got {len(alloc)}"
        )

    if np.any(alloc < 0):
        raise ValueError("bandwidth_allocation must contain nonnegative values")

    if not np.isclose(np.sum(alloc), 1.0):
        raise ValueError(
            f"bandwidth_allocation must sum to 1, got sum={np.sum(alloc)}"
        )

    return alloc


def compute_sensor_bandwidths(
    B: float,
    S: int,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute effective per-sensor bandwidths from the total bandwidth B.

    Returns
    -------
    tuple
        (allocation, B_per_sensor)
    """
    alloc = compute_bandwidth_allocation(S, bandwidth_allocation)
    B_per_sensor = B * alloc
    return alloc, B_per_sensor


# =============================================================================
# RbCP FEASIBLE NUMBER OF BINS UNDER BANDWIDTH SHARING
# =============================================================================

def compute_M_rbcp_single_sensor(W: float, tau: float, B_sensor: float, SNR: float) -> int:
    """
    Feasible number of RbCP bins for ONE sensor.

    For one sensor:
        2N log2(M_RbCP) <= tau * B_sensor * log2(1 + SNR)

    Parameters
    ----------
    W : float
        Signal bandwidth.
    tau : float
        Frame duration.
    B_sensor : float
        Bandwidth slice assigned to this sensor.
    SNR : float
        Signal-to-noise ratio in linear scale.

    Returns
    -------
    int
        Feasible M_RbCP for one sensor.
    """
    N = compute_N(W, tau)

    capacity_bits = tau * compute_capacity(B_sensor, SNR)
    bits_per_symbol = capacity_bits / (2.0 * N)

    bits_per_symbol = max(bits_per_symbol, 1e-12)
    bits_int = int(np.floor(bits_per_symbol))

    return int(2 ** bits_int)


def compute_M_rbcp_per_sensor(
    S: int,
    W: float,
    tau: float,
    B: float,
    SNR: float,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Per-sensor feasible M_RbCP values, accounting for bandwidth sharing.

    Returns
    -------
    np.ndarray
        Shape (S,)
    """
    _, B_per_sensor = compute_sensor_bandwidths(B, S, bandwidth_allocation)

    return np.array([
        compute_M_rbcp_single_sensor(W, tau, B_s, SNR)
        for B_s in B_per_sensor
    ], dtype=int)


def compute_M_rbcp(
    S: int,
    W: float,
    tau: float,
    B: float,
    SNR: float,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> int:
    """
    Common feasible M_RbCP across S sensors.

    If bandwidth is equally split, all sensors get the same value.
    If bandwidth is unevenly split, this returns the minimum per-sensor value
    so that it is safe as a common M_RbCP.

    This is the bandwidth-sharing-aware interpretation of the manuscript
    multiuser constraint.

    Returns
    -------
    int
        Common feasible M_RbCP.
    """
    M_vec = compute_M_rbcp_per_sensor(
        S=S,
        W=W,
        tau=tau,
        B=B,
        SNR=SNR,
        bandwidth_allocation=bandwidth_allocation,
    )
    return int(np.min(M_vec))


# =============================================================================
# BENCHMARK / NYQUIST COMMUNICATION BUDGET
# =============================================================================

def compute_benchmark_bits_per_sample_single_sensor(
    tau: float,
    B_sensor: float,
    SNR: float,
    sampling_rate: float
) -> int:
    """
    Feasible number of bits per sample for one Benchmark/Nyquist sensor.

    For one sensor:
        C_sensor = B_sensor * log2(1 + SNR)
        bits_total = tau * C_sensor
        num_samples = floor(tau * sampling_rate)
        bits_per_sample = bits_total / num_samples

    Returns
    -------
    int
        Feasible bits per sample.
    """
    C_sensor = compute_capacity(B_sensor, SNR)
    bits_total = tau * C_sensor

    num_samples = int(np.floor(tau * sampling_rate))
    if num_samples <= 0:
        raise ValueError("sampling_rate leads to zero samples in one cycle")

    bits_per_sample = bits_total / num_samples
    bits_int = int(np.floor(bits_per_sample))
    bits_int = max(bits_int, 1)

    return bits_int


def compute_benchmark_bits_per_sample_per_sensor(
    S: int,
    tau: float,
    B: float,
    SNR: float,
    sampling_rate: float,
    bandwidth_allocation: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Per-sensor feasible bits-per-sample for the Benchmark/Nyquist branch.

    Returns
    -------
    np.ndarray
        Shape (S,)
    """
    _, B_per_sensor = compute_sensor_bandwidths(B, S, bandwidth_allocation)

    return np.array([
        compute_benchmark_bits_per_sample_single_sensor(
            tau=tau,
            B_sensor=B_s,
            SNR=SNR,
            sampling_rate=sampling_rate
        )
        for B_s in B_per_sensor
    ], dtype=int)


# =============================================================================
# SFC TIME / SLOT RELATIONS
# =============================================================================

def compute_M_time(tau: float, B: float, R: int) -> int:
    """
    Number of time bins / symbol-start bins for the SFC time model:

        M_time = tau * B / R

    In the discrete simulation, we use the floor-consistent value:
        floor(tau * B / R)

    Returns
    -------
    int
        Discrete M_time used in the simulations.
    """
    return int(np.floor((tau * B) / R))


def compute_slot_duration(B: float, R: int) -> float:
    """
    Duration of one SFC symbol slot:

        T_slot = R / B

    Returns
    -------
    float
        Slot duration in seconds.
    """
    return R / B


def compute_slots_per_period(tau: float, B: float, R: int) -> int:
    """
    Number of possible event-start slots per period:

        floor(tau / (R/B)) = floor(tau * B / R)

    Returns
    -------
    int
        Slots per period.
    """
    return int(np.floor(tau * B / R))


def compute_event_slots_total(tau: float, B: float, R: int, n_periods: int) -> int:
    """
    Total number of possible event-start slots over n_periods.
    """
    return compute_slots_per_period(tau, B, R) * n_periods


def compute_rx_slots_total(tau: float, B: float, R: int, n_periods: int, L: int) -> int:
    """
    Total number of received discrete-time slots once the map length L is taken
    into account.

        rx_slots_total = event_slots_total + L - 1

    Returns
    -------
    int
        Length of the final received frame.
    """
    return compute_event_slots_total(tau, B, R, n_periods) + L - 1


# =============================================================================
# DUPLICATE-RECEPTION UPPER BOUND (LEMMA 5)
# =============================================================================

def epsilon_upper_bound(M_time: int, N: int, S: int) -> float:
    """
    Compute the upper bound from Lemma 5:

        epsilon <= 1 - M_time! / ((M_time - 2NS)! * M_time^(2NS))

    If M_time < 2NS, duplicates are guaranteed by the pigeonhole principle,
    so epsilon = 1.

    Parameters
    ----------
    M_time : int
        Number of time bins available in one cycle.
    N : int
        Number of harmonics.
    S : int
        Number of sensors.

    Returns
    -------
    float
        Upper bound on the average probability of receiving duplicate values.
    """
    k = 2 * N * S

    if M_time < k:
        return 1.0

    # log( M_time! / ((M_time-k)! * M_time^k) )
    log_ratio = (
        math.lgamma(M_time + 1)
        - math.lgamma(M_time - k + 1)
        - k * math.log(M_time)
    )

    no_duplicate_prob = math.exp(log_ratio)
    epsilon = 1.0 - no_duplicate_prob

    return float(max(0.0, min(1.0, epsilon)))


# =============================================================================
# PHYSICAL HELPER FORMULAS
# =============================================================================

def compute_N0(P: float, B: float, SNR: float) -> float:
    """
    Noise parameter from:

        SNR = P / (B * N0)

    Therefore:
        N0 = P / (B * SNR)

    IMPORTANT
    ---------
    This uses the TOTAL system bandwidth B.
    """
    return P / (B * SNR)


def compute_total_energy(P: float, tau: float) -> float:
    """
    Total available energy per cycle:

        E_tot = P * tau
    """
    return P * tau


def compute_symbol_energy(P: float, tau: float, L: int) -> float:
    """
    Energy per transmitted symbol/resource:

        E_s = (P * tau) / L
    """
    return (P * tau) / L


def compute_signal_level(P: float, tau: float, L: int) -> float:
    """
    Expected matched-filter output amplitude scale:

        sqrt(E_s)
    """
    return np.sqrt(compute_symbol_energy(P, tau, L))


def compute_default_detection_threshold(
    P: float,
    tau: float,
    L: int,
    threshold_factor: float = 0.5
) -> float:
    """
    Default detection threshold:

        threshold = threshold_factor * sqrt(E_s)

    with:
        E_s = (P * tau) / L
    """
    return threshold_factor * compute_signal_level(P, tau, L)


# =============================================================================
# OPTIONAL EXPORT LIST
# =============================================================================

__all__ = [
    # basic theory
    "compute_N",
    "compute_snr_linear",
    "compute_capacity",

    # RbCP theory
    "compute_q",
    "rbcp_mse_upper_bound",
    "rbcp_mse_lower_bound",
    "rbcp_mse_star",

    # M <-> M_RbCP
    "compute_M_rbcp_from_M",
    "compute_M_from_M_rbcp",

    # bandwidth sharing
    "compute_bandwidth_allocation",
    "compute_sensor_bandwidths",

    # RbCP feasible bins
    "compute_M_rbcp_single_sensor",
    "compute_M_rbcp_per_sensor",
    "compute_M_rbcp",

    # Benchmark bits/sample
    "compute_benchmark_bits_per_sample_single_sensor",
    "compute_benchmark_bits_per_sample_per_sensor",

    # SFC time / slot relations
    "compute_M_time",
    "compute_slot_duration",
    "compute_slots_per_period",
    "compute_event_slots_total",
    "compute_rx_slots_total",

    # Duplicate probability
    "epsilon_upper_bound",

    # Physical helpers
    "compute_N0",
    "compute_total_energy",
    "compute_symbol_energy",
    "compute_signal_level",
    "compute_default_detection_threshold",
]
