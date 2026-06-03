"""
sfc/core/theory.py

Central repository for closed-form theoretical relations used across the project.

This module centralizes formulas from the manuscript and from the simulation
framework, including:

- number of harmonics N
- Shannon capacity
- bandwidth allocation
- per-sensor SNR from P, N0 and B_s
- feasible RbCP M under per-sensor bandwidth/power/noise constraints
- feasible Benchmark M under per-sensor bandwidth/power/noise constraints
- SFC time-slot relations
- duplicate-reception upper bound
- physical helper formulas

Design rule
-----------
Pipelines, channel modules, debug scripts, and builders should import
theoretical relations from this file instead of reimplementing formulas locally.

Main physical convention
------------------------
The preferred physical model is parameterized by:

    P  = average available transmit power per sensor
    N0 = noise spectral-density / noise parameter

For methods that use a per-sensor bandwidth slice:

    B_s = alpha_s * B

the sensor SNR is:

    SNR_s = P / (B_s * N0)

Therefore, functions of the form compute_M_*(...) receive P and N0 by default
and compute SNR_s internally.

Quantization policy
-------------------
Default behavior:
- M and M_RbCP are free integers
- no power-of-two restriction
- floor rounding

Power-of-two restriction can still be requested explicitly via:

    force_power_of_two=True
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
    Number of harmonics:

        N = floor(W * tau / 2)
    """
    return int(np.floor((W * tau) / 2.0))


def compute_snr_linear(SNR_dB: float) -> float:
    """
    Convert SNR from dB to linear scale.
    """
    return 10.0 ** (SNR_dB / 10.0)


def compute_snr_db(SNR: float) -> float:
    """
    Convert SNR from linear scale to dB.
    """
    return 10.0 * np.log10(max(float(SNR), np.finfo(float).tiny))


def compute_capacity(B: float, SNR: float) -> float:
    """
    Shannon capacity:

        C = B * log2(1 + SNR)
    """
    return B * np.log2(1.0 + SNR)


def compute_sensor_snr(P: float, B_sensor: float, N0: float) -> float:
    """
    Compute per-sensor SNR:

        SNR_s = P / (B_s * N0)

    Parameters
    ----------
    P : float
        Average available transmit power per sensor.

    B_sensor : float
        Bandwidth assigned to the sensor.

    N0 : float
        Noise spectral-density / noise parameter.
    """
    if P <= 0:
        raise ValueError("P must be positive.")
    if B_sensor <= 0:
        raise ValueError("B_sensor must be positive.")
    if N0 <= 0:
        raise ValueError("N0 must be positive.")

    return P / (B_sensor * N0)


# =============================================================================
# QUANTIZATION ROUNDING HELPERS
# =============================================================================

def _apply_integer_rounding(value: float, rounding_mode: str = "floor") -> int:
    """
    Convert a positive real value to an integer according to the selected
    rounding mode.
    """

    value = max(float(value), 1.0)

    if rounding_mode == "floor":
        out = int(np.floor(value))
    elif rounding_mode == "ceil":
        out = int(np.ceil(value))
    elif rounding_mode == "round":
        out = int(np.round(value))
    else:
        raise ValueError(
            f"Invalid rounding_mode='{rounding_mode}'. "
            f"Use 'floor', 'ceil', or 'round'."
        )

    return max(out, 1)


def _finalize_M(
    M_continuous: float,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> int:
    """
    Finalize a continuous-valued number of quantization bins M.

    Default behavior:
    - free integer M
    - no power-of-two constraint
    - round according to rounding_mode

    If force_power_of_two=True:
    - constrain the final M to powers of 2
    """

    M_continuous = max(float(M_continuous), 1.0)

    if not force_power_of_two:
        return _apply_integer_rounding(M_continuous, rounding_mode)

    k_continuous = np.log2(M_continuous)
    k_int = _apply_integer_rounding(k_continuous, rounding_mode)

    return int(2 ** k_int)


# =============================================================================
# RbCP THEORY
# =============================================================================

def compute_q(M_rbcp: float) -> float:
    """
    Compute the Q term used in the RbCP MSE formulas:

        Q = (M_RbCP / (2*pi)) * sin(pi / M_RbCP)
    """

    M_rbcp = max(float(M_rbcp), 1.0)
    return (M_rbcp / (2.0 * np.pi)) * np.sin(np.pi / M_rbcp)


def rbcp_mse_upper_bound(N: int, Q: float) -> float:
    """
    RbCP upper bound:

        MSE_upper = 4 * N * (1/2 - Q) * (3/2 - Q)
    """
    return 4.0 * N * (0.5 - Q) * (1.5 - Q)


def rbcp_mse_lower_bound(N: int, Q: float) -> float:
    """
    RbCP lower bound:

        MSE_lower = N * (1 - 4Q^2)
    """
    return N * (1.0 - 4.0 * Q * Q)


def rbcp_mse_star(N: int, Q: float) -> float:
    """
    Proposition-style RbCP approximation:

        MSE*_RbCP = 2N (1 - 2Q)
    """
    return 2.0 * N * (1.0 - 2.0 * Q)


# =============================================================================
# RELATION BETWEEN M AND M_RbCP
# =============================================================================

def compute_M_rbcp_from_M(M: float, W: float, tau: float) -> float:
    """
    Equation-style mapping:

        M_RbCP = M * tau * W / (2N)

    with:

        N = floor(W * tau / 2)
    """
    N = compute_N(W, tau)
    return M * tau * W / (2.0 * N)


def compute_M_from_M_rbcp(M_rbcp: float, W: float, tau: float) -> float:
    """
    Inverse mapping:

        M = M_RbCP * (2N) / (tau * W)
    """
    N = compute_N(W, tau)
    return M_rbcp * (2.0 * N) / (tau * W)


# =============================================================================
# BANDWIDTH SHARING AMONG SENSORS
# =============================================================================

def compute_bandwidth_allocation(
    S: int,
    bandwidth_allocation: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Compute / validate the bandwidth-allocation vector.

    Rules
    -----
    - if bandwidth_allocation is None:
        equal split among S sensors
    - otherwise:
        * length must be S
        * entries must be nonnegative
        * sum must be 1
    """

    if S < 1:
        raise ValueError("S must be >= 1.")

    if bandwidth_allocation is None:
        return np.ones(S, dtype=float) / S

    alloc = np.asarray(bandwidth_allocation, dtype=float).reshape(-1)

    if len(alloc) != S:
        raise ValueError(
            f"bandwidth_allocation must have length S={S}, got {len(alloc)}"
        )

    if np.any(alloc < 0):
        raise ValueError("bandwidth_allocation must contain nonnegative values.")

    if not np.isclose(np.sum(alloc), 1.0):
        raise ValueError(
            f"bandwidth_allocation must sum to 1, got sum={np.sum(alloc)}"
        )

    return alloc


def compute_sensor_bandwidths(
    B: float,
    S: int,
    bandwidth_allocation: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute effective per-sensor bandwidths from the total bandwidth B.

    Returns
    -------
    tuple
        (allocation, B_per_sensor)
    """

    if B <= 0:
        raise ValueError("B must be positive.")

    allocation = compute_bandwidth_allocation(S, bandwidth_allocation)
    B_per_sensor = B * allocation

    return allocation, B_per_sensor


# =============================================================================
# RbCP FEASIBLE NUMBER OF BINS UNDER BANDWIDTH SHARING
# =============================================================================

def compute_M_rbcp_single_sensor(
    W: float,
    tau: float,
    B_sensor: float,
    P: float,
    N0: float,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> int:
    """
    Compute the feasible number of RbCP bins for one sensor.

    Constraint:

        2N log2(M_RbCP) <= tau * B_s * log2(1 + SNR_s)

    with:

        SNR_s = P / (B_s * N0)

    Therefore:

        M_RbCP <= (1 + SNR_s)^(tau * B_s / (2N))
    """

    N = compute_N(W, tau)
    if N <= 0:
        raise ValueError("Computed N must be positive.")

    SNR_sensor = compute_sensor_snr(P=P, B_sensor=B_sensor, N0=N0)

    exponent = (tau * B_sensor) / (2.0 * N)
    M_continuous = (1.0 + SNR_sensor) ** exponent

    return _finalize_M(
        M_continuous=M_continuous,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode,
    )


def compute_M_rbcp_per_sensor(
    S: int,
    W: float,
    tau: float,
    B: float,
    P: float,
    N0: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> np.ndarray:
    """
    Per-sensor feasible M_RbCP values, accounting for bandwidth sharing.
    """

    _, B_per_sensor = compute_sensor_bandwidths(
        B=B,
        S=S,
        bandwidth_allocation=bandwidth_allocation,
    )

    return np.array([
        compute_M_rbcp_single_sensor(
            W=W,
            tau=tau,
            B_sensor=B_s,
            P=P,
            N0=N0,
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
    P: float,
    N0: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> int:
    """
    Common feasible M_RbCP across S sensors:

        M_RbCP = min_s M_RbCP,s
    """

    M_vec = compute_M_rbcp_per_sensor(
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

    return int(np.min(M_vec))


# =============================================================================
# BENCHMARK FEASIBLE NUMBER OF BINS
# =============================================================================

def compute_benchmark_M_single_sensor(
    tau: float,
    B_sensor: float,
    P: float,
    N0: float,
    sampling_rate: float,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> int:
    """
    Compute the feasible number of Benchmark quantization bins M for one sensor.

    Constraint:

        sampling_rate * log2(M) <= B_s * log2(1 + SNR_s)

    with:

        SNR_s = P / (B_s * N0)

    Therefore:

        M <= (1 + SNR_s)^(B_s / sampling_rate)

    tau is kept for interface symmetry.
    """

    _ = tau

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    SNR_sensor = compute_sensor_snr(P=P, B_sensor=B_sensor, N0=N0)

    exponent = B_sensor / sampling_rate
    M_continuous = (1.0 + SNR_sensor) ** exponent

    return _finalize_M(
        M_continuous=M_continuous,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode,
    )


def compute_benchmark_M_per_sensor(
    S: int,
    tau: float,
    B: float,
    P: float,
    N0: float,
    sampling_rate: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> np.ndarray:
    """
    Per-sensor feasible Benchmark M values.
    """

    _, B_per_sensor = compute_sensor_bandwidths(
        B=B,
        S=S,
        bandwidth_allocation=bandwidth_allocation,
    )

    return np.array([
        compute_benchmark_M_single_sensor(
            tau=tau,
            B_sensor=B_s,
            P=P,
            N0=N0,
            sampling_rate=sampling_rate,
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode,
        )
        for B_s in B_per_sensor
    ], dtype=int)


def compute_benchmark_bits_per_sample_single_sensor(
    tau: float,
    B_sensor: float,
    P: float,
    N0: float,
    sampling_rate: float,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> int:
    """
    Return effective bits per sample derived from feasible Benchmark M.

    Default:
        bits = floor(log2(M))
    """

    M = compute_benchmark_M_single_sensor(
        tau=tau,
        B_sensor=B_sensor,
        P=P,
        N0=N0,
        sampling_rate=sampling_rate,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode,
    )

    bits = int(np.floor(np.log2(max(M, 1))))
    return max(bits, 1)


def compute_benchmark_bits_per_sample_per_sensor(
    S: int,
    tau: float,
    B: float,
    P: float,
    N0: float,
    sampling_rate: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor",
) -> np.ndarray:
    """
    Return effective bits per sample per sensor derived from feasible Benchmark M.
    """

    M_vec = compute_benchmark_M_per_sensor(
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

    return np.array([
        max(int(np.floor(np.log2(max(M, 1)))), 1)
        for M in M_vec
    ], dtype=int)


# =============================================================================
# SFC TIME / SLOT RELATIONS
# =============================================================================

def compute_M_time(tau: float, B: float, R: int) -> int:
    """
    Number of time bins / event-start bins for the SFC time model:

        M_time = floor(tau * B / R)

    Important:
    SFC uses the total B, not B_s.
    """

    return int(np.floor((tau * B) / R))


def compute_slot_duration(B: float, R: int) -> float:
    """
    Duration of one SFC symbol slot:

        T_slot = R / B
    """

    if B <= 0:
        raise ValueError("B must be positive.")
    if R <= 0:
        raise ValueError("R must be positive.")

    return R / B


def compute_slots_per_period(tau: float, B: float, R: int) -> int:
    """
    Number of possible event-start slots per period:

        floor(tau / (R/B)) = floor(tau * B / R)
    """

    return int(np.floor(tau * B / R))


def compute_event_slots_total(
    tau: float,
    B: float,
    R: int,
    n_periods: int,
) -> int:
    """
    Total number of possible event-start slots over n_periods.
    """

    return compute_slots_per_period(tau, B, R) * int(n_periods)


def compute_rx_slots_total(
    tau: float,
    B: float,
    R: int,
    n_periods: int,
    L: int,
) -> int:
    """
    Total number of received discrete-time slots once the map length L is
    considered:

        rx_slots_total = event_slots_total + L - 1
    """

    return compute_event_slots_total(tau, B, R, n_periods) + int(L) - 1


# =============================================================================
# DUPLICATE-RECEPTION UPPER BOUND
# =============================================================================

def epsilon_upper_bound(M_time: int, N: int, S: int) -> float:
    """
    Compute the duplicate-reception upper bound:

        epsilon <= 1 - M_time! / ((M_time - 2NS)! * M_time^(2NS))

    If M_time < 2NS, duplicates are guaranteed by the pigeonhole principle,
    so epsilon = 1.
    """

    k = 2 * N * S

    if M_time < k:
        return 1.0

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

    This helper uses whichever B is passed to it.
    """

    if P <= 0:
        raise ValueError("P must be positive.")
    if B <= 0:
        raise ValueError("B must be positive.")
    if SNR <= 0:
        raise ValueError("SNR must be positive.")

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
    threshold_factor: float = 0.5,
) -> float:
    """
    Default detection threshold:

        threshold = threshold_factor * sqrt(E_s)
    """

    return threshold_factor * compute_signal_level(P, tau, L)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # basic theory
    "compute_N",
    "compute_snr_linear",
    "compute_snr_db",
    "compute_capacity",
    "compute_sensor_snr",

    # quantization helpers
    "_apply_integer_rounding",
    "_finalize_M",

    # RbCP theory
    "compute_q",
    "rbcp_mse_upper_bound",
    "rbcp_mse_lower_bound",
    "rbcp_mse_star",

    # Eq. relation
    "compute_M_rbcp_from_M",
    "compute_M_from_M_rbcp",

    # bandwidth sharing
    "compute_bandwidth_allocation",
    "compute_sensor_bandwidths",

    # RbCP feasible bins
    "compute_M_rbcp_single_sensor",
    "compute_M_rbcp_per_sensor",
    "compute_M_rbcp",

    # Benchmark M / bits
    "compute_benchmark_M_single_sensor",
    "compute_benchmark_M_per_sensor",
    "compute_benchmark_bits_per_sample_single_sensor",
    "compute_benchmark_bits_per_sample_per_sensor",

    # SFC time / slot relations
    "compute_M_time",
    "compute_slot_duration",
    "compute_slots_per_period",
    "compute_event_slots_total",
    "compute_rx_slots_total",

    # duplicate probability
    "epsilon_upper_bound",

    # physical helpers
    "compute_N0",
    "compute_total_energy",
    "compute_symbol_energy",
    "compute_signal_level",
    "compute_default_detection_threshold",
]
