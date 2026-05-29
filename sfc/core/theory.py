"""
sfc/core/theory.py

Central repository for closed-form theoretical relations used across the project.

This module centralizes formulas from the manuscript, including:
- number of harmonics N
- Shannon capacity
- RbCP Q-factor
- MSE bounds / approximations for RbCP
- common / per-sensor feasible M_RbCP under bandwidth sharing
- benchmark feasible number of bins M under capacity constraints
- SFC time-slot relations
- duplicate-reception upper bound (Lemma 5)
- physical helper formulas (SNR, N0, energy, thresholds)

Design rule
-----------
Pipelines, channel modules, debug scripts, and builders should import the
theoretical relations from this file instead of reimplementing formulas locally.

Quantization policy
-------------------
Default behavior:
- M and M_RbCP are free integers
- no power-of-two restriction
- floor rounding

Power-of-two restriction can still be requested explicitly via:
    force_power_of_two=True

Important note
--------------
This file does NOT require any manuscript-specific switch such as:
    enforce_eq17_rate_matching

Eq. (17) remains available as a theoretical relationship between M and M_RbCP,
but it is not enforced here as a mandatory global policy.
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
    return int(np.floor((W * tau) / 2.0))


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
    return B * np.log2(1.0 + SNR)


# =============================================================================
# QUANTIZATION ROUNDING HELPERS
# =============================================================================

def _apply_integer_rounding(value: float, rounding_mode: str = "floor") -> int:
    """
    Convert a positive real value to an integer according to the selected
    rounding mode.

    Parameters
    ----------
    value : float
        Positive real value to be converted.

    rounding_mode : str, optional
        One of:
            - "floor"  (default)
            - "ceil"
            - "round"

    Returns
    -------
    int
        Integerized value, with minimum 1.
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
    rounding_mode: str = "floor"
) -> int:
    """
    Finalize a continuous-valued number of quantization bins M.

    Default behavior
    ----------------
    - free integer M
    - no power-of-two constraint
    - round according to `rounding_mode`

    If force_power_of_two=True
    --------------------------
    The nearest allowed value is constrained to:
        2^k

    with k obtained by applying the same `rounding_mode` to log2(M_continuous).

    Parameters
    ----------
    M_continuous : float
        Continuous-valued M obtained from a theoretical inequality.

    force_power_of_two : bool, optional
        If True, constrain the final M to powers of 2.

    rounding_mode : str, optional
        One of:
            - "floor"  (default)
            - "ceil"
            - "round"

    Returns
    -------
    int
        Final integer M.
    """

    M_continuous = max(float(M_continuous), 1.0)

    if not force_power_of_two:
        return _apply_integer_rounding(M_continuous, rounding_mode)

    k_continuous = np.log2(M_continuous)
    k_int = _apply_integer_rounding(k_continuous, rounding_mode)

    return int(2 ** k_int)


# =============================================================================
# RbCP THEORY (LEMMA 4 / PROPOSITION 2)
# =============================================================================

def compute_q(M_rbcp: float) -> float:
    """
    Compute the Q term used in Lemma 3 / Lemma 4 / Proposition 2:

        Q = (M_RbCP / (2*pi)) * sin(pi / M_RbCP)

    Parameters
    ----------
    M_rbcp : float
        Number of quantization bins for the RbCP representation.

    Returns
    -------
    float
        Q factor.
    """
    M_rbcp = max(float(M_rbcp), 1.0)
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
        Corresponding continuous-valued M_RbCP.
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
        Corresponding continuous-valued Benchmark M.
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

def compute_M_rbcp_single_sensor(
    W: float,
    tau: float,
    B_sensor: float,
    SNR: float,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor"
) -> int:
    """
    Compute the feasible number of RbCP bins for one sensor.

    Communication constraint for one sensor:
        2N log2(M_RbCP) <= tau * B_sensor * log2(1 + SNR)

    Therefore:
        M_RbCP <= (1 + SNR)^(tau * B_sensor / (2N))

    Default behavior
    ----------------
    - M_RbCP is a free integer
    - no power-of-two restriction
    - rounding_mode="floor"

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

    force_power_of_two : bool, optional
        If True, restrict the final M_RbCP to powers of 2.

    rounding_mode : str, optional
        One of:
            - "floor"  (default)
            - "ceil"
            - "round"

    Returns
    -------
    int
        Feasible M_RbCP for one sensor.
    """

    N = compute_N(W, tau)

    exponent = (tau * B_sensor) / (2.0 * N)
    M_continuous = (1.0 + SNR) ** exponent

    return _finalize_M(
        M_continuous=M_continuous,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode
    )


def compute_M_rbcp_per_sensor(
    S: int,
    W: float,
    tau: float,
    B: float,
    SNR: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor"
) -> np.ndarray:
    """
    Per-sensor feasible M_RbCP values, accounting for bandwidth sharing.

    Default behavior
    ----------------
    - free integer M_RbCP
    - no power-of-two restriction
    - rounding_mode="floor"

    Returns
    -------
    np.ndarray
        Shape (S,)
    """

    _, B_per_sensor = compute_sensor_bandwidths(B, S, bandwidth_allocation)

    return np.array([
        compute_M_rbcp_single_sensor(
            W=W,
            tau=tau,
            B_sensor=B_s,
            SNR=SNR,
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode
        )
        for B_s in B_per_sensor
    ], dtype=int)


def compute_M_rbcp(
    S: int,
    W: float,
    tau: float,
    B: float,
    SNR: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor"
) -> int:
    """
    Common feasible M_RbCP across S sensors.

    Default behavior
    ----------------
    - free integer M_RbCP
    - no power-of-two restriction
    - rounding_mode="floor"

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
    SNR: float,
    sampling_rate: float,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor"
) -> int:
    """
    Compute the feasible number of Benchmark quantization bins M for one sensor.

    Communication constraint for one sensor:
        sampling_rate * log2(M) <= B_sensor * log2(1 + SNR)

    Therefore:
        M <= (1 + SNR)^(B_sensor / sampling_rate)

    Default behavior
    ----------------
    - free integer M
    - no power-of-two restriction
    - rounding_mode="floor"

    Parameters
    ----------
    tau : float
        Frame duration. Present for interface symmetry.

    B_sensor : float
        Effective bandwidth assigned to one sensor.

    SNR : float
        Signal-to-noise ratio in linear scale.

    sampling_rate : float
        Sampling rate of the Benchmark branch.

    force_power_of_two : bool, optional
        If True, force M to be a power of 2.

    rounding_mode : str, optional
        One of:
            - "floor"  (default)
            - "ceil"
            - "round"

    Returns
    -------
    int
        Feasible number of bins M.
    """

    _ = tau  # kept for interface consistency

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    exponent = B_sensor / sampling_rate
    M_continuous = (1.0 + SNR) ** exponent

    return _finalize_M(
        M_continuous=M_continuous,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode
    )


def compute_benchmark_M_per_sensor(
    S: int,
    tau: float,
    B: float,
    SNR: float,
    sampling_rate: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor"
) -> np.ndarray:
    """
    Per-sensor feasible Benchmark M values.

    Default behavior
    ----------------
    - free integer M
    - no power-of-two restriction
    - rounding_mode="floor"

    Returns
    -------
    np.ndarray
        Shape (S,)
    """

    _, B_per_sensor = compute_sensor_bandwidths(B, S, bandwidth_allocation)

    return np.array([
        compute_benchmark_M_single_sensor(
            tau=tau,
            B_sensor=B_s,
            SNR=SNR,
            sampling_rate=sampling_rate,
            force_power_of_two=force_power_of_two,
            rounding_mode=rounding_mode
        )
        for B_s in B_per_sensor
    ], dtype=int)


# =============================================================================
# BENCHMARK BITS-PER-SAMPLE (BACKWARD COMPATIBILITY)
# =============================================================================

def compute_benchmark_bits_per_sample_single_sensor(
    tau: float,
    B_sensor: float,
    SNR: float,
    sampling_rate: float,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor"
) -> int:
    """
    Backward-compatible helper returning an effective bits-per-sample value
    derived from the feasible Benchmark M.

    Default behavior
    ----------------
    - M is free integer (not necessarily a power of 2)
    - bits are computed as floor(log2(M))

    Returns
    -------
    int
        Effective bits per sample.
    """

    M = compute_benchmark_M_single_sensor(
        tau=tau,
        B_sensor=B_sensor,
        SNR=SNR,
        sampling_rate=sampling_rate,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode
    )

    bits = int(np.floor(np.log2(max(M, 1))))
    return max(bits, 1)


def compute_benchmark_bits_per_sample_per_sensor(
    S: int,
    tau: float,
    B: float,
    SNR: float,
    sampling_rate: float,
    bandwidth_allocation: Optional[Sequence[float]] = None,
    force_power_of_two: bool = False,
    rounding_mode: str = "floor"
) -> np.ndarray:
    """
    Backward-compatible helper returning effective bits-per-sample per sensor
    derived from feasible Benchmark M values.
    """

    M_vec = compute_benchmark_M_per_sensor(
        S=S,
        tau=tau,
        B=B,
        SNR=SNR,
        sampling_rate=sampling_rate,
        bandwidth_allocation=bandwidth_allocation,
        force_power_of_two=force_power_of_two,
        rounding_mode=rounding_mode
    )

    bits_vec = np.array([
        max(int(np.floor(np.log2(max(M, 1)))), 1)
        for M in M_vec
    ], dtype=int)

    return bits_vec


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
    """
    return R / B


def compute_slots_per_period(tau: float, B: float, R: int) -> int:
    """
    Number of possible event-start slots per period:

        floor(tau / (R/B)) = floor(tau * B / R)
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

    # quantization helpers
    "_apply_integer_rounding",
    "_finalize_M",

    # RbCP theory
    "compute_q",
    "rbcp_mse_upper_bound",
    "rbcp_mse_lower_bound",
    "rbcp_mse_star",

    # Eq. (17)
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

    # Duplicate probability
    "epsilon_upper_bound",

    # Physical helpers
    "compute_N0",
    "compute_total_energy",
    "compute_symbol_energy",
    "compute_signal_level",
    "compute_default_detection_threshold",
]
