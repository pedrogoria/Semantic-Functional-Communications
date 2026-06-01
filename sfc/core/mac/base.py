"""
sfc/core/mac/base.py

Base abstractions for MAC-layer components.

Design goal
-----------
This module defines lightweight interfaces and reusable data containers for
medium-access-control (MAC) methods that should remain independent from the
source/modulation layer.

In this architecture:

- modulation is responsible for:
    * message normalization / denormalization when applicable
    * message sampling
    * pulse shaping
    * modulation / demodulation
    * optional continuous-time reconstruction

- MAC is responsible for:
    * how multiple sensors share the communication medium
    * bandwidth slicing
    * time slicing
    * subcarrier/resource allocation
    * multiplexing multiple sensor waveforms / signals
    * inverse demultiplexing

Therefore, concrete MAC implementations on top of this base should be:

- sensor-aware
- budget-aware
- modulation-agnostic (as much as practical)
- explicit about how total resources are split among sensors

Typical future implementations
------------------------------
- FDMA
- TDMA
- OFDM
- SFC

Tensor conventions
------------------
Whenever practical, sensor-wise continuous-time waveforms are represented as:

    (time, periods, sensors)

A MAC may produce:
- an aggregated waveform:
    (time, periods)
- one waveform per sensor after allocation:
    (time, periods, sensors)
- or a richer structure in aux metadata

This base module intentionally remains lightweight so that concrete MAC schemes
can specialize behavior without unnecessary constraints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class MACInput:
    """
    Standard container for MAC-layer input.

    Attributes
    ----------
    tx_waveform_per_sensor : np.ndarray or None
        Continuous-time waveform per sensor.
        Recommended shape:
            (time, periods, sensors)

        This is the most common input for classic MACs such as FDMA/TDMA/OFDM.

    symbols_per_sensor : np.ndarray or None
        Symbol-domain data per sensor when a MAC works more naturally in the
        symbol/resource domain instead of directly on continuous-time waveforms.

    event_matrix : np.ndarray or None
        Event-domain representation, useful for event-driven MACs such as SFC.

    t : np.ndarray or None
        Time grid associated with the waveform axis, when applicable.

    metadata : dict
        Free-form auxiliary information such as:
        - sampling frequencies
        - per-sensor rate targets
        - modulation-specific state
        - reference slot durations
    """

    tx_waveform_per_sensor: Optional[np.ndarray] = None
    symbols_per_sensor: Optional[np.ndarray] = None
    event_matrix: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MACMultiplexResult:
    """
    Standard container returned by the MAC multiplex/transmit-allocation stage.

    Attributes
    ----------
    multiplexed_waveform : np.ndarray or None
        Aggregated or composite waveform after MAC allocation.
        Recommended shape:
            (time, periods)
        or another scheme-specific shape documented in metadata.

    per_sensor_allocated_waveform : np.ndarray or None
        Optional waveform after MAC allocation but before aggregation.
        Recommended shape:
            (time, periods, sensors)

    allocation_metadata : dict
        Explicit description of how resources were assigned, e.g.:
        - bandwidth_per_sensor
        - slot_start_times
        - subcarrier indices
        - resource maps
        - orthogonality assumptions

    aux : dict
        Free-form scheme-specific outputs.
    """

    multiplexed_waveform: Optional[np.ndarray] = None
    per_sensor_allocated_waveform: Optional[np.ndarray] = None
    allocation_metadata: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MACDemultiplexResult:
    """
    Standard container returned by the MAC demultiplex / inverse-allocation stage.

    Attributes
    ----------
    recovered_waveform_per_sensor : np.ndarray or None
        Recovered waveform per sensor after inverse MAC operation.
        Recommended shape:
            (time, periods, sensors)

    recovered_symbols_per_sensor : np.ndarray or None
        Recovered symbol-domain structure per sensor, when relevant.

    recovered_event_matrix : np.ndarray or None
        Recovered event-domain structure, when relevant.

    aux : dict
        Free-form scheme-specific outputs such as:
        - detection metrics
        - per-resource estimates
        - per-sensor confidence
    """

    recovered_waveform_per_sensor: Optional[np.ndarray] = None
    recovered_symbols_per_sensor: Optional[np.ndarray] = None
    recovered_event_matrix: Optional[np.ndarray] = None
    aux: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE ABSTRACT CLASS
# =============================================================================

class MACCoreBase(ABC):
    """
    Abstract base class for MAC-layer implementations.

    Intended usage
    --------------
    A concrete MAC class should subclass this base and implement:

    - multiplex(...)
    - demultiplex(...)

    Core principles
    ---------------
    1. Sensor-aware:
       the implementation must make explicit how S sensors share the medium.

    2. Budget-aware:
       the implementation should expose how the physical communication budget
       is interpreted, for example:
       - bandwidth slices
       - time slots
       - subcarriers
       - map resources

    3. Modulation-agnostic when possible:
       the MAC should avoid depending on the internals of a specific modulation
       unless the MAC itself is inherently tied to a representation (e.g. SFC).

    4. Explicit resource accounting:
       the MAC should document in its metadata how total bandwidth B and any
       other shared resources are distributed among sensors.
    """

    def __init__(
        self,
        S: int,
        B_total: float,
        P_per_sensor: float,
        tau: float,
        bandwidth_allocation: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        S : int
            Number of sensors.

        B_total : float
            Total communication bandwidth.

        P_per_sensor : float
            Average available power per sensor.

        tau : float
            Period / frame duration.

        bandwidth_allocation : array-like or None
            Allocation fractions across sensors.
            If None, equal split is assumed whenever bandwidth slicing is needed.

        kwargs : dict
            Free-form auxiliary configuration.
        """

        if S < 1:
            raise ValueError("S must be >= 1.")
        if B_total <= 0:
            raise ValueError("B_total must be > 0.")
        if P_per_sensor <= 0:
            raise ValueError("P_per_sensor must be > 0.")
        if tau <= 0:
            raise ValueError("tau must be > 0.")

        self.S = int(S)
        self.B_total = float(B_total)
        self.P_per_sensor = float(P_per_sensor)
        self.tau = float(tau)

        self.bandwidth_allocation = _resolve_bandwidth_allocation(
            S=self.S,
            bandwidth_allocation=bandwidth_allocation
        )

        self.B_per_sensor = self.B_total * self.bandwidth_allocation

        self.config: Dict[str, Any] = dict(kwargs)

    # -------------------------------------------------------------------------
    # Main MAC interface
    # -------------------------------------------------------------------------

    @abstractmethod
    def multiplex(
        self,
        mac_input: MACInput,
        **kwargs
    ) -> MACMultiplexResult:
        """
        Apply MAC allocation / multiplexing.

        Parameters
        ----------
        mac_input : MACInput
            Sensor-wise modulation or event-domain input.

        Returns
        -------
        MACMultiplexResult
            Result of MAC allocation / multiplexing.
        """
        raise NotImplementedError

    @abstractmethod
    def demultiplex(
        self,
        received_signal: Any,
        multiplex_result: Optional[MACMultiplexResult] = None,
        **kwargs
    ) -> MACDemultiplexResult:
        """
        Invert the MAC allocation / multiplexing.

        Parameters
        ----------
        received_signal : Any
            Received object at the MAC output. Depending on the concrete MAC,
            this may be a waveform, resource grid, event matrix, etc.

        multiplex_result : MACMultiplexResult or None
            Optional transmit-side metadata used to guide demultiplexing.

        Returns
        -------
        MACDemultiplexResult
            Recovered sensor-wise structure.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def get_bandwidth_per_sensor(self) -> np.ndarray:
        """
        Return the absolute bandwidth slice per sensor.

        Returns
        -------
        np.ndarray
            Shape: (S,)
        """
        return np.array(self.B_per_sensor, dtype=float)

    def get_power_per_sensor(self) -> np.ndarray:
        """
        Return a vector with the average power budget per sensor.

        Returns
        -------
        np.ndarray
            Shape: (S,)
        """
        return np.full(self.S, self.P_per_sensor, dtype=float)

    def describe_budget(self) -> Dict[str, Any]:
        """
        Return a lightweight dictionary summarizing the MAC budget.

        Returns
        -------
        dict
            Contains:
            - S
            - B_total
            - P_per_sensor
            - tau
            - bandwidth_allocation
            - B_per_sensor
        """
        return {
            "S": self.S,
            "B_total": self.B_total,
            "P_per_sensor": self.P_per_sensor,
            "tau": self.tau,
            "bandwidth_allocation": np.array(self.bandwidth_allocation, dtype=float),
            "B_per_sensor": np.array(self.B_per_sensor, dtype=float),
        }


# =============================================================================
# SHAPE / INPUT HELPERS
# =============================================================================

def ensure_3d_waveform_tensor(x: np.ndarray) -> np.ndarray:
    """
    Convert a waveform array to canonical 3D tensor form:

        (time, periods, sensors)

    Accepted inputs
    ---------------
    - 1D:
        (time,)
        -> interpreted as one period, one sensor

    - 2D:
        (time, periods)
        -> interpreted as one sensor

    - 3D:
        returned unchanged

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray
        Tensor with shape (time, periods, sensors).
    """

    x = np.asarray(x)

    if x.ndim == 1:
        return x[:, None, None]

    if x.ndim == 2:
        return x[:, :, None]

    if x.ndim == 3:
        return x

    raise ValueError(
        f"Expected waveform tensor with ndim in {{1, 2, 3}}, got shape={x.shape}"
    )


def ensure_2d_aggregated_waveform(x: np.ndarray) -> np.ndarray:
    """
    Convert an aggregated waveform to canonical 2D form:

        (time, periods)

    Accepted inputs
    ---------------
    - 1D:
        (time,)
        -> interpreted as one period

    - 2D:
        returned unchanged

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray
        Tensor with shape (time, periods).
    """

    x = np.asarray(x)

    if x.ndim == 1:
        return x[:, None]

    if x.ndim == 2:
        return x

    raise ValueError(
        f"Expected aggregated waveform with ndim in {{1, 2}}, got shape={x.shape}"
    )


def validate_sensor_dimension(x: np.ndarray, S: int):
    """
    Validate that the last axis of a canonical waveform tensor matches S.

    Parameters
    ----------
    x : np.ndarray
        Expected canonical shape:
            (time, periods, sensors)

    S : int
        Number of sensors.
    """

    if x.ndim != 3:
        raise ValueError(
            f"Expected a 3D tensor (time, periods, sensors), got shape={x.shape}"
        )

    if x.shape[2] != S:
        raise ValueError(
            f"Sensor dimension mismatch: x.shape[2]={x.shape[2]} but S={S}"
        )


def validate_time_dimension(x: np.ndarray, t: np.ndarray):
    """
    Validate that waveform tensor first axis matches len(t).

    Parameters
    ----------
    x : np.ndarray
        Waveform tensor with time along axis 0.

    t : np.ndarray
        Time vector.
    """

    t = np.asarray(t).reshape(-1)

    if x.shape[0] != len(t):
        raise ValueError(
            f"Time mismatch: waveform has x.shape[0]={x.shape[0]} "
            f"but len(t)={len(t)}"
        )


# =============================================================================
# BANDWIDTH ALLOCATION HELPERS
# =============================================================================

def _resolve_bandwidth_allocation(
    S: int,
    bandwidth_allocation: Optional[np.ndarray]
) -> np.ndarray:
    """
    Resolve a bandwidth-allocation vector.

    Rules
    -----
    - If bandwidth_allocation is None:
        equal split among S sensors
    - Otherwise:
        * length must be S
        * entries must be >= 0
        * sum must be 1

    Parameters
    ----------
    S : int
        Number of sensors.

    bandwidth_allocation : array-like or None

    Returns
    -------
    np.ndarray
        Shape: (S,)
    """

    if bandwidth_allocation is None:
        return np.ones(S, dtype=float) / S

    alloc = np.asarray(bandwidth_allocation, dtype=float).reshape(-1)

    if len(alloc) != S:
        raise ValueError(
            f"bandwidth_allocation length mismatch: len={len(alloc)} but S={S}"
        )

    if np.any(alloc < 0):
        raise ValueError("bandwidth_allocation must be nonnegative")

    total = np.sum(alloc)
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"bandwidth_allocation must sum to 1. Current sum={total}"
        )

    return alloc


# =============================================================================
# OPTIONAL EXPORTS
# =============================================================================

__all__ = [
    "MACInput",
    "MACMultiplexResult",
    "MACDemultiplexResult",
    "MACCoreBase",
    "ensure_3d_waveform_tensor",
    "ensure_2d_aggregated_waveform",
    "validate_sensor_dimension",
    "validate_time_dimension",
]
