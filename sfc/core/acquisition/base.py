"""
sfc/core/acquisition/base.py

Base abstractions for acquisition-layer components.

Design goal
-----------
This module defines lightweight interfaces and reusable data containers for
source-acquisition methods that should remain independent from modulation-layer
and MAC-layer decisions.

In this architecture:

- acquisition is responsible for:
    * source sampling / source representation
    * compression / measurement generation
    * reconstruction back to the source-signal domain

- modulation is responsible for:
    * transforming the acquisition payload into a waveform/signaling object

- MAC is responsible for:
    * how multiple sensors share the communication medium

Therefore, concrete acquisition implementations should be:

- sensor-aware
- period-aware
- modulation-agnostic
- MAC-agnostic

Tensor convention
-----------------
Whenever possible, acquisition implementations should support the tensor format:

    (time, periods, sensors)

for continuous-time source signals.

Compressed or representation-domain outputs may use natural shapes such as:

- measurements:
    (measurements, periods, sensors)

- coefficient vectors:
    (coefficients, periods, sensors)

This base module intentionally stays lightweight so that concrete acquisition
schemes (e.g. CS) can specialize behavior without unnecessary constraints.
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
class AcquisitionResult:
    """
    Generic container returned by an acquisition stage.

    Attributes
    ----------
    measurements : np.ndarray or None
        Acquired measurement-domain representation.
        Recommended shape:
            (measurements, periods, sensors)

    representation : np.ndarray or None
        Optional source-domain representation coefficients, if explicitly
        produced during acquisition.
        Recommended shape:
            (coefficients, periods, sensors)

    t : np.ndarray or None
        Time grid associated with the original signal.

    aux : dict
        Free-form scheme-specific outputs, e.g.:
        - sensing matrix
        - basis matrix
        - sampling rate
        - representation metadata
    """

    measurements: Optional[np.ndarray] = None
    representation: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """
    Generic container returned by an acquisition-layer reconstruction stage.

    Attributes
    ----------
    reconstructed_signal : np.ndarray or None
        Reconstructed source signal.
        Recommended shape:
            (time, periods, sensors)

    recovered_representation : np.ndarray or None
        Recovered representation coefficients, if applicable.
        Recommended shape:
            (coefficients, periods, sensors)

    aux : dict
        Free-form scheme-specific reconstruction outputs, e.g.:
        - support sets
        - residual norms
        - OMP diagnostics
    """

    reconstructed_signal: Optional[np.ndarray] = None
    recovered_representation: Optional[np.ndarray] = None
    aux: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE ABSTRACT CLASS
# =============================================================================

class AcquisitionCoreBase(ABC):
    """
    Abstract base class for acquisition-layer implementations.

    Intended usage
    --------------
    A concrete acquisition class should subclass this base and implement:

    - acquire(...)
    - reconstruct(...)

    Design principles
    -----------------
    1. Sensor-aware:
       the implementation should work with one or multiple sensors.

    2. Period-aware:
       the implementation should handle one or multiple periods whenever the
       experiment requires it.

    3. Modulation-agnostic:
       the acquisition stage should not decide how the payload becomes a
       waveform.

    4. MAC-agnostic:
       the acquisition stage should not decide how sensors share the medium.
    """

    def __init__(self, **kwargs):
        """
        Store free-form configuration in a generic dictionary.
        """
        self.config: Dict[str, Any] = dict(kwargs)

    @abstractmethod
    def acquire(
        self,
        x: np.ndarray,
        t: np.ndarray,
        **kwargs
    ) -> AcquisitionResult:
        """
        Acquire / compress / represent a source signal.

        Parameters
        ----------
        x : np.ndarray
            Input continuous-time source signal.
            Recommended shape:
                (time, periods, sensors)

        t : np.ndarray
            Time grid associated with the first axis of x.

        Returns
        -------
        AcquisitionResult
            Structured acquisition outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct(
        self,
        acquisition_result: AcquisitionResult,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct the source signal from acquired measurements or
        representation-domain data.

        Parameters
        ----------
        acquisition_result : AcquisitionResult
            Result returned by acquire(...)

        Returns
        -------
        ReconstructionResult
            Structured reconstruction outputs.
        """
        raise NotImplementedError


# =============================================================================
# SHAPE HELPERS
# =============================================================================

def ensure_3d_signal_tensor(x: np.ndarray) -> np.ndarray:
    """
    Convert a signal array to the canonical 3D tensor:

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
    """

    x = np.asarray(x)

    if x.ndim == 1:
        return x[:, None, None]

    if x.ndim == 2:
        return x[:, :, None]

    if x.ndim == 3:
        return x

    raise ValueError(
        f"Expected x with ndim in {{1, 2, 3}}, got shape={x.shape}"
    )


def ensure_1d_time_vector(t: np.ndarray) -> np.ndarray:
    """
    Ensure that the time grid is a 1D vector.
    """

    t = np.asarray(t).reshape(-1)

    if t.ndim != 1:
        raise ValueError("Time vector t must be one-dimensional.")

    return t


def validate_time_axis_length(x: np.ndarray, t: np.ndarray):
    """
    Validate that the first axis of x matches len(t).
    """

    if x.shape[0] != len(t):
        raise ValueError(
            f"Signal/time mismatch: x.shape[0]={x.shape[0]} "
            f"but len(t)={len(t)}"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AcquisitionResult",
    "ReconstructionResult",
    "AcquisitionCoreBase",
    "ensure_3d_signal_tensor",
    "ensure_1d_time_vector",
    "validate_time_axis_length",
]