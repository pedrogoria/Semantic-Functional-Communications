"""
sfc/core/modulation/base.py

Base abstractionsBase abstractions for modulation-layer components.
-----------------
Whenever possible, modulation implementations should support the tensor format:

    (time, periods, sensors)

for continuous-time signals.

Additional symbol-domain outputs may use their own natural shapes, for example:

- sampled messages:
    (symbols, periods, sensors)

- pulse positions:
    (symbols, periods, sensors)

- waveform per sensor:
    (time, periods, sensors)

This base module intentionally stays lightweight so that concrete modulation
schemes (e.g. PPM) can specialize behavior without unnecessary constraints.
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
class NormalizationState:
    """
    Stores the parameters needed to invert a modulation-specific normalization.

    This is useful for modulation schemes that internally require a bounded
    message domain, for example mapping a waveform into the interval (0, 1).

    Attributes
    ----------
    enabled : bool
        Whether normalization was actually applied.

    x_min : float or None
        Minimum value used in normalization.

    x_max : float or None
        Maximum value used in normalization.

    eps_margin : float
        Margin applied when mapping into an open interval, for example:
            [0, 1] -> [eps, 1-eps]

    metadata : dict
        Free-form auxiliary metadata for scheme-specific use.
    """

    enabled: bool = False
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    eps_margin: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModulationResult:
    """
    Generic container returned by a modulation stage.

    Attributes
    ----------
    tx_waveform : np.ndarray or None
        Continuous-time transmitted waveform.
        Recommended shape:
            (time, periods, sensors)

    sampled_message : np.ndarray or None
        Message samples used internally by the modulation.
        Recommended shape:
            (symbols, periods, sensors)

    symbol_times : np.ndarray or None
        Symbol start times or symbol reference times.
        Recommended shape:
            (symbols,)

    normalization_state : NormalizationState
        Information needed to invert internal normalization, if any.

    aux : dict
        Free-form scheme-specific outputs, e.g.:
        - pulse_positions
        - pulse_shape
        - samples_per_symbol
        - carrier / timing metadata
    """

    tx_waveform: Optional[np.ndarray] = None
    sampled_message: Optional[np.ndarray] = None
    symbol_times: Optional[np.ndarray] = None
    normalization_state: NormalizationState = field(default_factory=NormalizationState)
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemodulationResult:
    """
    Generic container returned by a demodulation stage.

    Attributes
    ----------
    recovered_samples : np.ndarray or None
        Recovered message samples in the symbol domain.
        Recommended shape:
            (symbols, periods, sensors)

    recovered_continuous : np.ndarray or None
        Recovered continuous-time signal, when the modulation defines a
        reconstruction stage.
        Recommended shape:
            (time, periods, sensors)

    normalization_state : NormalizationState or None
        Normalization state used to recover the original signal domain.

    aux : dict
        Free-form scheme-specific outputs, e.g.:
        - matched_filter_output
        - detected_peak_times
        - confidence / scores
    """

    recovered_samples: Optional[np.ndarray] = None
    recovered_continuous: Optional[np.ndarray] = None
    normalization_state: Optional[NormalizationState] = None
    aux: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE ABSTRACT CLASS
# =============================================================================

class ModulationCoreBase(ABC):
    """
    Abstract base class for modulation-layer implementations.

    Intended usage
    --------------
    A concrete modulation class should subclass this base and implement:

    - modulate(...)
    - demodulate(...)

    Optionally, a concrete class may also override:
    - normalize_message(...)
    - denormalize_message(...)
    - reconstruct_continuous(...)

    Design principles
    -----------------
    1. Sensor-aware:
       the implementation should work with one or multiple sensors.

    2. Period-aware:
       the implementation should handle one or multiple periods whenever the
       experiment requires it.

    3. MAC-agnostic:
       the implementation should not decide how sensors share the medium.
       It may generate one waveform per sensor, but MAC-layer combination
       belongs elsewhere.

    4. Shape-preserving when practical:
       prefer explicit tensor conventions over ad hoc reshaping hidden inside
       the implementation.
    """

    def __init__(self, **kwargs):
        """
        Store free-form configuration in a generic dictionary.

        Concrete subclasses are encouraged to expose explicit constructor
        arguments, but keeping kwargs here is useful for lightweight extension
        and metadata inspection.
        """

        self.config: Dict[str, Any] = dict(kwargs)

    # -------------------------------------------------------------------------
    # Optional normalization interface
    # -------------------------------------------------------------------------

    def normalize_message(
        self,
        x: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, NormalizationState]:
        """
        Optional message normalization step.

        Default behavior
        ----------------
        No normalization is applied.

        Parameters
        ----------
        x : np.ndarray
            Input message tensor.

        Returns
        -------
        tuple
            (x_normalized, normalization_state)
        """

        _ = kwargs
        return x, NormalizationState(enabled=False)

    def denormalize_message(
        self,
        x: np.ndarray,
        normalization_state: Optional[NormalizationState],
        **kwargs
    ) -> np.ndarray:
        """
        Optional inverse normalization step.

        Default behavior
        ----------------
        Returns the input unchanged.

        Parameters
        ----------
        x : np.ndarray
            Input message tensor.

        normalization_state : NormalizationState or None
            State produced during normalization.

        Returns
        -------
        np.ndarray
            Denormalized signal.
        """

        _ = kwargs
        _ = normalization_state
        return x

    # -------------------------------------------------------------------------
    # Mandatory modulation / demodulation interface
    # -------------------------------------------------------------------------

    @abstractmethod
    def modulate(
        self,
        x: np.ndarray,
        t: np.ndarray,
        **kwargs
    ) -> ModulationResult:
        """
        Modulate a continuous-time input signal.

        Parameters
        ----------
        x : np.ndarray
            Input continuous-time signal.
            Recommended shape:
                (time, periods, sensors)
            but single-signal 1D / 2D convenience handling may be implemented
            by subclasses.

        t : np.ndarray
            Time grid associated with the first axis of x.

        Returns
        -------
        ModulationResult
            Structured modulation outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def demodulate(
        self,
        y: np.ndarray,
        t: np.ndarray,
        modulation_result: Optional[ModulationResult] = None,
        **kwargs
    ) -> DemodulationResult:
        """
        Demodulate a received signal.

        Parameters
        ----------
        y : np.ndarray
            Received continuous-time signal.
            Recommended shape:
                (time, periods, sensors)

        t : np.ndarray
            Time grid associated with the first axis of y.

        modulation_result : ModulationResult or None, optional
            Optional transmitter-side metadata that may help demodulation,
            for example normalization state or pulse-shaping information.

        Returns
        -------
        DemodulationResult
            Structured demodulation outputs.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Optional reconstruction helper
    # -------------------------------------------------------------------------

    def reconstruct_continuous(
        self,
        recovered_samples: np.ndarray,
        t: np.ndarray,
        modulation_result: Optional[ModulationResult] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Optional continuous-time reconstruction helper.

        Default behavior
        ----------------
        Raises NotImplementedError.

        Concrete subclasses may override this when the modulation defines a
        natural continuous-time reconstruction stage (e.g. sinc-based
        reconstruction or symbol-to-waveform interpolation).

        Parameters
        ----------
        recovered_samples : np.ndarray
            Recovered symbol-domain samples.

        t : np.ndarray
            Target continuous-time grid.

        modulation_result : ModulationResult or None
            Optional transmitter-side metadata.

        Returns
        -------
        np.ndarray
            Reconstructed continuous-time signal.
        """

        _ = recovered_samples
        _ = t
        _ = modulation_result
        _ = kwargs

        raise NotImplementedError(
            "This modulation class does not implement "
            "reconstruct_continuous(...)."
        )


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
        OR
        (time, sensors)
        -> ambiguous by itself

    To avoid ambiguous interpretation, 2D inputs are treated as:
        (time, periods)
        with one sensor.

    If you need a different interpretation, reshape explicitly before calling.

    - 3D:
        returned unchanged

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Tensor with shape (time, periods, sensors).
    """

    x = np.asarray(x)

    if x.ndim == 1:
        return x[:, None, None]

    if x.ndim == 2:
        # Interpret as (time, periods), single sensor
        return x[:, :, None]

    if x.ndim == 3:
        return x

    raise ValueError(
        f"Expected x with ndim in {{1, 2, 3}}, got shape={x.shape}"
    )


def ensure_1d_time_vector(t: np.ndarray) -> np.ndarray:
    """
    Ensure that the time grid is a 1D vector.

    Parameters
    ----------
    t : np.ndarray

    Returns
    -------
    np.ndarray
        Flattened 1D time vector.
    """

    t = np.asarray(t).reshape(-1)

    if t.ndim != 1:
        raise ValueError("Time vector t must be one-dimensional.")

    return t


def validate_time_axis_length(x: np.ndarray, t: np.ndarray):
    """
    Validate that the first axis of x matches len(t).

    Parameters
    ----------
    x : np.ndarray
        Signal tensor with time on axis 0.

    t : np.ndarray
        Time vector.
    """

    if x.shape[0] != len(t):
        raise ValueError(
            f"Signal/time mismatch: x.shape[0]={x.shape[0]} "
            f"but len(t)={len(t)}"
        )


# =============================================================================
# OPTIONAL EXPORTS
# =============================================================================

__all__ = [
    "NormalizationState",
    "ModulationResult",
    "DemodulationResult",
    "ModulationCoreBase",
    "ensure_3d_signal_tensor",
    "ensure_1d_time_vector",
    "validate_time_axis_length",
]
