"""
samplers.py

PHY-facing sampling/representation API.

This module exposes stable, high-level classes that wrap the trusted
implementation in sfc/sampling.py. The purpose is to create a clean "core"
surface for a larger research codebase while preserving the original behavior.

Design rules (project constraints):
- Do not reimplement RbCP math: it already exists in sfc/sampling.py.
- Keep behavior identical to the trusted code for the RbCP path.
- Provide a stable class-based interface (PHY-style) to be used by tests, runs,
  and later system-level layers.
- Keep the implementation explicit and easy to debug in an IDE.

RbCP definition reference:
- The manuscript defines RbCP via the cosine-phase representation (Eq. (5))
  and the phase-shift mapping (Eq. (6)–(8)). The trusted code implements this
  mapping in CPSample.calc_ta_tb and the DFT extraction in CPSample.calc_an_bn_dft. [1](https://github.com/pedrogoria/Semantic-Functional-Communications)

Author: SFC Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import numpy as np

# Trusted backend (source of truth)
# This must point to your reliable file: sfc/sampling.py
from sfc.sampling import CPSample


@dataclass
class RbCPConfig:
    """
    Configuration container for RbCPSampler.

    This mirrors the CPSample constructor parameters, but keeps the core API stable.
    """
    T: float = 1.0
    harmonics: int = 3
    n_sub_symbol: int = 6
    resource: int = 7
    sensor_nodes: int = 1
    bandwidth: float = 100.0

    # Optional CPSample behavior flags
    detect_errors: bool = False
    threshold_harmonics: float = 0.001
    dft_signal_periods: int = 1


class RbCPSampler:
    """
    RbCP sampler/representer wrapper.

    This class delegates all RbCP operations to the trusted CPSample implementation:
    - DFT-based coefficient extraction: calc_an_bn_dft()
    - Phase parameter mapping: calc_ta_tb()
    - Event mapping (time-position pulses): CPSample.__call__()
    - Event decoding: event_to_ta_tb()
    - Reconstruction: recover_signal()

    The intention is to provide a clean PHY-style interface:
        sampler = RbCPSampler(config)
        ta, tb, x_used = sampler.sample(x, Tt)
        events = sampler.encode_events(x, Tt)
        ta_r, tb_r = sampler.decode_events(events)
        x_hat = sampler.recover(ta_r, tb_r, t)

    Notes on channel:
    - "Error-free channel" for RbCP means: the parameters/events are received
      exactly as sent (no collisions, no noise, no decoding mistakes). This is
      the baseline used in theoretical comparisons. [1](https://github.com/pedrogoria/Semantic-Functional-Communications)
    """

    def __init__(self, config: Optional[RbCPConfig] = None, **kwargs):
        """
        Initialize RbCPSampler.

        Parameters
        ----------
        config : RbCPConfig or None
            If provided, used as base configuration.
        kwargs : dict
            Overrides for config fields, or extra CPSample options.
        """
        if config is None:
            config = RbCPConfig()

        # Apply overrides from kwargs to dataclass fields when possible
        cfg_dict = config.__dict__.copy()
        for k, v in list(kwargs.items()):
            if k in cfg_dict:
                cfg_dict[k] = v
                kwargs.pop(k)

        self.config = RbCPConfig(**cfg_dict)

        # Build the trusted backend object
        # Any remaining kwargs are passed as CPSample **options
        self._backend = CPSample(
            T=self.config.T,
            harmonics=self.config.harmonics,
            n_sub_symbol=self.config.n_sub_symbol,
            resource=self.config.resource,
            sensor_nodes=self.config.sensor_nodes,
            bandwidth=self.config.bandwidth,
            detect_errors=self.config.detect_errors,
            threshold_harmonics=self.config.threshold_harmonics,
            dft_signal_periods=self.config.dft_signal_periods,
            **kwargs
        )

    @property
    def backend(self) -> CPSample:
        """
        Return the trusted backend instance for debugging/inspection.
        """
        return self._backend

    @property
    def w0(self) -> float:
        """
        Fundamental angular frequency (rad/s).
        """
        return self._backend.w0

    @property
    def n(self) -> np.ndarray:
        """
        Harmonic indices (1..N).
        """
        return self._backend.n

    @property
    def rs_per_period(self) -> int:
        """
        Number of resource slots per signal period used by the event mapping.
        """
        return int(self._backend.rs_per_period)

    def sample(self, x: np.ndarray, Tt: float, **options) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (ta, tb) parameters from a time-domain signal using trusted code.

        Parameters
        ----------
        x : np.ndarray
            Signal samples. Trusted backend accepts shapes:
            - (time,)
            - (time, periods)
            - (time, periods, sensors)
        Tt : float
            Sampling time step used to compute the DFT.
        options : dict
            Additional backend options (e.g., 't', 'normalize', 'norm').

        Returns
        -------
        ta : np.ndarray
            Phase-shift parameters ta, shape (periods, harmonics, sensors).
        tb : np.ndarray
            Phase-shift parameters tb, shape (periods, harmonics, sensors).
        x_used : np.ndarray
            Possibly normalized/scaled signal as returned by the backend.
        """
        ta, tb, x_used = self._backend.sample(x, Tt, **options)
        # Trusted code may return complex due to numerical operations;
        return np.real(ta), np.real(tb), x_used

    def encode_events(self, x: np.ndarray, Tt: float, **options) -> np.ndarray:
        """
        Encode a signal into an event matrix (pulse positions) using trusted CPSample.__call__.

        Parameters
        ----------
        x : np.ndarray
            Signal samples.
        Tt : float
            Sampling time step.
        options : dict
            Backend options forwarded to CPSample.__call__.

        Returns
        -------
        events : np.ndarray
            Binary event matrix of shape (time_slots, 2*harmonics*sensors).
        """
        events = self._backend(x, Tt, **options)
        return events

    def encode_events_from_params(self, ta: np.ndarray, tb: np.ndarray) -> np.ndarray:
        """
        Encode already-computed (ta, tb) into events using trusted mapping.

        This is useful when the channel model or other blocks produce ta/tb directly.

        Parameters
        ----------
        ta, tb : np.ndarray
            Phase parameters shaped as (periods, harmonics, sensors).

        Returns
        -------
        events : np.ndarray
            Binary event matrix.
        """
        events = self._backend(None, None, ta_tb=True, ta=ta, tb=tb)
        return events

    def decode_events(self, events: np.ndarray, **options) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Decode an event matrix back into (ta, tb) using trusted code.

        Parameters
        ----------
        events : np.ndarray
            Event matrix.
        options : dict
            Backend options forwarded to event_to_ta_tb().

        Returns
        -------
        ta : np.ndarray
            Decoded ta.
        tb : np.ndarray
            Decoded tb.
        signal_error : np.ndarray or None
            If detect_errors is enabled in the backend, returns the detected error flags.
            Otherwise returns None.
        """
        if self.config.detect_errors:
            ta, tb, signal_error = self._backend.event_to_ta_tb(events, **options)
            return ta, tb, signal_error
        else:
            ta, tb = self._backend.event_to_ta_tb(events, **options)
            return ta, tb, None

    def recover(self, ta: np.ndarray, tb: np.ndarray, t: np.ndarray, **options) -> np.ndarray:
        """
        Reconstruct the time-domain signal from (ta, tb) using trusted code.

        Parameters
        ----------
        ta, tb : np.ndarray
            Phase parameters.
        t : np.ndarray
            Time vector for reconstruction over one period (or more).
        options : dict
            Extra options (currently unused, kept for API stability).

        Returns
        -------
        xr : np.ndarray
            Reconstructed signal array shaped (len(t), periods, sensors).
        """
        xr = self._backend.recover_signal(ta, tb, t, **options)
        return xr


@dataclass
class DeltaConfig:
    """
    Configuration container for DeltaSampler.

    Delta representation was mentioned as an alternative representation method.
    However, the trusted code excerpt you provided only includes RbCP sampling
    and quantization. No trusted delta implementation was shown yet.

    This config is created now so the API is stable as the project grows.
    """
    T: float = 1.0
    sensor_nodes: int = 1

    # A typical delta-modulation representation would include:
    # - threshold (delta step)
    # - optional max event rate
    # These are placeholders until the trusted delta code is integrated.
    delta_step: float = 0.1


class DeltaSampler:
    """
    Delta sampler/representer placeholder.

    This class is intentionally implemented as a stub because the trusted delta
    implementation is not present in the provided reliable code excerpt.

    Once you share the delta code in sfc/sampling.py (or confirm where it lives),
    we will replace the internals while keeping this API stable.

    The intended interface parallels RbCPSampler:
        sampler = DeltaSampler(config)
        representation = sampler.sample(x, Tt)
        events = sampler.encode_events(...)
        x_hat = sampler.recover(...)
    """

    def __init__(self, config: Optional[DeltaConfig] = None, **kwargs):
        if config is None:
            config = DeltaConfig()
        cfg_dict = config.__dict__.copy()
        for k, v in list(kwargs.items()):
            if k in cfg_dict:
                cfg_dict[k] = v
                kwargs.pop(k)
        self.config = DeltaConfig(**cfg_dict)

        # Keep kwargs for future compatibility, but do not use them now.
        self._extra_options: Dict[str, Any] = kwargs

    def sample(self, x: np.ndarray, Tt: float, **options):
        """
        Placeholder for delta sampling/representation.

        Raises
        ------
        NotImplementedError
            Until the trusted delta implementation is integrated.
        """
        raise NotImplementedError(
            "DeltaSampler is not implemented yet because no trusted delta code was provided. "
            "Please point to the delta implementation in sfc/sampling.py (or share it) and we will wire it here."
        )

    def encode_events(self, *args, **kwargs):
        """
        Placeholder for delta-to-events mapping.
        """
        raise NotImplementedError(
            "DeltaSampler event encoding is not implemented yet. "
            "We need the trusted delta representation specification/code."
        )

    def decode_events(self, *args, **kwargs):
        """
        Placeholder for events-to-delta decoding.
        """
        raise NotImplementedError(
            "DeltaSampler event decoding is not implemented yet. "
            "We need the trusted delta representation specification/code."
        )

    def recover(self, *args, **kwargs):
        """
        Placeholder for delta reconstruction.
        """
        raise NotImplementedError(
            "DeltaSampler recovery is not implemented yet. "
            "We need the trusted delta representation specification/code."
        )
