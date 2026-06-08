"""
sfc/core/acquisition/sod.py

Send-on-Delta acquisition core.

This module implements a reusable Send-on-Delta (SoD) acquisition and
reconstruction core for fair-comparison pipelines.

Tensor convention
-----------------
Input and output signal tensors use:

    (time, periods, sensors)

Event-domain quantities are stored per period and per sensor because the number
of SoD events is signal-dependent and therefore variable.

Mathematical model
------------------
For each period p and sensor s, the source signal is represented by a dense
time-domain vector:

    x[p, s] in R^N

Given a threshold Delta_SoD > 0, the SoD sampler emits an event whenever the
signal deviates from the last transmitted value by at least Delta_SoD.

In continuous notation, if t_k is the most recent event time, then:

    t_{k+1} = inf { t > t_k : |x(t) - x(t_k)| >= Delta_SoD }.

In this implementation, the infimum is evaluated on the provided discrete time
grid. Therefore, event times are grid times.

What is transmitted?
--------------------
The natural SoD event payload is a time-amplitude pair:

    (t_k, x(t_k)).

By default, this core assumes that SoD transmits both event time and event
amplitude:

    transmit_event_times = True

This is the physically explicit SoD setting.

An optimistic amplitude-only setting is also supported through:

    transmit_event_times = False

In that case, event times are still used internally for reconstruction, but they
are not counted in the payload. This corresponds to an optimistic baseline in
which event timing is assumed to be available at the receiver.

Quantization
------------
Amplitude and time dequantized values are computed using the trusted
project-wide quantization function:

    sfc.core.quantization.quantize(...)

Therefore, when amplitude quantization is enabled, the same scalar quantization
rule Q(.) used by the Benchmark Approach can be used for SoD amplitudes.

Budget-aware construction
-------------------------
The class method:

    SoDAcquisitionCore.from_benchmark_budget(...)

uses:

    sfc.core.system_parameters.compute_benchmark_M_per_sensor(...)

to set the SoD amplitude quantization bins equal to the Benchmark Approach bins.
This preserves the comparison convention:

    M_SoD,amplitude = M_Benchmark

The time-bin budget is configurable. By default, if event-time transmission is
enabled and params.M_time is available, time_bins is set to params.M_time.

This module does not implement modulation, channel coding, AWGN, packetization,
or physical transmission. It provides acquisition, quantization, reconstruction,
and payload accounting metadata. Higher-level pipelines decide feasibility
under a channel budget.

Supported reconstruction rules
------------------------------
1. zero_order_hold:
       x_hat(t) = last received event amplitude.

2. linear:
       linear interpolation between consecutive received events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from sfc.core.quantization import quantize
from sfc.core.system_parameters import compute_benchmark_M_per_sensor


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class SoDAcquisitionResult:
    """
    Output of SoD acquisition.

    Attributes
    ----------
    event_times : dict
        key -> unquantized event times.

    event_values : dict
        key -> unquantized event amplitudes.

    event_time_indices : dict or None
        key -> quantized time-bin indices, if event-time quantization is enabled.

    event_value_indices : dict or None
        key -> quantized amplitude-bin indices, if amplitude quantization is enabled.

    event_times_quantized : dict or None
        key -> dequantized event times, if event-time quantization is enabled.

    event_values_quantized : dict or None
        key -> dequantized event amplitudes, if amplitude quantization is enabled.

    event_count : dict
        key -> number of events.

    payload_bits : dict
        key -> payload bits implied by the configured event representation.

    original_shape : tuple
        Shape of input signal tensor.

    time : np.ndarray
        Original time vector.

    threshold : float
        SoD threshold Delta_SoD.

    transmit_event_times : bool
        Whether event times are counted as transmitted payload.

    amplitude_bins : array-like or None
        The amplitude quantization bins actually used (scalar or per-sensor).

    time_bins : array-like or None
        The event-time quantization bins actually used (scalar or per-sensor).

    amplitude_quantization_metadata : dict
        Amplitude quantization metadata.

    time_quantization_metadata : dict
        Time quantization metadata.

    payload_metadata : dict
        Payload accounting metadata.

    aux : dict
        Additional diagnostics.
    """

    event_times: Dict[str, np.ndarray]
    event_values: Dict[str, np.ndarray]

    event_time_indices: Optional[Dict[str, np.ndarray]]
    event_value_indices: Optional[Dict[str, np.ndarray]]

    event_times_quantized: Optional[Dict[str, np.ndarray]]
    event_values_quantized: Optional[Dict[str, np.ndarray]]

    event_count: Dict[str, int]
    payload_bits: Dict[str, float]

    original_shape: Tuple[int, int, int]
    time: np.ndarray
    threshold: float
    transmit_event_times: bool

    amplitude_bins: Optional[np.ndarray]
    time_bins: Optional[np.ndarray]

    amplitude_quantization_metadata: Dict[str, Any] = field(default_factory=dict)
    time_quantization_metadata: Dict[str, Any] = field(default_factory=dict)
    payload_metadata: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)

    @property
    def events(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Return event streams used by the receiver.

        If quantized/dequantized event times or amplitudes are available, those
        are returned. Otherwise, the original unquantized values are returned.
        """
        out = {}

        for key in self.event_times:
            times = (
                self.event_times_quantized[key]
                if self.event_times_quantized is not None
                else self.event_times[key]
            )
            values = (
                self.event_values_quantized[key]
                if self.event_values_quantized is not None
                else self.event_values[key]
            )
            out[key] = (times, values)

        return out


@dataclass
class SoDReconstructionResult:
    """
    Output of SoD reconstruction.

    Attributes
    ----------
    reconstructed_signal : np.ndarray
        Reconstructed signal tensor with shape:

            (time, periods, sensors)

    reconstruction_mode : str
        Reconstruction rule.

    event_count : dict
        Event count per period/sensor.

    payload_bits : dict
        Payload bits per period/sensor.

    aux : dict
        Additional diagnostics.
    """

    reconstructed_signal: np.ndarray
    reconstruction_mode: str
    event_count: Dict[str, int]
    payload_bits: Dict[str, float]
    aux: Dict[str, Any] = field(default_factory=dict)

    @property
    def x_hat(self) -> np.ndarray:
        return self.reconstructed_signal


# =============================================================================
# CORE
# =============================================================================

class SoDAcquisitionCore:
    """
    Send-on-Delta acquisition and reconstruction core.

    Parameters
    ----------
    threshold : float
        Send-on-Delta threshold Delta_SoD.

    initial_event : bool
        If True, emit an event at the first time sample of every period/sensor.

    reconstruction_mode : str
        Supported:
        - "zero_order_hold"
        - "linear"

    transmit_event_times : bool
        If True, event times are counted in the payload and may be quantized.
        This is the default physically explicit SoD setting.

        If False, event times are not counted in the payload. This is an
        optimistic amplitude-only setting.

    quantize_amplitudes : bool
        If True, quantize event amplitudes using sfc.core.quantization.quantize.

    amplitude_bins : int, sequence, ndarray, or None
        Number of amplitude quantization bins. Can be scalar or per-sensor.

    amplitude_quantization_range : str
        Supported:
        - "per_signal"
        - "global"
        - "fixed"

    amplitude_quantization_min : float or None
        Lower bound for fixed amplitude quantization range.

    amplitude_quantization_max : float or None
        Upper bound for fixed amplitude quantization range.

    quantize_times : bool
        If True, quantize event times using sfc.core.quantization.quantize.

    time_bins : int, sequence, ndarray, or None
        Number of time quantization bins. Can be scalar or per-sensor.

    time_quantization_range : str
        Supported:
        - "period"
        - "fixed"

    time_quantization_min : float or None
        Lower bound for fixed time quantization range.

    time_quantization_max : float or None
        Upper bound for fixed time quantization range.

    quantization_poss : float
        Position offset inside the quantization bin passed to trusted quantize(...).
        The trusted default is 1/2.

    clip_quantization : bool
        If False, out-of-range values raise an error before calling quantize(...).
        If True, out-of-range values are clipped before calling quantize(...).
    """

    def __init__(
        self,
        threshold,
        initial_event=True,
        reconstruction_mode="zero_order_hold",
        transmit_event_times=True,
        quantize_amplitudes=False,
        amplitude_bins=None,
        amplitude_quantization_range="per_signal",
        amplitude_quantization_min=None,
        amplitude_quantization_max=None,
        quantize_times=False,
        time_bins=None,
        time_quantization_range="period",
        time_quantization_min=None,
        time_quantization_max=None,
        quantization_poss=1 / 2,
        clip_quantization=True,
        **kwargs
    ):
        _ = kwargs

        self.threshold = float(threshold)
        if self.threshold <= 0.0:
            raise ValueError("threshold must be positive.")

        self.initial_event = bool(initial_event)

        self.reconstruction_mode = str(reconstruction_mode).lower()
        if self.reconstruction_mode not in {"zero_order_hold", "linear"}:
            raise ValueError(
                "Unsupported reconstruction_mode. "
                "Supported: 'zero_order_hold', 'linear'."
            )

        self.transmit_event_times = bool(transmit_event_times)

        self.quantize_amplitudes = bool(quantize_amplitudes)
        self.amplitude_bins = amplitude_bins
        self.amplitude_quantization_range = str(amplitude_quantization_range).lower()
        self.amplitude_quantization_min = amplitude_quantization_min
        self.amplitude_quantization_max = amplitude_quantization_max

        self.quantize_times = bool(quantize_times)
        self.time_bins = time_bins
        self.time_quantization_range = str(time_quantization_range).lower()
        self.time_quantization_min = time_quantization_min
        self.time_quantization_max = time_quantization_max

        self.quantization_poss = float(quantization_poss)
        self.clip_quantization = bool(clip_quantization)

        if not self.transmit_event_times:
            self.quantize_times = False

        self._validate_static_config()

    # =========================================================================
    # BUDGET-AWARE CONSTRUCTION
    # =========================================================================

    @classmethod
    def from_benchmark_budget(
        cls,
        cfg,
        params,
        threshold,
        sampling_rate,
        transmit_event_times=True,
        time_bins=None,
        reconstruction_mode="zero_order_hold",
        initial_event=True,
        quantize_amplitudes=True,
        quantize_times=None,
        **kwargs
    ):
        """
        Construct a SoD core using Benchmark Approach amplitude bins.

        The amplitude quantization bins are computed with:

            compute_benchmark_M_per_sensor(...)

        Hence:

            M_SoD,amplitude = M_Benchmark

        Parameters
        ----------
        cfg : dict
            Parsed configuration.

        params : DerivedSystemParameters
            Output of build_derived_system_parameters(cfg).

        threshold : float
            SoD threshold.

        sampling_rate : float
            Sampling rate used by the Benchmark Approach.

        transmit_event_times : bool
            If True, event-time payload is counted and time quantization is
            enabled unless explicitly disabled.

        time_bins : int, sequence, ndarray, or None
            If None and transmit_event_times=True, uses params.M_time when
            available.

        reconstruction_mode : str
            Reconstruction rule.

        initial_event : bool
            Whether to emit the first sample as an event.

        quantize_amplitudes : bool
            Whether to quantize amplitudes.

        quantize_times : bool or None
            If None, equals transmit_event_times.
        """
        if quantize_times is None:
            quantize_times = bool(transmit_event_times)

        quant_cfg = cfg.get("quantization", {})

        amplitude_bins = compute_benchmark_M_per_sensor(
            S=params.S,
            tau=params.tau,
            B=params.B,
            P=params.P,
            N0=params.N0,
            sampling_rate=sampling_rate,
            bandwidth_allocation=params.bandwidth_allocation,
            force_power_of_two=bool(
                quant_cfg.get(
                    "force_power_of_two",
                    params.quantization_force_power_of_two,
                )
            ),
            rounding_mode=quant_cfg.get(
                "rounding_mode",
                params.quantization_rounding_mode,
            ),
        )

        if time_bins is None and bool(transmit_event_times):
            if hasattr(params, "M_time") and params.M_time is not None:
                time_bins = int(params.M_time)

        return cls(
            threshold=threshold,
            initial_event=initial_event,
            reconstruction_mode=reconstruction_mode,
            transmit_event_times=transmit_event_times,
            quantize_amplitudes=quantize_amplitudes,
            amplitude_bins=amplitude_bins,
            quantize_times=bool(quantize_times),
            time_bins=time_bins,
            **kwargs
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def acquire(
        self,
        x,
        t,
        **kwargs
    ):
        """
        Acquire SoD events from a signal tensor.
        """
        _ = kwargs

        x = _ensure_3d_signal_tensor(x)
        t = _ensure_1d_time_vector(t)

        if x.shape[0] != len(t):
            raise ValueError(
                "x.shape[0]={} must match len(t)={}.".format(x.shape[0], len(t))
            )

        _, n_periods, n_sensors = x.shape

        amplitude_bins_vec = _resolve_bins_vector(
            self.amplitude_bins,
            n_sensors,
            name="amplitude_bins",
            required=self.quantize_amplitudes,
        )

        time_bins_vec = _resolve_bins_vector(
            self.time_bins,
            n_sensors,
            name="time_bins",
            required=self.quantize_times,
        )

        event_times = {}
        event_values = {}
        event_count = {}

        for p in range(n_periods):
            for s in range(n_sensors):
                key = _event_key(p, s)

                times, values = self._detect_events_1d(
                    x_vec=x[:, p, s],
                    t=t,
                )

                event_times[key] = times
                event_values[key] = values
                event_count[key] = int(len(times))

        (
            event_value_indices,
            event_values_quantized,
            amplitude_metadata,
        ) = self._maybe_quantize_amplitudes(
            x=x,
            event_values=event_values,
            amplitude_bins_vec=amplitude_bins_vec,
        )

        (
            event_time_indices,
            event_times_quantized,
            time_metadata,
        ) = self._maybe_quantize_times(
            t=t,
            event_times=event_times,
            time_bins_vec=time_bins_vec,
        )

        payload_bits, payload_metadata = self._compute_payload_bits(
            event_count=event_count,
            amplitude_bins_vec=amplitude_bins_vec,
            time_bins_vec=time_bins_vec,
            n_periods=n_periods,
            n_sensors=n_sensors,
        )

        aux = {
            "threshold": float(self.threshold),
            "initial_event": bool(self.initial_event),
            "reconstruction_mode": self.reconstruction_mode,
            "transmit_event_times": bool(self.transmit_event_times),
            "quantize_amplitudes": bool(self.quantize_amplitudes),
            "quantize_times": bool(self.quantize_times),
            "quantization_function": "sfc.core.quantization.quantize",
        }

        return SoDAcquisitionResult(
            event_times=event_times,
            event_values=event_values,
            event_time_indices=event_time_indices,
            event_value_indices=event_value_indices,
            event_times_quantized=event_times_quantized,
            event_values_quantized=event_values_quantized,
            event_count=event_count,
            payload_bits=payload_bits,
            original_shape=x.shape,
            time=t,
            threshold=self.threshold,
            transmit_event_times=self.transmit_event_times,
            amplitude_bins=amplitude_bins_vec,
            time_bins=time_bins_vec,
            amplitude_quantization_metadata=amplitude_metadata,
            time_quantization_metadata=time_metadata,
            payload_metadata=payload_metadata,
            aux=aux,
        )

    def reconstruct(
        self,
        acquisition_result,
        t_eval=None,
        **kwargs
    ):
        """
        Reconstruct a signal tensor from a SoD event stream.
        """
        _ = kwargs

        acq = acquisition_result

        if t_eval is None:
            t_eval = acq.time
        else:
            t_eval = _ensure_1d_time_vector(t_eval)

        _, n_periods, n_sensors = acq.original_shape
        x_hat = np.zeros((len(t_eval), n_periods, n_sensors), dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                key = _event_key(p, s)

                times = (
                    acq.event_times_quantized[key]
                    if acq.event_times_quantized is not None
                    else acq.event_times[key]
                )
                values = (
                    acq.event_values_quantized[key]
                    if acq.event_values_quantized is not None
                    else acq.event_values[key]
                )

                x_hat[:, p, s] = self._reconstruct_1d(
                    event_times=times,
                    event_values=values,
                    t_eval=t_eval,
                    mode=self.reconstruction_mode,
                )

        aux = {
            "threshold": float(acq.threshold),
            "used_quantized_times": acq.event_times_quantized is not None,
            "used_quantized_values": acq.event_values_quantized is not None,
            "transmit_event_times": bool(acq.transmit_event_times),
        }

        return SoDReconstructionResult(
            reconstructed_signal=x_hat,
            reconstruction_mode=self.reconstruction_mode,
            event_count=acq.event_count,
            payload_bits=acq.payload_bits,
            aux=aux,
        )

    def acquire_and_reconstruct(
        self,
        x,
        t,
        t_eval=None,
        **kwargs
    ):
        """
        Convenience method: acquire SoD events and reconstruct immediately.
        """
        acq = self.acquire(x=x, t=t, **kwargs)
        rec = self.reconstruct(acquisition_result=acq, t_eval=t_eval)
        return acq, rec

    # =========================================================================
    # EVENT DETECTION
    # =========================================================================

    def _detect_events_1d(
        self,
        x_vec,
        t,
    ):
        """
        Detect SoD events for one 1D signal.
        """
        x_vec = np.asarray(x_vec, dtype=float).reshape(-1)
        t = np.asarray(t, dtype=float).reshape(-1)

        if x_vec.shape[0] != t.shape[0]:
            raise ValueError("x_vec and t must have the same length.")

        if x_vec.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        event_times = []
        event_values = []

        last_value = float(x_vec[0])

        if self.initial_event:
            event_times.append(float(t[0]))
            event_values.append(last_value)

        for i in range(1, len(x_vec)):
            xi = float(x_vec[i])

            if abs(xi - last_value) >= self.threshold:
                event_times.append(float(t[i]))
                event_values.append(xi)
                last_value = xi

        if not event_times:
            event_times.append(float(t[0]))
            event_values.append(float(x_vec[0]))

        return (
            np.asarray(event_times, dtype=float),
            np.asarray(event_values, dtype=float),
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_static_config(self):
        """
        Validate configuration that does not require knowing the number of sensors.
        """
        if self.amplitude_quantization_range not in {
            "per_signal",
            "global",
            "fixed",
        }:
            raise ValueError(
                "Unsupported amplitude_quantization_range. "
                "Supported: 'per_signal', 'global', 'fixed'."
            )

        if self.amplitude_quantization_range == "fixed":
            if self.amplitude_quantization_min is None:
                raise ValueError(
                    "amplitude_quantization_min is required for fixed range."
                )
            if self.amplitude_quantization_max is None:
                raise ValueError(
                    "amplitude_quantization_max is required for fixed range."
                )
            if self.amplitude_quantization_max <= self.amplitude_quantization_min:
                raise ValueError(
                    "amplitude_quantization_max must be larger than "
                    "amplitude_quantization_min."
                )

        if self.time_quantization_range not in {"period", "fixed"}:
            raise ValueError(
                "Unsupported time_quantization_range. "
                "Supported: 'period', 'fixed'."
            )

        if self.time_quantization_range == "fixed":
            if self.time_quantization_min is None:
                raise ValueError(
                    "time_quantization_min is required for fixed range."
                )
            if self.time_quantization_max is None:
                raise ValueError(
                    "time_quantization_max is required for fixed range."
                )
            if self.time_quantization_max <= self.time_quantization_min:
                raise ValueError(
                    "time_quantization_max must be larger than "
                    "time_quantization_min."
                )

        if self.quantize_times and self.time_bins is None:
            raise ValueError(
                "time_bins must be provided when quantize_times=True, unless "
                "using SoDAcquisitionCore.from_benchmark_budget(...) with "
                "params.M_time available."
            )

        if self.quantize_amplitudes and self.amplitude_bins is None:
            raise ValueError(
                "amplitude_bins must be provided when quantize_amplitudes=True, "
                "unless using SoDAcquisitionCore.from_benchmark_budget(...)."
            )

    # =========================================================================
    # QUANTIZATION
    # =========================================================================

    def _maybe_quantize_amplitudes(
        self,
        x,
        event_values,
        amplitude_bins_vec,
    ):
        """
        Quantize event amplitudes if enabled.
        """
        if not self.quantize_amplitudes:
            return None, None, {"enabled": False}

        if amplitude_bins_vec is None:
            raise ValueError("amplitude_bins_vec is required.")

        q_indices = {}
        values_tilde = {}

        metadata = {
            "enabled": True,
            "mode": "trusted_quantize",
            "function": "sfc.core.quantization.quantize",
            "range_mode": self.amplitude_quantization_range,
            "clip": bool(self.clip_quantization),
            "poss": float(self.quantization_poss),
            "bins_per_sensor": np.asarray(amplitude_bins_vec, dtype=int).tolist(),
            "per_signal": {},
        }

        _, n_periods, n_sensors = x.shape

        if self.amplitude_quantization_range == "global":
            value_min = float(np.min(x))
            value_max = float(np.max(x))

            for p in range(n_periods):
                for s in range(n_sensors):
                    key = _event_key(p, s)
                    bins = int(amplitude_bins_vec[s])

                    q, v_tilde, meta = _quantize_values_with_trusted_core(
                        values=event_values[key],
                        value_min=value_min,
                        value_max=value_max,
                        bins=bins,
                        poss=self.quantization_poss,
                        clip=self.clip_quantization,
                    )

                    q_indices[key] = q
                    values_tilde[key] = v_tilde
                    metadata["per_signal"][key] = meta

            metadata["global_range"] = {
                "value_min": value_min,
                "value_max": value_max,
            }
            return q_indices, values_tilde, metadata

        if self.amplitude_quantization_range == "fixed":
            value_min = float(self.amplitude_quantization_min)
            value_max = float(self.amplitude_quantization_max)

            for p in range(n_periods):
                for s in range(n_sensors):
                    key = _event_key(p, s)
                    bins = int(amplitude_bins_vec[s])

                    q, v_tilde, meta = _quantize_values_with_trusted_core(
                        values=event_values[key],
                        value_min=value_min,
                        value_max=value_max,
                        bins=bins,
                        poss=self.quantization_poss,
                        clip=self.clip_quantization,
                    )

                    q_indices[key] = q
                    values_tilde[key] = v_tilde
                    metadata["per_signal"][key] = meta

            metadata["fixed_range"] = {
                "value_min": value_min,
                "value_max": value_max,
            }
            return q_indices, values_tilde, metadata

        # per_signal
        for p in range(n_periods):
            for s in range(n_sensors):
                key = _event_key(p, s)
                bins = int(amplitude_bins_vec[s])

                value_min = float(np.min(x[:, p, s]))
                value_max = float(np.max(x[:, p, s]))

                q, v_tilde, meta = _quantize_values_with_trusted_core(
                    values=event_values[key],
                    value_min=value_min,
                    value_max=value_max,
                    bins=bins,
                    poss=self.quantization_poss,
                    clip=self.clip_quantization,
                )

                q_indices[key] = q
                values_tilde[key] = v_tilde
                metadata["per_signal"][key] = meta

        return q_indices, values_tilde, metadata

    def _maybe_quantize_times(
        self,
        t,
        event_times,
        time_bins_vec,
    ):
        """
        Quantize event times if enabled.
        """
        if not self.quantize_times:
            return None, None, {"enabled": False}

        if time_bins_vec is None:
            raise ValueError("time_bins_vec is required.")

        q_indices = {}
        times_tilde = {}

        metadata = {
            "enabled": True,
            "mode": "trusted_quantize",
            "function": "sfc.core.quantization.quantize",
            "range_mode": self.time_quantization_range,
            "clip": bool(self.clip_quantization),
            "poss": float(self.quantization_poss),
            "bins_per_sensor": np.asarray(time_bins_vec, dtype=int).tolist(),
            "per_signal": {},
        }

        if self.time_quantization_range == "period":
            dt = _infer_dt(t)
            value_min = float(t[0])
            value_max = float(t[-1] + dt)
        elif self.time_quantization_range == "fixed":
            value_min = float(self.time_quantization_min)
            value_max = float(self.time_quantization_max)
        else:
            raise ValueError("Invalid time_quantization_range.")

        n_periods = len(set([_period_index_from_key(k) for k in event_times.keys()]))
        n_sensors = len(time_bins_vec)

        for p in range(n_periods):
            for s in range(n_sensors):
                key = _event_key(p, s)
                if key not in event_times:
                    continue

                bins = int(time_bins_vec[s])

                q, t_tilde, meta = _quantize_values_with_trusted_core(
                    values=event_times[key],
                    value_min=value_min,
                    value_max=value_max,
                    bins=bins,
                    poss=self.quantization_poss,
                    clip=self.clip_quantization,
                )

                q_indices[key] = q
                times_tilde[key] = t_tilde
                metadata["per_signal"][key] = meta

        metadata["time_range"] = {
            "value_min": value_min,
            "value_max": value_max,
        }

        return q_indices, times_tilde, metadata

    # =========================================================================
    # PAYLOAD ACCOUNTING
    # =========================================================================

    def _compute_payload_bits(
        self,
        event_count,
        amplitude_bins_vec,
        time_bins_vec,
        n_periods,
        n_sensors,
    ):
        """
        Compute payload bits implied by the configured SoD event representation.
        """
        payload_bits = {}

        metadata = {
            "transmit_event_times": bool(self.transmit_event_times),
            "amplitude_bins_per_sensor": (
                None
                if amplitude_bins_vec is None
                else np.asarray(amplitude_bins_vec, dtype=int).tolist()
            ),
            "time_bins_per_sensor": (
                None
                if time_bins_vec is None
                else np.asarray(time_bins_vec, dtype=int).tolist()
            ),
            "bits_per_event": {},
        }

        for p in range(n_periods):
            for s in range(n_sensors):
                key = _event_key(p, s)
                K = int(event_count[key])

                bits_amp = 0.0
                bits_time = 0.0

                if self.quantize_amplitudes and amplitude_bins_vec is not None:
                    bits_amp = float(np.log2(max(int(amplitude_bins_vec[s]), 1)))

                if self.transmit_event_times:
                    if self.quantize_times and time_bins_vec is not None:
                        bits_time = float(np.log2(max(int(time_bins_vec[s]), 1)))
                    else:
                        bits_time = 0.0

                bits_per_event = bits_amp + bits_time
                payload_bits[key] = float(K * bits_per_event)

                metadata["bits_per_event"][key] = {
                    "amplitude_bits": bits_amp,
                    "time_bits": bits_time,
                    "total_bits": bits_per_event,
                }

        metadata["total_payload_bits"] = float(sum(payload_bits.values()))

        return payload_bits, metadata

    # =========================================================================
    # RECONSTRUCTION
    # =========================================================================

    @staticmethod
    def _reconstruct_1d(
        event_times,
        event_values,
        t_eval,
        mode,
    ):
        """
        Reconstruct one 1D signal from one event stream.
        """
        event_times = np.asarray(event_times, dtype=float).reshape(-1)
        event_values = np.asarray(event_values, dtype=float).reshape(-1)
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

        if event_times.size != event_values.size:
            raise ValueError("event_times and event_values must have the same length.")

        if event_times.size == 0:
            return np.zeros_like(t_eval, dtype=float)

        order = np.argsort(event_times)
        event_times = event_times[order]
        event_values = event_values[order]

        event_times, event_values = _deduplicate_event_times(
            event_times,
            event_values,
        )

        if mode == "zero_order_hold":
            return _zero_order_hold(
                event_times=event_times,
                event_values=event_values,
                t_eval=t_eval,
            )

        if mode == "linear":
            return np.interp(
                t_eval,
                event_times,
                event_values,
                left=event_values[0],
                right=event_values[-1],
            )

        raise ValueError("Unsupported reconstruction mode: {}".format(mode))


# =============================================================================
# TRUSTED-QUANTIZATION WRAPPER
# =============================================================================

def _quantize_values_with_trusted_core(
    values,
    value_min,
    value_max,
    bins,
    poss,
    clip,
):
    """
    Quantize values using sfc.core.quantization.quantize(...).

    The trusted quantize(...) returns the dequantized values, not the integer
    indices. Therefore, this helper computes integer indices using the same
    bin-selection rule only for payload accounting and diagnostics.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    bins = int(np.floor(bins))

    if bins < 1:
        raise ValueError("bins must be >= 1.")

    if not np.isfinite(value_min) or not np.isfinite(value_max):
        raise ValueError("Quantization range must be finite.")

    if value_max < value_min:
        raise ValueError("value_max must be >= value_min.")

    if np.isclose(value_max, value_min):
        q = np.zeros(values.shape, dtype=np.int64)
        values_q = np.full(values.shape, fill_value=value_min, dtype=float)

        metadata = {
            "enabled": True,
            "bins": int(bins),
            "mode": "trusted_quantize",
            "function": "sfc.core.quantization.quantize",
            "value_min": float(value_min),
            "value_max": float(value_max),
            "delta": 0.0,
            "poss": float(poss),
            "saturated_low": 0,
            "saturated_high": 0,
            "degenerate": True,
        }

        return q, values_q, metadata

    delta = (value_max - value_min) / float(bins)

    below = values < value_min
    above = values > value_max

    if not clip and (np.any(below) or np.any(above)):
        raise ValueError("Input contains values outside quantization range.")

    values_clip = np.clip(values, value_min, value_max)

    x_s = np.ceil((values_clip - value_min) / delta)
    x_s[np.where(x_s > bins)] = bins
    x_s[np.where(x_s <= 0)] = 1

    q = x_s.astype(np.int64) - 1

    values_q = quantize(
        values_clip,
        value_min,
        value_max,
        bins,
        poss=poss,
    )

    metadata = {
        "enabled": True,
        "bins": int(bins),
        "mode": "trusted_quantize",
        "function": "sfc.core.quantization.quantize",
        "value_min": float(value_min),
        "value_max": float(value_max),
        "delta": float(delta),
        "poss": float(poss),
        "saturated_low": int(np.sum(below)),
        "saturated_high": int(np.sum(above)),
        "degenerate": False,
    }

    return q, np.asarray(values_q, dtype=float).reshape(-1), metadata


# =============================================================================
# RECONSTRUCTION HELPERS
# =============================================================================

def _zero_order_hold(
    event_times,
    event_values,
    t_eval,
):
    """
    Zero-order-hold reconstruction from an event stream.
    """
    event_times = np.asarray(event_times, dtype=float).reshape(-1)
    event_values = np.asarray(event_values, dtype=float).reshape(-1)
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

    indices = np.searchsorted(event_times, t_eval, side="right") - 1
    indices = np.clip(indices, 0, len(event_values) - 1)

    return event_values[indices]


def _deduplicate_event_times(
    event_times,
    event_values,
):
    """
    Remove repeated event times by keeping the last value for each repeated time.
    """
    if event_times.size <= 1:
        return event_times, event_values

    unique_times = []
    unique_values = []

    i = 0
    n = len(event_times)

    while i < n:
        current_time = event_times[i]
        j = i + 1

        while j < n and np.isclose(event_times[j], current_time):
            j += 1

        unique_times.append(current_time)
        unique_values.append(event_values[j - 1])

        i = j

    return (
        np.asarray(unique_times, dtype=float),
        np.asarray(unique_values, dtype=float),
    )


# =============================================================================
# VALIDATION / UTILITY HELPERS
# =============================================================================

def _ensure_3d_signal_tensor(x):
    """
    Ensure x has shape (time, periods, sensors).
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        return x[:, None, None]

    if x.ndim == 2:
        return x[:, :, None]

    if x.ndim == 3:
        return x

    raise ValueError(
        "Signal tensor must have shape (time,), (time, periods), "
        "or (time, periods, sensors)."
    )


def _ensure_1d_time_vector(t):
    """
    Ensure t is a 1D, strictly increasing time vector.
    """
    t = np.asarray(t, dtype=float).reshape(-1)

    if t.ndim != 1:
        raise ValueError("t must be a 1D vector.")

    if len(t) < 2:
        raise ValueError("t must contain at least two samples.")

    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")

    return t


def _infer_dt(t):
    """
    Infer the time step from a 1D grid.
    """
    t = _ensure_1d_time_vector(t)

    dt = float(t[1] - t[0])
    if dt <= 0.0:
        raise ValueError("Time step must be positive.")

    return dt


def _event_key(period_index, sensor_index):
    """
    Standard key for period/sensor event dictionaries.
    """
    return "period_{}_sensor_{}".format(int(period_index), int(sensor_index))


def _period_index_from_key(key):
    """
    Extract period index from the standard key period_p_sensor_s.
    """
    try:
        left = str(key).split("_sensor_")[0]
        return int(left.split("period_")[1])
    except Exception as exc:
        raise ValueError("Invalid event key: {}".format(key)) from exc


def _resolve_bins_vector(
    bins,
    n_sensors,
    name,
    required,
):
    """
    Resolve scalar or per-sensor bins into a vector of shape (n_sensors,).
    """
    if bins is None:
        if required:
            raise ValueError("{} must be provided.".format(name))
        return None

    arr = np.asarray(bins, dtype=object).reshape(-1)

    if arr.size == 1:
        value = int(arr[0])
        if value < 1:
            raise ValueError("{} must be >= 1.".format(name))
        return np.full(n_sensors, value, dtype=int)

    if arr.size != n_sensors:
        raise ValueError(
            "{} must be scalar or have length n_sensors={}. Got length {}.".format(
                name, n_sensors, arr.size
            )
        )

    out = np.asarray([int(v) for v in arr], dtype=int)

    if np.any(out < 1):
        raise ValueError("All entries of {} must be >= 1.".format(name))

    return out


__all__ = [
    "SoDAcquisitionCore",
    "SoDAcquisitionResult",
    "SoDReconstructionResult",
]
