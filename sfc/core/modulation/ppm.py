"""
sfc/core/modulation/ppm.py

Pulse-Position Modulation (PPM) core.

Purpose
-------
This module implements a reusable, MAC-agnostic PPM modem that is compatible
with the new modulation-layer architecture:

    sfc/core/modulation/

Design principles
-----------------
1. Sensor-aware:
   The implementation supports multiple sensors.

2. Period-aware:
   The implementation supports multiple periods.

3. MAC-agnostic:
   This module generates one waveform per sensor and per period, but it does
   NOT decide how multiple sensors share the medium. MAC-layer combination
   belongs elsewhere (e.g. SFC / TDMA / FDMA / OFDM).

4. Pulse-shaping ready:
   The transmit pulse is obtained through:
       sfc.core.modulation.pulse_shaping

   so that pulse-shaping choices (e.g. rectangular, raised cosine) can be
   reused by other modulation schemes and even by future SFC PHY variants.

Tensor conventions
------------------
Continuous-time signal tensors use the canonical shape:

    (time, periods, sensors)

Symbol-domain quantities use:

    (symbols, periods, sensors)

MATLAB-reference correspondence
-------------------------------
This implementation follows the same conceptual steps as the provided MATLAB
reference, while generalizing them to:
- multiple sensors
- multiple periods
- tensorized handling
- separation between modulation and MAC
- reusable pulse-shaping

Current assumptions
-------------------
- The time grid t corresponds to ONE signal period, typically [0, tau).
- The same time grid is reused for all periods in the tensor.
- The PPM message is normalized into the open interval:
      [eps_margin, 1 - eps_margin]
  before modulation.
- Demodulation returns recovered normalized samples and, when possible,
  denormalized samples and continuous-time reconstruction.

Notes
-----
- This module does NOT add AWGN/noise by itself.
- This module does NOT perform plots/PSD/debug visualizations.
- This module does NOT implement MAC/resource sharing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from sfc.core.modulation.base import (
    DemodulationResult,
    ModulationCoreBase,
    ModulationResult,
    NormalizationState,
    ensure_1d_time_vector,
    ensure_3d_signal_tensor,
    validate_time_axis_length,
)
from sfc.core.modulation.pulse_shaping import (
    build_matched_filter,
    build_tx_pulse,
)


class PPMCore(ModulationCoreBase):
    """
    Pulse-Position Modulation (PPM) modem core.

    Parameters
    ----------
    fc : float
        Symbol / pulse repetition frequency in Hz.

    pulse_width : float
        Effective pulse width in seconds.

    rec_pulse : float, optional
        Recovery / guard parameter following the MATLAB reference.
        Default is 0.

    pulse_type : str, optional
        Pulse-shaping type.
        Supported values depend on pulse_shaping.py, but the intended main
        options are:
            - "rect"
            - "raised_cosine"

    rolloff : float, optional
        Raised-cosine rolloff factor, typically in [0, 1].
        Used when pulse_type = "raised_cosine".

    span : int, optional
        Raised-cosine span in symbols.
        Used when pulse_type = "raised_cosine".

    eps_margin : float, optional
        Margin used when normalizing the message into the open interval:
            [eps_margin, 1 - eps_margin]

    interp_mode : str, optional
        Sampling interpolation mode.
        Current implementation uses linear interpolation through numpy.

    periodic_replicas : int, optional
        Number of period replicas used on each side when reconstructing the
        continuous-time waveform through periodic sinc reconstruction.

    clip_recovered_to_unit_interval : bool, optional
        Whether to clip recovered normalized samples into:
            [eps_margin, 1 - eps_margin]
        Default is True.
    """

    def __init__(
        self,
        fc: float,
        pulse_width: float,
        rec_pulse: float = 0.0,
        pulse_type: str = "raised_cosine",
        rolloff: float = 0.99,
        span: int = 12,
        eps_margin: float = 1e-3,
        interp_mode: str = "linear",
        periodic_replicas: int = 10,
        clip_recovered_to_unit_interval: bool = True,
        **kwargs
    ):
        super().__init__(
            fc=fc,
            pulse_width=pulse_width,
            rec_pulse=rec_pulse,
            pulse_type=pulse_type,
            rolloff=rolloff,
            span=span,
            eps_margin=eps_margin,
            interp_mode=interp_mode,
            periodic_replicas=periodic_replicas,
            clip_recovered_to_unit_interval=clip_recovered_to_unit_interval,
            **kwargs,
        )

        if fc <= 0:
            raise ValueError("fc must be positive.")

        if pulse_width <= 0:
            raise ValueError("pulse_width must be positive.")

        if eps_margin < 0 or eps_margin >= 0.5:
            raise ValueError("eps_margin must satisfy 0 <= eps_margin < 0.5.")

        self.fc = float(fc)
        self.Tc = 1.0 / self.fc
        self.pulse_width = float(pulse_width)
        self.rec_pulse = float(rec_pulse)

        self.pulse_type = str(pulse_type)
        self.rolloff = float(rolloff)
        self.span = int(span)

        self.eps_margin = float(eps_margin)
        self.interp_mode = str(interp_mode)
        self.periodic_replicas = int(periodic_replicas)
        self.clip_recovered_to_unit_interval = bool(clip_recovered_to_unit_interval)

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def normalize_message(
        self,
        x: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, NormalizationState]:
        """
        Normalize each (period, sensor) waveform into:

            [eps_margin, 1 - eps_margin]

        Parameters
        ----------
        x : np.ndarray
            Signal tensor with shape:
                (time, periods, sensors)

        Returns
        -------
        tuple
            (x_normalized, normalization_state)

        Notes
        -----
        The normalization is performed independently for each (period, sensor)
        pair, preserving the MATLAB-reference spirit while extending naturally
        to multiple sensors and periods.
        """

        _ = kwargs

        x = ensure_3d_signal_tensor(x).astype(float, copy=False)

        _, n_periods, n_sensors = x.shape

        x_min = np.min(x, axis=0)  # shape: (periods, sensors)
        x_max = np.max(x, axis=0)  # shape: (periods, sensors)
        denom = x_max - x_min

        x_norm = np.empty_like(x, dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                if np.isclose(denom[p, s], 0.0):
                    # Degenerate constant signal: place it at the center
                    x_norm[:, p, s] = 0.5
                else:
                    x_01 = (x[:, p, s] - x_min[p, s]) / denom[p, s]
                    x_norm[:, p, s] = (
                        self.eps_margin
                        + (1.0 - 2.0 * self.eps_margin) * x_01
                    )

        state = NormalizationState(
            enabled=True,
            eps_margin=self.eps_margin,
            metadata={
                "x_min": x_min,
                "x_max": x_max,
            },
        )

        return x_norm, state

    def denormalize_message(
        self,
        x: np.ndarray,
        normalization_state: Optional[NormalizationState],
        **kwargs
    ) -> np.ndarray:
        """
        Invert the normalization defined in normalize_message(...).

        Parameters
        ----------
        x : np.ndarray
            Normalized samples with shape:
                (symbols, periods, sensors)
            or another tensor compatible with the same (period, sensor) metadata.

        normalization_state : NormalizationState
            State produced by normalize_message(...)

        Returns
        -------
        np.ndarray
            Denormalized tensor.
        """

        _ = kwargs

        if normalization_state is None or not normalization_state.enabled:
            return x

        x = np.asarray(x, dtype=float)
        x_min = np.asarray(normalization_state.metadata["x_min"], dtype=float)
        x_max = np.asarray(normalization_state.metadata["x_max"], dtype=float)

        # Broadcast over the first axis (symbols or time)
        # x_min/x_max have shape (periods, sensors)
        x_denorm = np.empty_like(x, dtype=float)

        denom = 1.0 - 2.0 * normalization_state.eps_margin
        if np.isclose(denom, 0.0):
            raise ValueError("Invalid normalization: eps_margin makes denominator zero.")

        leading_dim = x.shape[0]

        for k in range(leading_dim):
            x_denorm[k, ...] = x_min + (
                (x[k, ...] - normalization_state.eps_margin) / denom
            ) * (x_max - x_min)

        return x_denorm

    # =========================================================================
    # SYMBOL-DOMAIN HELPERS
    # =========================================================================

    def sample_message(
        self,
        x: np.ndarray,
        t: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample the normalized continuous-time message at the symbol rate fc.

        Parameters
        ----------
        x : np.ndarray
            Input tensor with shape:
                (time, periods, sensors)

        t : np.ndarray
            Continuous-time grid for one period.

        Returns
        -------
        tuple
            (x_sampled, symbol_times)

            x_sampled shape:
                (symbols, periods, sensors)

            symbol_times shape:
                (symbols,)
        """

        _ = kwargs

        x = ensure_3d_signal_tensor(x).astype(float, copy=False)
        t = ensure_1d_time_vector(t)
        validate_time_axis_length(x, t)

        dt = _infer_dt(t)
        tau = _infer_tau(t, dt)

        symbol_times = np.arange(0.0, tau, self.Tc, dtype=float)
        n_symbols = len(symbol_times)
        _, n_periods, n_sensors = x.shape

        x_sampled = np.zeros((n_symbols, n_periods, n_sensors), dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                x_sampled[:, p, s] = np.interp(
                    symbol_times,
                    t,
                    x[:, p, s]
                )

        return x_sampled, symbol_times

    def compute_pulse_positions(
        self,
        x_sampled: np.ndarray,
        symbol_times: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Map normalized symbol values into pulse positions inside each symbol slot.

        Formula
        -------
        Following the MATLAB-reference logic:

            t_pulse = t0 + pulse_width/2
                      + x_sampled * (Tc - (1 + rec_pulse) * pulse_width)

        Parameters
        ----------
        x_sampled : np.ndarray
            Normalized symbol-domain message with shape:
                (symbols, periods, sensors)

        symbol_times : np.ndarray
            Symbol reference times (typically symbol starts), shape:
                (symbols,)

        Returns
        -------
        np.ndarray
            Pulse positions with shape:
                (symbols, periods, sensors)
        """

        _ = kwargs

        x_sampled = np.asarray(x_sampled, dtype=float)
        symbol_times = np.asarray(symbol_times, dtype=float).reshape(-1)

        if x_sampled.shape[0] != len(symbol_times):
            raise ValueError(
                f"x_sampled.shape[0]={x_sampled.shape[0]} does not match "
                f"len(symbol_times)={len(symbol_times)}"
            )

        displacement_scale = self.Tc - (1.0 + self.rec_pulse) * self.pulse_width
        if displacement_scale <= 0:
            raise ValueError(
                "Invalid PPM geometry: Tc - (1 + rec_pulse) * pulse_width must be positive."
            )

        pulse_positions = (
            symbol_times[:, None, None]
            + self.pulse_width / 2.0
            + x_sampled * displacement_scale
        )

        return pulse_positions

    # =========================================================================
    # MODULATION
    # =========================================================================

    def modulate(
        self,
        x: np.ndarray,
        t: np.ndarray,
        **kwargs
    ) -> ModulationResult:
        """
        Modulate a continuous-time message tensor into a PPM waveform tensor.

        Parameters
        ----------
        x : np.ndarray
            Input continuous-time signal.
            Supported shapes:
                (time,)
                (time, periods)
                (time, periods, sensors)

        t : np.ndarray
            Time grid for one period.

        Returns
        -------
        ModulationResult
            Structured modulation outputs.

        Main outputs
        ------------
        - tx_waveform:
            (time, periods, sensors)

        - sampled_message:
            (symbols, periods, sensors)

        - symbol_times:
            (symbols,)

        - aux["pulse_positions"]:
            (symbols, periods, sensors)

        - aux["tx_pulse"]:
            1D pulse-shaping filter
        """

        # Convert to canonical tensor form
        x = ensure_3d_signal_tensor(x).astype(float, copy=False)
        t = ensure_1d_time_vector(t)
        validate_time_axis_length(x, t)

        dt = _infer_dt(t)
        tau = _infer_tau(t, dt)

        # 1) Normalize message into the admissible PPM interval
        x_norm, norm_state = self.normalize_message(x)

        # 2) Sample message at fc
        x_sampled, symbol_times = self.sample_message(x_norm, t)

        # 3) Compute pulse positions
        pulse_positions = self.compute_pulse_positions(x_sampled, symbol_times)

        # 4) Build transmit pulse
        tx_pulse = build_tx_pulse(
            pulse_type=self.pulse_type,
            pulse_width=self.pulse_width,
            dt=dt,
            rolloff=self.rolloff,
            span=self.span,
            normalize_energy=True,
        )

        # 5) Build waveform per (period, sensor)
        tx_waveform = self._build_ppm_waveform_tensor(
            pulse_positions=pulse_positions,
            tx_pulse=tx_pulse,
            t=t
        )

        return ModulationResult(
            tx_waveform=tx_waveform,
            sampled_message=x_sampled,
            symbol_times=symbol_times,
            normalization_state=norm_state,
            aux={
                "pulse_positions": pulse_positions,
                "tx_pulse": tx_pulse,
                "dt": dt,
                "tau": tau,
                "Tc": self.Tc,
                "pulse_width": self.pulse_width,
                "pulse_type": self.pulse_type,
                "rolloff": self.rolloff,
                "span": self.span,
            },
        )

    def _build_ppm_waveform_tensor(
        self,
        pulse_positions: np.ndarray,
        tx_pulse: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Build the continuous-time PPM waveform tensor from pulse positions.

        Parameters
        ----------
        pulse_positions : np.ndarray
            Shape:
                (symbols, periods, sensors)

        tx_pulse : np.ndarray
            1D transmit pulse.

        t : np.ndarray
            Time grid for one period.

        Returns
        -------
        np.ndarray
            Waveform tensor with shape:
                (time, periods, sensors)
        """

        t = np.asarray(t, dtype=float)
        tx_pulse = np.asarray(tx_pulse, dtype=float).reshape(-1)

        n_symbols, n_periods, n_sensors = pulse_positions.shape
        n_time = len(t)

        y = np.zeros((n_time, n_periods, n_sensors), dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                waveform = np.zeros(n_time, dtype=float)

                for k in range(n_symbols):
                    t_pulse = pulse_positions[k, p, s]

                    # Place a discrete impulse at the closest time-grid index
                    idx = int(np.argmin(np.abs(t - t_pulse)))

                    temp = np.zeros(n_time, dtype=float)
                    temp[idx] = 1.0

                    waveform += np.convolve(temp, tx_pulse, mode="same")

                y[:, p, s] = waveform

        return y

    # =========================================================================
    # DEMODULATION
    # =========================================================================

    def demodulate(
        self,
        y: np.ndarray,
        t: np.ndarray,
        modulation_result: Optional[ModulationResult] = None,
        **kwargs
    ) -> DemodulationResult:
        """
        Demodulate a received PPM waveform tensor.

        Parameters
        ----------
        y : np.ndarray
            Received waveform tensor.
            Supported shapes:
                (time,)
                (time, periods)
                (time, periods, sensors)

        t : np.ndarray
            Time grid for one period.

        modulation_result : ModulationResult, optional
            Optional transmitter-side metadata.
            If supplied, it is used to:
            - recover symbol_times
            - recover pulse-shaping parameters
            - invert normalization
            - optionally reconstruct the continuous-time signal

        Keyword arguments
        -----------------
        reconstruct_continuous : bool, optional
            If True and modulation_result is available, also reconstruct the
            continuous-time signal. Default is True.

        Returns
        -------
        DemodulationResult
            Structured demodulation outputs.
        """

        reconstruct_cont = kwargs.get("reconstruct_continuous", True)

        y = ensure_3d_signal_tensor(y).astype(float, copy=False)
        t = ensure_1d_time_vector(t)
        validate_time_axis_length(y, t)

        dt = _infer_dt(t)
        tau = _infer_tau(t, dt)

        # Obtain pulse/matched filter and symbol times
        if modulation_result is not None and modulation_result.aux.get("tx_pulse") is not None:
            tx_pulse = np.asarray(modulation_result.aux["tx_pulse"], dtype=float)
        else:
            tx_pulse = build_tx_pulse(
                pulse_type=self.pulse_type,
                pulse_width=self.pulse_width,
                dt=dt,
                rolloff=self.rolloff,
                span=self.span,
                normalize_energy=True,
            )

        matched_filter = build_matched_filter(tx_pulse)

        if modulation_result is not None and modulation_result.symbol_times is not None:
            symbol_times = np.asarray(modulation_result.symbol_times, dtype=float)
        else:
            symbol_times = np.arange(0.0, tau, self.Tc, dtype=float)

        n_symbols = len(symbol_times)
        _, n_periods, n_sensors = y.shape

        matched_filter_output = np.zeros_like(y, dtype=float)
        recovered_norm = np.zeros((n_symbols, n_periods, n_sensors), dtype=float)
        detected_peak_times = np.zeros((n_symbols, n_periods, n_sensors), dtype=float)

        displacement_scale = self.Tc - (1.0 + self.rec_pulse) * self.pulse_width
        if displacement_scale <= 0:
            raise ValueError(
                "Invalid PPM geometry: Tc - (1 + rec_pulse) * pulse_width must be positive."
            )

        for p in range(n_periods):
            for s in range(n_sensors):
                z = np.convolve(y[:, p, s], matched_filter, mode="same")
                matched_filter_output[:, p, s] = z

                for k, t0 in enumerate(symbol_times):
                    idx_seg = (t >= t0) & (t < t0 + self.Tc)

                    if not np.any(idx_seg):
                        raise ValueError(
                            f"Empty symbol interval during demodulation for symbol k={k}."
                        )

                    z_seg = z[idx_seg]
                    t_seg = t[idx_seg]

                    imax = int(np.argmax(z_seg))
                    t_peak = t_seg[imax]
                    detected_peak_times[k, p, s] = t_peak

                    xk = (t_peak - t0 - self.pulse_width / 2.0) / displacement_scale

                    if self.clip_recovered_to_unit_interval:
                        xk = np.clip(
                            xk,
                            self.eps_margin,
                            1.0 - self.eps_margin
                        )

                    recovered_norm[k, p, s] = xk

        # Denormalize recovered samples if possible
        if modulation_result is not None:
            recovered_samples = self.denormalize_message(
                recovered_norm,
                modulation_result.normalization_state
            )
            norm_state = modulation_result.normalization_state
        else:
            recovered_samples = recovered_norm
            norm_state = None

        recovered_continuous = None
        if reconstruct_cont:
            recovered_continuous = self.reconstruct_continuous(
                recovered_samples=recovered_samples,
                t=t,
                modulation_result=modulation_result
            )

        return DemodulationResult(
            recovered_samples=recovered_samples,
            recovered_continuous=recovered_continuous,
            normalization_state=norm_state,
            aux={
                "recovered_normalized_samples": recovered_norm,
                "matched_filter_output": matched_filter_output,
                "matched_filter": matched_filter,
                "detected_peak_times": detected_peak_times,
                "symbol_times": symbol_times,
                "dt": dt,
                "tau": tau,
                "Tc": self.Tc,
            },
        )

    # =========================================================================
    # CONTINUOUS-TIME RECONSTRUCTION
    # =========================================================================

    def reconstruct_continuous(
        self,
        recovered_samples: np.ndarray,
        t: np.ndarray,
        modulation_result: Optional[ModulationResult] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Reconstruct a continuous-time waveform from recovered symbol-domain samples
        using periodic sinc reconstruction.

        Parameters
        ----------
        recovered_samples : np.ndarray
            Denormalized recovered symbol-domain samples with shape:
                (symbols, periods, sensors)

        t : np.ndarray
            Target time grid for one period.

        modulation_result : ModulationResult, optional
            Used to recover symbol_times if available.

        Returns
        -------
        np.ndarray
            Continuous-time reconstruction with shape:
                (time, periods, sensors)

        Notes
        -----
        This follows the same conceptual reconstruction as the MATLAB reference,
        but in a simpler and more reusable tensor form:

            x_rec(t) ≈ sum_m sum_k x[k] sinc((t - (t_k + m*tau)) / Tc)

        where m spans a finite number of period replicas.
        """

        _ = kwargs

        recovered_samples = np.asarray(recovered_samples, dtype=float)
        if recovered_samples.ndim != 3:
            raise ValueError(
                "recovered_samples must have shape (symbols, periods, sensors)."
            )

        t = ensure_1d_time_vector(t)
        dt = _infer_dt(t)
        tau = _infer_tau(t, dt)

        if modulation_result is not None and modulation_result.symbol_times is not None:
            symbol_times = np.asarray(modulation_result.symbol_times, dtype=float)
        else:
            symbol_times = np.arange(0.0, tau, self.Tc, dtype=float)

        n_symbols, n_periods, n_sensors = recovered_samples.shape
        if len(symbol_times) != n_symbols:
            raise ValueError(
                f"recovered_samples has n_symbols={n_symbols}, "
                f"but len(symbol_times)={len(symbol_times)}"
            )

        x_rec = np.zeros((len(t), n_periods, n_sensors), dtype=float)

        replica_ids = np.arange(
            -self.periodic_replicas,
            self.periodic_replicas + 1,
            dtype=int
        )

        for p in range(n_periods):
            for s in range(n_sensors):
                y = np.zeros(len(t), dtype=float)

                for m in replica_ids:
                    shifted_times = symbol_times + m * tau

                    for k in range(n_symbols):
                        y += recovered_samples[k, p, s] * np.sinc(
                            (t - shifted_times[k]) / self.Tc
                        )

                x_rec[:, p, s] = y

        return x_rec


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _infer_dt(t: np.ndarray) -> float:
    """
    Infer the time step from a 1D grid.
    """

    if len(t) < 2:
        raise ValueError("Time vector must contain at least two samples.")

    dt = float(t[1] - t[0])
    if dt <= 0:
        raise ValueError("Time vector must be strictly increasing.")

    return dt


def _infer_tau(t: np.ndarray, dt: float) -> float:
    """
    Infer the signal period from a one-period grid.

    If the grid is:
        t = [0, dt, 2dt, ..., tau-dt]

    then:
        tau = t[-1] + dt
    """

    return float(t[-1] + dt)


__all__ = [
    "PPMCore",
]
