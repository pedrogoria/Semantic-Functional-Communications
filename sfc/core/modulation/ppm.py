"""
sfc/core/modulation/ppm.py

Pulse-Position Modulation (PPM) core.

 implementation supports multiple sensors.Purpose

2. Period-aware:
   The implementation supports multiple periods.

3. MAC-agnostic:
   This module generates one waveform per sensor and per period, but it does
   NOT decide how multiple sensors share the medium. MAC-layer combination
   belongs elsewhere.

4. Pulse-shaping ready:
   The transmit pulse is obtained through:

       sfc.core.modulation.pulse_shaping

5. Core reconstruction rule:
   Continuous-time reconstruction from recovered symbol-domain samples must use:

       sfc.core.filters.sinc_reconstruct_from_samples

   This avoids local sinc implementations inside the PPM modem.

Tensor conventions
------------------
Continuous-time signal tensors use:

    (time, periods, sensors)

Symbol-domain quantities use:

    (symbols, periods, sensors)

Notes
-----
- This module does NOT add AWGN/noise by itself.
- This module does NOT perform plots/PSD/debug visualizations.
- This module does NOT implement MAC/resource sharing.
- Physical AWGN must be handled outside this module, e.g. through:
      sfc.core.channel.physical_channel.apply_awgn
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from sfc.core.filters import sinc_reconstruct_from_samples
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
        Recovery / guard parameter.

    pulse_type : str, optional
        Pulse-shaping type.

    rolloff : float, optional
        Raised-cosine rolloff factor.

    span : int, optional
        Raised-cosine span in symbols.

    eps_margin : float, optional
        Margin used when normalizing the message into:

            [eps_margin, 1 - eps_margin]

    interp_mode : str, optional
        Continuous-time reconstruction mode.

        Current supported value:
            "sinc"

        The sinc reconstruction is delegated to:

            sfc.core.filters.sinc_reconstruct_from_samples

    periodic_replicas : int, optional
        Number of period replicas used during periodic sinc reconstruction.

    clip_recovered_to_unit_interval : bool, optional
        Whether to clip recovered normalized samples into:

            [eps_margin, 1 - eps_margin]
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
            interp_mode: str = "sinc",
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

        if periodic_replicas < 0:
            raise ValueError("periodic_replicas must be nonnegative.")

        self.fc = float(fc)
        self.Tc = 1.0 / self.fc
        self.pulse_width = float(pulse_width)
        self.rec_pulse = float(rec_pulse)

        if self.Tc - (1.0 + self.rec_pulse) * self.pulse_width <= 0:
            raise ValueError(
                "Invalid PPM geometry: "
                "Tc - (1 + rec_pulse) * pulse_width must be positive."
            )

        self.pulse_type = str(pulse_type)
        self.rolloff = float(rolloff)
        self.span = int(span)

        self.eps_margin = float(eps_margin)
        self.interp_mode = str(interp_mode).lower()
        self.periodic_replicas = int(periodic_replicas)
        self.clip_recovered_to_unit_interval = bool(clip_recovered_to_unit_interval)

        if self.interp_mode not in {"sinc"}:
            raise ValueError(
                "Unsupported interp_mode for PPMCore. "
                "Currently supported: 'sinc'."
            )

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def normalize_message(
            self,
            x: np.ndarray,
            **kwargs
    ) -> Tuple[np.ndarray, NormalizationState]:
        """
        Normalize each (period, sensor) waveform into:

            [eps_margin, 1 - eps_margin]
        """
        _ = kwargs

        x = ensure_3d_signal_tensor(x).astype(float, copy=False)

        _, n_periods, n_sensors = x.shape

        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        denom = x_max - x_min

        x_norm = np.empty_like(x, dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                if np.isclose(denom[p, s], 0.0):
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
        """
        _ = kwargs

        if normalization_state is None or not normalization_state.enabled:
            return x

        x = np.asarray(x, dtype=float)
        x_min = np.asarray(normalization_state.metadata["x_min"], dtype=float)
        x_max = np.asarray(normalization_state.metadata["x_max"], dtype=float)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the normalized continuous-time message at the symbol rate fc.

        The number of PPM symbols per period is:

            n_symbols = len(np.arange(0.0, tau, Tc))

        where:

            Tc = 1 / fc.

        Therefore, approximately:

            n_symbols ≈ tau * fc.
        """
        _ = kwargs

        x = ensure_3d_signal_tensor(x).astype(float, copy=False)
        t = ensure_1d_time_vector(t)
        validate_time_axis_length(x, t)

        dt = _infer_dt(t)
        tau = _infer_tau(t, dt)

        symbol_times = np.arange(0.0, tau, self.Tc, dtype=float)

        # Numerical guard:
        # np.arange(0, tau, Tc) should already stop before tau. This additional
        # guard protects against floating-point edge effects.
        symbol_times = symbol_times[symbol_times < tau + 0.5 * dt]

        n_symbols = len(symbol_times)
        _, n_periods, n_sensors = x.shape

        x_sampled = np.zeros((n_symbols, n_periods, n_sensors), dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                x_sampled[:, p, s] = np.interp(
                    symbol_times,
                    t,
                    x[:, p, s],
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
                "Invalid PPM geometry: "
                "Tc - (1 + rec_pulse) * pulse_width must be positive."
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
        """
        _ = kwargs

        x = ensure_3d_signal_tensor(x).astype(float, copy=False)
        t = ensure_1d_time_vector(t)
        validate_time_axis_length(x, t)

        dt = _infer_dt(t)
        tau = _infer_tau(t, dt)

        x_norm, norm_state = self.normalize_message(x)
        x_sampled, symbol_times = self.sample_message(x_norm, t)
        pulse_positions = self.compute_pulse_positions(x_sampled, symbol_times)

        tx_pulse = build_tx_pulse(
            pulse_type=self.pulse_type,
            pulse_width=self.pulse_width,
            dt=dt,
            rolloff=self.rolloff,
            span=self.span,
            normalize_energy=True,
        )

        tx_waveform = self._build_ppm_waveform_tensor(
            pulse_positions=pulse_positions,
            tx_pulse=tx_pulse,
            t=t,
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

                    idx = int(np.argmin(np.abs(t - t_pulse)))

                    temp = np.zeros(n_time, dtype=float)
                    temp[idx] = 1.0

                    waveform += self._convolve_same_length(temp, tx_pulse)

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

        Robustness note
        ---------------
        If a symbol interval is empty because of floating-point/grid mismatch
        at the edge of the period, the demodulator falls back to the nearest
        available time-grid sample instead of failing immediately.

        Complex input convention
        ------------------------
        PPMCore is a real-valued pulse-position modem. If the received waveform
        is complex-valued because an upstream channel produced complex samples,
        the demodulator uses the real part before matched filtering.

        In the current fair-comparison pipelines, apply_awgn(...) normally adds
        real AWGN to real PPM waveforms, so this conversion is mostly a safety
        guard.
        """
        reconstruct_cont = kwargs.get("reconstruct_continuous", True)

        y_arr = np.asarray(y)

        if np.iscomplexobj(y_arr):
            y_arr = np.real(y_arr)

        y = ensure_3d_signal_tensor(y_arr).astype(float, copy=False)
        t = ensure_1d_time_vector(t)
        validate_time_axis_length(y, t)

        dt = _infer_dt(t)
        tau = _infer_tau(t, dt)

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
            symbol_times = symbol_times[symbol_times < tau + 0.5 * dt]

        n_symbols = len(symbol_times)
        _, n_periods, n_sensors = y.shape

        matched_filter_output = np.zeros_like(y, dtype=float)
        recovered_norm = np.zeros((n_symbols, n_periods, n_sensors), dtype=float)
        detected_peak_times = np.zeros((n_symbols, n_periods, n_sensors), dtype=float)

        displacement_scale = self.Tc - (1.0 + self.rec_pulse) * self.pulse_width
        if displacement_scale <= 0:
            raise ValueError(
                "Invalid PPM geometry: "
                "Tc - (1 + rec_pulse) * pulse_width must be positive."
            )

        for p in range(n_periods):
            for s in range(n_sensors):
                z = self._convolve_same_length(y[:, p, s], matched_filter)
                matched_filter_output[:, p, s] = z

                for k, t0 in enumerate(symbol_times):
                    idx_seg = (t >= t0) & (t < t0 + self.Tc)

                    if not np.any(idx_seg):
                        idx_nearest = int(np.argmin(np.abs(t - t0)))
                        idx_nearest = max(0, min(idx_nearest, len(t) - 1))

                        idx_seg = np.zeros_like(t, dtype=bool)
                        idx_seg[idx_nearest] = True

                    z_seg = z[idx_seg]
                    t_seg = t[idx_seg]

                    if z_seg.size == 0:
                        raise ValueError(
                            f"Empty symbol interval during demodulation for symbol k={k}."
                        )

                    imax = int(np.argmax(z_seg))
                    t_peak = t_seg[imax]
                    detected_peak_times[k, p, s] = t_peak

                    xk = (t_peak - t0 - self.pulse_width / 2.0) / displacement_scale

                    if self.clip_recovered_to_unit_interval:
                        xk = np.clip(
                            xk,
                            self.eps_margin,
                            1.0 - self.eps_margin,
                        )

                    recovered_norm[k, p, s] = xk

        if modulation_result is not None:
            recovered_samples = self.denormalize_message(
                recovered_norm,
                modulation_result.normalization_state,
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
                modulation_result=modulation_result,
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

    @staticmethod
    def _convolve_same_length(x, h):
        """
        Convolve x with h and return an output with exactly len(x).

        np.convolve(x, h, mode="same") returns max(len(x), len(h)).
        This is not safe when the pulse h is longer than the waveform x.

        This helper always returns a center-cropped convolution with length len(x).
        """
        x = np.asarray(x)
        h = np.asarray(h)

        if x.ndim != 1 or h.ndim != 1:
            raise ValueError("_convolve_same_length expects 1-D arrays.")

        if len(x) == 0:
            return np.asarray([], dtype=np.result_type(x, h))

        if len(h) == 0:
            return np.zeros_like(x)

        y_full = np.convolve(x, h, mode="full")

        start = (len(y_full) - len(x)) // 2
        stop = start + len(x)

        y = y_full[start:stop]

        if len(y) != len(x):
            raise RuntimeError(
                "Internal convolution cropping error: "
                f"len(x)={len(x)}, len(h)={len(h)}, "
                f"len(y_full)={len(y_full)}, len(y)={len(y)}."
            )

        return y

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
        Reconstruct a continuous-time waveform from recovered symbol-domain
        samples using the project-wide core sinc reconstruction.

        This function intentionally delegates sample-to-continuous sinc
        interpolation to:

            sfc.core.filters.sinc_reconstruct_from_samples

        Therefore, this module does not implement a local sinc loop.

        Parameters
        ----------
        recovered_samples : np.ndarray
            Symbol-domain recovered samples with shape:

                (symbols, periods, sensors)

        t : np.ndarray
            Dense time grid.

        modulation_result : ModulationResult or None
            If provided, symbol_times are read from modulation_result.
            Otherwise, symbol_times are inferred from fc and the dense grid.

        Returns
        -------
        np.ndarray
            Reconstructed dense-time signal with shape:

                (time, periods, sensors)
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
            symbol_times = symbol_times[symbol_times < tau + 0.5 * dt]

        n_symbols, _, _ = recovered_samples.shape

        if len(symbol_times) != n_symbols:
            raise ValueError(
                f"recovered_samples has n_symbols={n_symbols}, "
                f"but len(symbol_times)={len(symbol_times)}"
            )

        return sinc_reconstruct_from_samples(
            x_samples=recovered_samples,
            t_samples=symbol_times,
            t_eval=t,
            tau=tau,
            sample_period=self.Tc,
            periodic_replicas=self.periodic_replicas,
        )


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
