"""
sfc/core/acquisition/fri.py

Finite Rate of Innovation (FRI)-inspired acquisition baseline.

Purpose
-------
This module provides a simulation-oriented FRI baseline for comparison against
RbCP/SFC, Nyquist, CS, and other acquisition methods.

Important scope note
--------------------
This is NOT intended to be a full theorem-level implementation of classical
annihilating-filter FRI reconstruction. Instead, it is a practical and explicit
FRI-inspired sparse-innovation acquisition model for repository simulations.

The model assumes that a signal over one period can be represented, or
approximated, by K localized innovations:

    x_hat(t) = sum_k a_k phi(t - tau_k)

where tau_k are innovation locations, a_k are amplitudes, and phi is a chosen
reconstruction kernel. The module estimates innovations from a dense sampled
signal, optionally quantizes innovation locations/amplitudes under a bit budget,
optionally applies a binary-symmetric bit-error model, and reconstructs the
signal on the original time grid.

Design goals
------------
- Provide a fair simulation baseline requested by reviewers.
- Keep the baseline explicit and reproducible.
- Keep channel/bit-budget effects configurable.
- Avoid changing existing RbCP/SFC/Nyquist/CS modules.
- Use only NumPy for portability.

Supported input shapes
----------------------
The public methods accept signal arrays with shapes:

    (time,)
    (time, periods)
    (time, periods, sensors)

Internally, signals are converted to the canonical shape:

    (time, periods, sensors)

and converted back if requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class FRIResult:
    """
    Result returned by FRIAcquisition.run(...).

    Attributes
    ----------
    x_hat : np.ndarray
        Reconstructed signal with the same shape convention as the input signal.

    innovations : dict
        Dictionary containing innovation locations, amplitudes, and quantization
        metadata. Arrays use canonical shape:

            locations.shape  = (K, periods, sensors)
            amplitudes.shape = (K, periods, sensors)

    diagnostics : dict
        Diagnostic quantities for reproducibility and debugging.
    """

    x_hat: np.ndarray
    innovations: dict[str, Any]
    diagnostics: dict[str, Any]


# =============================================================================
# FRI ACQUISITION BASELINE
# =============================================================================

class FRIAcquisition:
    """
    FRI-inspired sparse-innovation acquisition baseline.

    Configuration
    -------------
    The module reads optional configuration from:

        cfg["acquisition"]["fri"]

    Supported YAML keys
    -------------------
    acquisition:
      fri:
        K: 3
        min_separation: 0.0
        kernel: gaussian          # gaussian | sinc | triangular | nearest
        kernel_sigma: null        # if null, defaults to 2*Tt
        sinc_bandwidth: null      # if null, defaults to 1 / max(Tt, eps)
        periodic: true

        quantize: true
        budget_bits: null         # if null, uses bits_location + bits_amplitude
        bits_location: 10
        bits_amplitude: 10
        amplitude_range: sample   # sample | peak_to_peak | [min, max]
        peak_to_peak: null

        bit_error_rate: 0.0
        seed: null

    Notes
    -----
    This class intentionally exposes both low-level and high-level methods:

    - estimate_innovations(...)
    - quantize_innovations(...)
    - reconstruct(...)
    - run(...)

    so pipelines can either use the complete baseline or inspect intermediate
    quantities.
    """

    def __init__(self, cfg: dict | None = None, **overrides):
        """
        Initialize the FRI acquisition baseline.

        Parameters
        ----------
        cfg : dict or None
            Project configuration dictionary.

        **overrides : dict
            Explicit parameter overrides. These take precedence over YAML.
        """
        self.cfg = cfg if cfg is not None else {}

        acq_cfg = self.cfg.get("acquisition", {})
        fri_cfg = dict(acq_cfg.get("fri", {}))
        fri_cfg.update(overrides)

        signal_cfg = self.cfg.get("signal", {})
        system_cfg = self.cfg.get("system", {})

        self.K = int(fri_cfg.get("K", fri_cfg.get("num_innovations", 3)))
        self.min_separation = float(fri_cfg.get("min_separation", 0.0))

        self.kernel = str(fri_cfg.get("kernel", "gaussian")).lower()
        self.kernel_sigma = fri_cfg.get("kernel_sigma", None)
        self.sinc_bandwidth = fri_cfg.get("sinc_bandwidth", None)
        self.periodic = bool(fri_cfg.get("periodic", True))

        self.quantize_enabled = bool(fri_cfg.get("quantize", True))
        self.budget_bits = fri_cfg.get("budget_bits", None)
        self.bits_location = int(fri_cfg.get("bits_location", 10))
        self.bits_amplitude = int(fri_cfg.get("bits_amplitude", 10))
        self.amplitude_range = fri_cfg.get("amplitude_range", "sample")
        self.peak_to_peak = fri_cfg.get(
            "peak_to_peak",
            signal_cfg.get("peak_to_peak", None),
        )

        self.bit_error_rate = float(fri_cfg.get("bit_error_rate", 0.0))

        self.seed = fri_cfg.get(
            "seed",
            self.cfg.get("reproducibility", {}).get(
                "seed",
                self.cfg.get("monte_carlo", {}).get("seed", None),
            ),
        )

        self.default_tau = fri_cfg.get(
            "tau",
            system_cfg.get("tau", signal_cfg.get("tau", None)),
        )

        self.default_Tt = fri_cfg.get("Tt", signal_cfg.get("Tt", None))

        self._validate_config()

        self.rng = (
            np.random.default_rng()
            if self.seed is None
            else np.random.default_rng(int(self.seed))
        )

        self.last_diagnostics: dict[str, Any] = {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run(
        self,
        x,
        t=None,
        *,
        Tt=None,
        tau=None,
        budget_bits=None,
        bit_error_rate=None,
        return_result=True,
    ):
        """
        Estimate, optionally quantize, and reconstruct an FRI approximation.

        Parameters
        ----------
        x : np.ndarray
            Input signal with shape (time,), (time, periods), or
            (time, periods, sensors).

        t : np.ndarray or None
            Time grid. If None, it is built from Tt and tau.

        Tt : float or None
            Time step. Required if t is None and not available in cfg.

        tau : float or None
            Period/window duration. Required if t is None and not available in
            cfg.

        budget_bits : int or None
            Optional total bits per signal/period/sensor. If provided, bits are
            split between innovation locations and amplitudes.

        bit_error_rate : float or None
            Optional override for the configured bit error rate.

        return_result : bool
            If True, return FRIResult. If False, return x_hat only.

        Returns
        -------
        FRIResult or np.ndarray
            FRIResult if return_result=True, otherwise x_hat.
        """
        x3, original_ndim = self._as_3d(x)
        t = self._resolve_time_grid(x3.shape[0], t=t, Tt=Tt, tau=tau)
        tau = float(t[-1] + self._infer_Tt(t)) if tau is None else float(tau)
        Tt = self._infer_Tt(t) if Tt is None else float(Tt)

        innovations = self.estimate_innovations(x3, t=t, tau=tau)

        if self.quantize_enabled:
            innovations = self.quantize_innovations(
                innovations=innovations,
                x=x3,
                t=t,
                tau=tau,
                budget_bits=budget_bits,
                bit_error_rate=bit_error_rate,
            )

        x_hat3 = self.reconstruct(
            innovations=innovations,
            t=t,
            tau=tau,
            Tt=Tt,
        )

        x_hat = self._restore_shape(x_hat3, original_ndim)

        diagnostics = self._build_diagnostics(
            x=x3,
            x_hat=x_hat3,
            t=t,
            tau=tau,
            Tt=Tt,
            innovations=innovations,
        )

        self.last_diagnostics = diagnostics

        if return_result:
            return FRIResult(
                x_hat=x_hat,
                innovations=innovations,
                diagnostics=diagnostics,
            )

        return x_hat

    def __call__(self, *args, **kwargs):
        """
        Alias for run(...).
        """
        return self.run(*args, **kwargs)

    # =========================================================================
    # INNOVATION ESTIMATION
    # =========================================================================

    def estimate_innovations(self, x, *, t, tau=None):
        """
        Estimate K sparse innovations using greedy peak selection.

        Parameters
        ----------
        x : np.ndarray
            Canonical input signal with shape (time, periods, sensors).

        t : np.ndarray
            Time grid.

        tau : float or None
            Period duration. If None, inferred from t.

        Returns
        -------
        dict
            Innovation dictionary with locations and amplitudes.
        """
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)

        if x.ndim != 3:
            raise ValueError("x must have shape (time, periods, sensors).")

        if t.ndim != 1 or len(t) != x.shape[0]:
            raise ValueError("t must be a 1D array with len(t) == x.shape[0].")

        tau = float(t[-1] + self._infer_Tt(t)) if tau is None else float(tau)

        n_time, n_periods, n_sensors = x.shape
        K_eff = min(self.K, n_time)

        locations = np.zeros((K_eff, n_periods, n_sensors), dtype=float)
        amplitudes = np.zeros((K_eff, n_periods, n_sensors), dtype=float)
        indices = np.zeros((K_eff, n_periods, n_sensors), dtype=int)

        min_sep_samples = self._min_separation_samples(t)

        for p in range(n_periods):
            for s in range(n_sensors):
                sig = x[:, p, s]
                selected = self._select_peak_indices(
                    sig,
                    K=K_eff,
                    min_sep_samples=min_sep_samples,
                )

                selected = self._pad_indices(selected, K_eff, n_time)

                indices[:, p, s] = selected
                locations[:, p, s] = t[selected]
                amplitudes[:, p, s] = sig[selected]

        return {
            "locations": locations,
            "amplitudes": amplitudes,
            "indices": indices,
            "K": K_eff,
            "tau": tau,
            "quantized": False,
            "bits_location": None,
            "bits_amplitude": None,
            "bit_error_rate": 0.0,
        }

    # =========================================================================
    # QUANTIZATION AND BIT ERRORS
    # =========================================================================

    def quantize_innovations(
        self,
        *,
        innovations,
        x,
        t,
        tau,
        budget_bits=None,
        bit_error_rate=None,
    ):
        """
        Quantize innovation locations and amplitudes.

        Locations are uniformly quantized over [0, tau). Amplitudes are
        uniformly quantized over the configured amplitude range.
        """
        locations = np.asarray(innovations["locations"], dtype=float)
        amplitudes = np.asarray(innovations["amplitudes"], dtype=float)

        bits_location, bits_amplitude = self._resolve_bit_allocation(
            budget_bits=budget_bits,
        )

        ber = self.bit_error_rate if bit_error_rate is None else float(bit_error_rate)
        if ber < 0 or ber > 1:
            raise ValueError("bit_error_rate must be in [0, 1].")

        loc_levels = max(2 ** bits_location, 1)
        amp_levels = max(2 ** bits_amplitude, 1)

        loc_indices = self._quantize_to_indices(
            values=np.mod(locations, tau),
            vmin=0.0,
            vmax=float(tau),
            levels=loc_levels,
        )

        amp_min, amp_max = self._amplitude_bounds(x)
        amp_indices = self._quantize_to_indices(
            values=amplitudes,
            vmin=amp_min,
            vmax=amp_max,
            levels=amp_levels,
        )

        if ber > 0.0:
            loc_indices = self._apply_bsc_to_indices(
                loc_indices,
                bits=bits_location,
                ber=ber,
            )
            amp_indices = self._apply_bsc_to_indices(
                amp_indices,
                bits=bits_amplitude,
                ber=ber,
            )

        q_locations = self._dequantize_from_indices(
            indices=loc_indices,
            vmin=0.0,
            vmax=float(tau),
            levels=loc_levels,
        )

        q_amplitudes = self._dequantize_from_indices(
            indices=amp_indices,
            vmin=amp_min,
            vmax=amp_max,
            levels=amp_levels,
        )

        q_locations = np.mod(q_locations, tau)

        out = dict(innovations)
        out.update(
            {
                "locations": q_locations,
                "amplitudes": q_amplitudes,
                "location_indices": loc_indices,
                "amplitude_indices": amp_indices,
                "amplitude_min": amp_min,
                "amplitude_max": amp_max,
                "quantized": True,
                "bits_location": bits_location,
                "bits_amplitude": bits_amplitude,
                "bit_error_rate": ber,
            }
        )

        return out

    # =========================================================================
    # RECONSTRUCTION
    # =========================================================================

    def reconstruct(self, *, innovations, t, tau, Tt=None):
        """
        Reconstruct signal from sparse innovations on the provided grid.
        """
        locations = np.asarray(innovations["locations"], dtype=float)
        amplitudes = np.asarray(innovations["amplitudes"], dtype=float)
        t = np.asarray(t, dtype=float)

        if locations.shape != amplitudes.shape:
            raise ValueError("locations and amplitudes must have matching shapes.")

        if locations.ndim != 3:
            raise ValueError("locations must have shape (K, periods, sensors).")

        K, n_periods, n_sensors = locations.shape
        n_time = len(t)

        Tt = self._infer_Tt(t) if Tt is None else float(Tt)
        tau = float(tau)

        x_hat = np.zeros((n_time, n_periods, n_sensors), dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                acc = np.zeros(n_time, dtype=float)

                for k in range(K):
                    dt = self._time_difference(
                        t,
                        locations[k, p, s],
                        tau=tau,
                    )
                    kernel_values = self._kernel_values(dt, Tt=Tt)
                    acc += amplitudes[k, p, s] * kernel_values

                x_hat[:, p, s] = acc

        return x_hat

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def diagnostics(self):
        """
        Return last-run diagnostics.
        """
        return dict(self.last_diagnostics)

    # =========================================================================
    # INTERNAL HELPERS: CONFIG AND SHAPES
    # =========================================================================

    def _validate_config(self):
        if self.K < 1:
            raise ValueError("FRIAcquisition requires K >= 1.")

        if self.min_separation < 0:
            raise ValueError("min_separation must be nonnegative.")

        valid_kernels = {"gaussian", "sinc", "triangular", "nearest"}
        if self.kernel not in valid_kernels:
            raise ValueError(
                f"Invalid FRI kernel: {self.kernel}. "
                f"Expected one of {sorted(valid_kernels)}."
            )

        if self.bits_location < 0 or self.bits_amplitude < 0:
            raise ValueError("bits_location and bits_amplitude must be >= 0.")

        if self.budget_bits is not None and int(self.budget_bits) < 0:
            raise ValueError("budget_bits must be >= 0 when provided.")

        if self.bit_error_rate < 0 or self.bit_error_rate > 1:
            raise ValueError("bit_error_rate must be in [0, 1].")

    @staticmethod
    def _as_3d(x):
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            return x[:, None, None], 1

        if x.ndim == 2:
            return x[:, :, None], 2

        if x.ndim == 3:
            return x, 3

        raise ValueError("x must have shape (time,), (time, periods), or (time, periods, sensors).")

    @staticmethod
    def _restore_shape(x3, original_ndim):
        if original_ndim == 1:
            return x3[:, 0, 0]

        if original_ndim == 2:
            return x3[:, :, 0]

        return x3

    def _resolve_time_grid(self, n_time, *, t=None, Tt=None, tau=None):
        if t is not None:
            t = np.asarray(t, dtype=float)
            if t.ndim != 1 or len(t) != n_time:
                raise ValueError("t must be a 1D array with length equal to x.shape[0].")
            return t

        if Tt is None:
            if self.default_Tt is None:
                raise ValueError("Tt must be provided if t is None and cfg['signal']['Tt'] is unavailable.")
            Tt = float(self.default_Tt)

        if tau is None:
            if self.default_tau is None:
                tau = n_time * float(Tt)
            else:
                tau = float(self.default_tau)

        return np.arange(0.0, float(tau), float(Tt))[:n_time]

    @staticmethod
    def _infer_Tt(t):
        t = np.asarray(t, dtype=float)
        if len(t) < 2:
            raise ValueError("t must contain at least two samples to infer Tt.")
        return float(t[1] - t[0])

    # =========================================================================
    # INTERNAL HELPERS: PEAK SELECTION
    # =========================================================================

    def _min_separation_samples(self, t):
        if self.min_separation <= 0:
            return 0
        Tt = self._infer_Tt(t)
        return int(np.ceil(self.min_separation / Tt))

    @staticmethod
    def _pad_indices(indices, K, n_time):
        indices = list(indices)

        if len(indices) == 0:
            indices = [0]

        while len(indices) < K:
            indices.append(indices[-1])

        indices = np.asarray(indices[:K], dtype=int)
        return np.clip(indices, 0, n_time - 1)

    @staticmethod
    def _select_peak_indices(sig, *, K, min_sep_samples):
        score = np.abs(np.asarray(sig, dtype=float))
        order = np.argsort(score)[::-1]

        selected = []

        for idx in order:
            idx = int(idx)
            if not np.isfinite(score[idx]):
                continue

            if min_sep_samples > 0:
                too_close = any(abs(idx - j) < min_sep_samples for j in selected)
                if too_close:
                    continue

            selected.append(idx)

            if len(selected) >= K:
                break

        selected.sort()
        return selected

    # =========================================================================
    # INTERNAL HELPERS: QUANTIZATION
    # =========================================================================

    def _resolve_bit_allocation(self, *, budget_bits=None):
        if budget_bits is None:
            budget_bits = self.budget_bits

        if budget_bits is None:
            return self.bits_location, self.bits_amplitude

        budget_bits = int(budget_bits)
        if budget_bits < 0:
            raise ValueError("budget_bits must be nonnegative.")

        if self.K <= 0:
            raise ValueError("K must be positive.")

        bits_per_innovation = budget_bits // self.K
        bits_location = bits_per_innovation // 2
        bits_amplitude = bits_per_innovation - bits_location

        return int(bits_location), int(bits_amplitude)

    def _amplitude_bounds(self, x):
        if isinstance(self.amplitude_range, (list, tuple, np.ndarray)):
            bounds = np.asarray(self.amplitude_range, dtype=float).reshape(-1)
            if len(bounds) != 2:
                raise ValueError("amplitude_range list/tuple must have length 2.")
            amin, amax = float(bounds[0]), float(bounds[1])
            if not amax > amin:
                raise ValueError("amplitude_range max must be greater than min.")
            return amin, amax

        if self.amplitude_range == "peak_to_peak":
            if self.peak_to_peak is None:
                raise ValueError("peak_to_peak must be configured for amplitude_range='peak_to_peak'.")
            p2p = float(self.peak_to_peak)
            return -0.5 * p2p, 0.5 * p2p

        if self.amplitude_range == "sample":
            amin = float(np.nanmin(x))
            amax = float(np.nanmax(x))
            if np.isclose(amax, amin):
                delta = max(abs(amax), 1.0) * 1e-12
                return amin - delta, amax + delta
            return amin, amax

        raise ValueError(
            "amplitude_range must be 'sample', 'peak_to_peak', or [min, max]."
        )

    @staticmethod
    def _quantize_to_indices(values, *, vmin, vmax, levels):
        values = np.asarray(values, dtype=float)
        levels = int(levels)

        if levels <= 1:
            return np.zeros_like(values, dtype=int)

        if not vmax > vmin:
            raise ValueError("vmax must be greater than vmin.")

        normalized = (values - vmin) / (vmax - vmin)
        indices = np.floor(normalized * levels).astype(int)
        return np.clip(indices, 0, levels - 1)

    @staticmethod
    def _dequantize_from_indices(indices, *, vmin, vmax, levels):
        indices = np.asarray(indices, dtype=int)
        levels = int(levels)

        if levels <= 1:
            return np.full_like(indices, fill_value=0.5 * (vmin + vmax), dtype=float)

        delta = (vmax - vmin) / levels
        return vmin + (indices.astype(float) + 0.5) * delta

    def _apply_bsc_to_indices(self, indices, *, bits, ber):
        indices = np.asarray(indices, dtype=int)
        bits = int(bits)

        if bits <= 0 or ber <= 0:
            return indices

        flat = indices.reshape(-1)
        out = np.zeros_like(flat)

        for i, value in enumerate(flat):
            bit_vec = self._int_to_bits(int(value), bits)
            flips = self.rng.random(bits) < ber
            bit_vec = np.logical_xor(bit_vec, flips).astype(int)
            out[i] = self._bits_to_int(bit_vec)

        max_value = 2 ** bits - 1
        out = np.clip(out, 0, max_value)
        return out.reshape(indices.shape)

    @staticmethod
    def _int_to_bits(value, bits):
        return np.array(
            [(value >> b) & 1 for b in range(bits - 1, -1, -1)],
            dtype=int,
        )

    @staticmethod
    def _bits_to_int(bit_vec):
        value = 0
        for bit in bit_vec:
            value = (value << 1) | int(bit)
        return value

    # =========================================================================
    # INTERNAL HELPERS: KERNELS
    # =========================================================================

    def _time_difference(self, t, center, *, tau):
        dt = np.asarray(t, dtype=float) - float(center)

        if self.periodic:
            dt = (dt + 0.5 * tau) % tau - 0.5 * tau

        return dt

    def _kernel_values(self, dt, *, Tt):
        if self.kernel == "gaussian":
            sigma = self.kernel_sigma
            if sigma is None:
                sigma = 2.0 * Tt
            sigma = float(sigma)
            if sigma <= 0:
                raise ValueError("kernel_sigma must be positive for gaussian kernel.")
            return np.exp(-0.5 * (dt / sigma) ** 2)

        if self.kernel == "sinc":
            bw = self.sinc_bandwidth
            if bw is None:
                bw = 1.0 / max(float(Tt), np.finfo(float).eps)
            bw = float(bw)
            if bw <= 0:
                raise ValueError("sinc_bandwidth must be positive for sinc kernel.")
            return np.sinc(bw * dt)

        if self.kernel == "triangular":
            width = self.kernel_sigma
            if width is None:
                width = 2.0 * Tt
            width = float(width)
            if width <= 0:
                raise ValueError("kernel_sigma/width must be positive for triangular kernel.")
            return np.maximum(1.0 - np.abs(dt) / width, 0.0)

        if self.kernel == "nearest":
            out = np.zeros_like(dt, dtype=float)
            out[np.argmin(np.abs(dt))] = 1.0
            return out

        raise ValueError(f"Unknown kernel: {self.kernel}")

    # =========================================================================
    # INTERNAL HELPERS: DIAGNOSTICS
    # =========================================================================

    def _build_diagnostics(self, *, x, x_hat, t, tau, Tt, innovations):
        residual = x - x_hat
        mse = float(np.mean(residual ** 2))

        return {
            "method": "FRIAcquisition",
            "K": int(innovations.get("K", self.K)),
            "kernel": self.kernel,
            "kernel_sigma": self.kernel_sigma,
            "sinc_bandwidth": self.sinc_bandwidth,
            "periodic": self.periodic,
            "quantized": bool(innovations.get("quantized", False)),
            "bits_location": innovations.get("bits_location", None),
            "bits_amplitude": innovations.get("bits_amplitude", None),
            "budget_bits": self.budget_bits,
            "bit_error_rate": innovations.get("bit_error_rate", 0.0),
            "amplitude_range": self.amplitude_range,
            "tau": float(tau),
            "Tt": float(Tt),
            "num_time_samples": int(len(t)),
            "input_shape": tuple(int(v) for v in x.shape),
            "output_shape": tuple(int(v) for v in x_hat.shape),
            "mse": mse,
        }


__all__ = [
    "FRIAcquisition",
    "FRIResult",
]
