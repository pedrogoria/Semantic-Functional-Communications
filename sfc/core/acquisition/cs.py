"""
sfc/core/acquisition/cs.py

Compressive Sensing acquisition core.

This module implements a reusable CS acquisition/reconstruction core used by
pipelines such as:

    sfc/pipelines/fair_methods_comparison_vs_B.py

Tensor convention
-----------------
Input and output signal tensors use:

    (time, periods, sensors)

Mathematical model
------------------
For each period p and sensor s, the source signal is represented as a vector:

    x[p, s] in R^N

The ideal CS measurement model is:

    y = Phi x

If a sparsifying basis Psi is used:

    x = Psi alpha

then the effective reconstruction dictionary is:

    A = Phi Psi

and the reconstruction problem is approximately:

    y = A alpha

with alpha assumed K-sparse.

The current reconstruction method is Orthogonal Matching Pursuit (OMP).

Important modeling distinction
------------------------------
There are two measurement modes.

1. Ideal real-valued measurement mode:

       y = Phi x
       y_rx = y

   In this mode, the CS measurement vector is treated as perfectly recovered.
   Higher-level pipelines may still use bits_per_measurement to decide how many
   measurements fit into a channel-capacity budget, but this core does not
   quantize y.

2. Quantized measurement mode:

       y = Phi x
       q = Q(y)
       y_tilde = Q^{-1}(q)
       y_rx = y_tilde

   In this mode, each component of y is explicitly mapped to a finite
   quantization bin. Reconstruction uses the dequantized vector y_tilde.

   This makes the interpretation of bits_per_measurement more physical:
   each measurement is represented using measurement_bits bits, i.e.
   2^measurement_bits quantization bins.

What is shared by transmitter and receiver?
-------------------------------------------
For reconstruction, transmitter and receiver are assumed to share:

- sensing matrix construction rule or seed;
- basis type;
- sparsity level or stopping rule;
- measurement quantizer definition, when quantization is enabled.

What is transmitted?
--------------------
In ideal mode:
    the model behaves as if the real-valued measurement vector y is recovered
    perfectly.

In quantized mode:
    the physically transmitted payload is represented by the integer bin
    indices q. The receiver reconstructs y_tilde from q and then runs OMP.

Notes
-----
This module does not implement channel coding, modulation, packetization, or
bit errors. It only models CS acquisition, optional measurement quantization,
and sparse reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class CSQuantizationMetadata:
    """
    Metadata describing measurement quantization.
    """

    enabled: bool
    bits: Optional[int] = None
    num_bins: Optional[int] = None
    mode: Optional[str] = None
    range_mode: Optional[str] = None
    clip: Optional[bool] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    delta: Optional[float] = None
    saturated_low: Optional[int] = None
    saturated_high: Optional[int] = None


@dataclass
class CSAcquisitionResult:
    """
    Output of CS acquisition.

    Attributes
    ----------
    measurements : np.ndarray
        Measurement tensor used by the reconstructor.

        Shape:
            (measurements, periods, sensors)

        In ideal mode, this equals measurements_real.
        In quantized mode, this equals measurements_dequantized.

    measurements_real : np.ndarray
        Real-valued CS measurements y = Phi x.

    measurements_quantized_indices : np.ndarray or None
        Integer bin indices q, if quantization is enabled.

    measurements_dequantized : np.ndarray or None
        Dequantized measurement tensor y_tilde, if quantization is enabled.

    sensing_matrix : np.ndarray
        Phi matrix.

    basis_matrix : np.ndarray
        Psi matrix.

    dictionary_matrix : np.ndarray
        A = Phi Psi.

    original_shape : tuple
        Shape of input signal tensor.

    time : np.ndarray
        Time vector.

    quantization_metadata : dict
        Metadata per period/sensor for measurement quantization.

    aux : dict
        Additional diagnostic information.
    """

    measurements: np.ndarray
    measurements_real: np.ndarray
    measurements_quantized_indices: Optional[np.ndarray]
    measurements_dequantized: Optional[np.ndarray]
    sensing_matrix: np.ndarray
    basis_matrix: np.ndarray
    dictionary_matrix: np.ndarray
    original_shape: Tuple[int, int, int]
    time: np.ndarray
    quantization_metadata: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)

    # Backward-compatible aliases used by some debug scripts.
    @property
    def y(self) -> np.ndarray:
        return self.measurements

    @property
    def compressed_signal(self) -> np.ndarray:
        return self.measurements

    @property
    def acquired_signal(self) -> np.ndarray:
        return self.measurements


@dataclass
class CSReconstructionResult:
    """
    Output of CS reconstruction.

    Attributes
    ----------
    reconstructed_signal : np.ndarray
        Reconstructed signal tensor with shape:

            (time, periods, sensors)

    recovered_representation : np.ndarray
        Estimated sparse coefficient tensor alpha_hat with shape:

            (basis_coefficients, periods, sensors)

    support : dict
        Selected support indices per period/sensor.

    residual_norm : dict
        Final OMP residual norm per period/sensor.

    aux : dict
        Additional diagnostic information.
    """

    reconstructed_signal: np.ndarray
    recovered_representation: np.ndarray
    support: Dict[str, list[int]]
    residual_norm: Dict[str, float]
    aux: Dict[str, Any] = field(default_factory=dict)

    # Backward-compatible aliases.
    @property
    def alpha_hat(self) -> np.ndarray:
        return self.recovered_representation

    @property
    def sparse_coefficients(self) -> np.ndarray:
        return self.recovered_representation

    @property
    def coefficients(self) -> np.ndarray:
        return self.recovered_representation


# =============================================================================
# CORE
# =============================================================================

class CSAcquisitionCore:
    """
    Compressive Sensing acquisition and reconstruction core.

    Parameters
    ----------
    n_measurements : int
        Number of CS measurements M.

    sparsity : int
        OMP target sparsity K.

    basis : str
        Sparsifying basis.

        Supported:
        - "dct"
        - "identity"

    sensing_matrix : str
        Sensing matrix type.

        Supported:
        - "gaussian"
        - "bernoulli"
        - "identity"

    random_state : int or None
        Random seed used to generate the sensing matrix.

    normalize_dictionary_columns : bool
        If True, normalize columns of A = Phi Psi before OMP correlation.
        Least-squares coefficients are mapped back internally so that the final
        alpha is compatible with the original dictionary.

    store_true_representation : bool
        If True, acquisition stores the true basis coefficients of x.

    quantize_measurements : bool
        If True, apply explicit scalar quantization to y = Phi x.

    measurement_bits : int or None
        Number of bits per CS measurement when quantize_measurements is True.

    quantization_mode : str
        Measurement quantization mode.

        Supported:
        - "uniform_midrise"

    quantization_range : str
        Range strategy.

        Supported:
        - "per_signal"
        - "global"
        - "fixed"

        "per_signal":
            each period/sensor y vector gets its own [min, max].

        "global":
            one [min, max] is computed over all y values in the acquisition
            tensor.

        "fixed":
            use measurement_quantization_min and measurement_quantization_max.

    measurement_quantization_min : float or None
        Lower bound for fixed quantization range.

    measurement_quantization_max : float or None
        Upper bound for fixed quantization range.

    clip_quantization : bool
        If True, values outside the quantization range are clipped to the edge
        bins. If False, out-of-range values raise an error.
    """

    def __init__(
        self,
        n_measurements: int,
        sparsity: int,
        basis: str = "dct",
        sensing_matrix: str = "gaussian",
        random_state: Optional[int] = None,
        normalize_dictionary_columns: bool = True,
        store_true_representation: bool = False,
        quantize_measurements: bool = False,
        measurement_bits: Optional[int] = None,
        quantization_mode: str = "uniform_midrise",
        quantization_range: str = "per_signal",
        measurement_quantization_min: Optional[float] = None,
        measurement_quantization_max: Optional[float] = None,
        clip_quantization: bool = True,
        **kwargs,
    ):
        _ = kwargs

        self.n_measurements = int(n_measurements)
        self.sparsity = int(sparsity)

        if self.n_measurements < 1:
            raise ValueError("n_measurements must be >= 1.")

        if self.sparsity < 1:
            raise ValueError("sparsity must be >= 1.")

        self.basis = str(basis).lower()
        self.sensing_matrix = str(sensing_matrix).lower()
        self.random_state = random_state

        self.normalize_dictionary_columns = bool(normalize_dictionary_columns)
        self.store_true_representation = bool(store_true_representation)

        self.quantize_measurements = bool(quantize_measurements)
        self.measurement_bits = (
            int(measurement_bits)
            if measurement_bits is not None
            else None
        )
        self.quantization_mode = str(quantization_mode).lower()
        self.quantization_range = str(quantization_range).lower()
        self.measurement_quantization_min = measurement_quantization_min
        self.measurement_quantization_max = measurement_quantization_max
        self.clip_quantization = bool(clip_quantization)

        if self.quantize_measurements:
            if self.measurement_bits is None:
                raise ValueError(
                    "measurement_bits must be provided when "
                    "quantize_measurements=True."
                )

            if self.measurement_bits < 1:
                raise ValueError("measurement_bits must be >= 1.")

            if self.quantization_mode not in {"uniform_midrise"}:
                raise ValueError(
                    "Unsupported quantization_mode. "
                    "Supported: 'uniform_midrise'."
                )

            if self.quantization_range not in {"per_signal", "global", "fixed"}:
                raise ValueError(
                    "Unsupported quantization_range. "
                    "Supported: 'per_signal', 'global', 'fixed'."
                )

            if self.quantization_range == "fixed":
                if self.measurement_quantization_min is None:
                    raise ValueError(
                        "measurement_quantization_min is required for fixed range."
                    )
                if self.measurement_quantization_max is None:
                    raise ValueError(
                        "measurement_quantization_max is required for fixed range."
                    )
                if self.measurement_quantization_max <= self.measurement_quantization_min:
                    raise ValueError(
                        "measurement_quantization_max must be larger than "
                        "measurement_quantization_min."
                    )

        self._rng = np.random.default_rng(self.random_state)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def acquire(
        self,
        x: np.ndarray,
        t: np.ndarray,
        **kwargs,
    ) -> CSAcquisitionResult:
        """
        Acquire CS measurements from a signal tensor.

        Parameters
        ----------
        x : np.ndarray
            Signal tensor with shape:

                (time, periods, sensors)

        t : np.ndarray
            Time vector with length equal to x.shape[0].

        Returns
        -------
        CSAcquisitionResult
        """

        _ = kwargs

        x = _ensure_3d_signal_tensor(x)
        t = _ensure_1d_time_vector(t)

        if x.shape[0] != len(t):
            raise ValueError(
                f"x.shape[0]={x.shape[0]} must match len(t)={len(t)}."
            )

        n_time, n_periods, n_sensors = x.shape

        M = min(self.n_measurements, n_time)
        if M < self.n_measurements:
            # Keep behavior safe if caller requested too many measurements.
            self_n_measurements_original = self.n_measurements
            self.n_measurements = M
        else:
            self_n_measurements_original = self.n_measurements

        Phi = self._build_sensing_matrix(M=M, N=n_time)
        Psi = self._build_basis_matrix(N=n_time)
        A = Phi @ Psi

        measurements_real = np.zeros((M, n_periods, n_sensors), dtype=float)
        true_representation = (
            np.zeros((n_time, n_periods, n_sensors), dtype=float)
            if self.store_true_representation
            else None
        )

        for p in range(n_periods):
            for s in range(n_sensors):
                x_vec = np.asarray(x[:, p, s], dtype=float)
                measurements_real[:, p, s] = Phi @ x_vec

                if self.store_true_representation:
                    true_representation[:, p, s] = Psi.T @ x_vec

        if self_n_measurements_original != self.n_measurements:
            self.n_measurements = self_n_measurements_original

        if self.quantize_measurements:
            (
                q_indices,
                measurements_dequantized,
                q_metadata,
            ) = self._quantize_measurement_tensor(measurements_real)

            measurements_for_reconstruction = measurements_dequantized
        else:
            q_indices = None
            measurements_dequantized = None
            q_metadata = {
                "enabled": False,
            }
            measurements_for_reconstruction = measurements_real

        aux = {
            "basis": self.basis,
            "sensing_matrix": self.sensing_matrix,
            "n_measurements": int(M),
            "sparsity": int(self.sparsity),
            "quantize_measurements": bool(self.quantize_measurements),
        }

        if true_representation is not None:
            aux["true_representation"] = true_representation

        return CSAcquisitionResult(
            measurements=measurements_for_reconstruction,
            measurements_real=measurements_real,
            measurements_quantized_indices=q_indices,
            measurements_dequantized=measurements_dequantized,
            sensing_matrix=Phi,
            basis_matrix=Psi,
            dictionary_matrix=A,
            original_shape=x.shape,
            time=t,
            quantization_metadata=q_metadata,
            aux=aux,
        )

    def reconstruct(
        self,
        acquisition_result: CSAcquisitionResult,
        **kwargs,
    ) -> CSReconstructionResult:
        """
        Reconstruct the signal from CS measurements using OMP.

        Parameters
        ----------
        acquisition_result : CSAcquisitionResult
            Output from acquire(...).

        Returns
        -------
        CSReconstructionResult
        """

        _ = kwargs

        acq = acquisition_result

        y_tensor = np.asarray(acq.measurements, dtype=float)
        Phi = np.asarray(acq.sensing_matrix, dtype=float)
        Psi = np.asarray(acq.basis_matrix, dtype=float)
        A = np.asarray(acq.dictionary_matrix, dtype=float)

        n_time, n_periods, n_sensors = acq.original_shape

        x_hat = np.zeros((n_time, n_periods, n_sensors), dtype=float)
        alpha_hat_tensor = np.zeros((n_time, n_periods, n_sensors), dtype=float)

        support_dict: Dict[str, list[int]] = {}
        residual_norm_dict: Dict[str, float] = {}

        A_for_omp, column_norms = _prepare_dictionary_for_omp(
            A,
            normalize_columns=self.normalize_dictionary_columns,
        )

        sparsity_eff = min(self.sparsity, y_tensor.shape[0], n_time)

        for p in range(n_periods):
            for s in range(n_sensors):
                y_vec = y_tensor[:, p, s]

                alpha_scaled, support, residual_norm = _omp(
                    A=A_for_omp,
                    y=y_vec,
                    sparsity=sparsity_eff,
                )

                if self.normalize_dictionary_columns:
                    alpha_vec = _unscale_coefficients(
                        alpha_scaled=alpha_scaled,
                        column_norms=column_norms,
                    )
                else:
                    alpha_vec = alpha_scaled

                x_hat[:, p, s] = Psi @ alpha_vec
                alpha_hat_tensor[:, p, s] = alpha_vec

                key = f"period_{p}_sensor_{s}"
                support_dict[key] = list(map(int, support))
                residual_norm_dict[key] = float(residual_norm)

        aux = {
            "sparsity_effective": int(sparsity_eff),
            "basis": self.basis,
            "sensing_matrix": self.sensing_matrix,
            "quantize_measurements": bool(self.quantize_measurements),
            "measurement_bits": (
                int(self.measurement_bits)
                if self.measurement_bits is not None
                else None
            ),
        }

        return CSReconstructionResult(
            reconstructed_signal=x_hat,
            recovered_representation=alpha_hat_tensor,
            support=support_dict,
            residual_norm=residual_norm_dict,
            aux=aux,
        )

    # =========================================================================
    # MATRIX BUILDERS
    # =========================================================================

    def _build_sensing_matrix(self, M: int, N: int) -> np.ndarray:
        """
        Build sensing matrix Phi.
        """

        if M < 1 or N < 1:
            raise ValueError("M and N must be positive.")

        if self.sensing_matrix == "gaussian":
            return self._rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(M),
                size=(M, N),
            )

        if self.sensing_matrix == "bernoulli":
            signs = self._rng.choice([-1.0, 1.0], size=(M, N))
            return signs / np.sqrt(M)

        if self.sensing_matrix == "identity":
            Phi = np.zeros((M, N), dtype=float)
            rows = min(M, N)
            Phi[:rows, :rows] = np.eye(rows)
            return Phi

        raise ValueError(
            "Unsupported sensing_matrix. "
            "Supported: 'gaussian', 'bernoulli', 'identity'."
        )

    def _build_basis_matrix(self, N: int) -> np.ndarray:
        """
        Build sparsifying basis Psi.
        """

        if N < 1:
            raise ValueError("N must be positive.")

        if self.basis == "identity":
            return np.eye(N, dtype=float)

        if self.basis == "dct":
            return _build_orthonormal_dct_basis(N)

        raise ValueError(
            "Unsupported basis. Supported: 'dct', 'identity'."
        )

    # =========================================================================
    # QUANTIZATION
    # =========================================================================

    def _quantize_measurement_tensor(
        self,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Quantize measurement tensor y.

        Input shape:
            (measurements, periods, sensors)

        Returns
        -------
        q_indices : np.ndarray
            Integer quantization bin indices.

        y_tilde : np.ndarray
            Dequantized measurements.

        metadata : dict
            Quantization metadata.
        """

        y = np.asarray(y, dtype=float)

        if y.ndim != 3:
            raise ValueError("Measurement tensor must have shape (M, periods, sensors).")

        M, n_periods, n_sensors = y.shape

        q_indices = np.zeros((M, n_periods, n_sensors), dtype=np.int64)
        y_tilde = np.zeros_like(y, dtype=float)

        metadata: Dict[str, Any] = {
            "enabled": True,
            "bits": int(self.measurement_bits),
            "num_bins": int(2 ** int(self.measurement_bits)),
            "mode": self.quantization_mode,
            "range_mode": self.quantization_range,
            "clip": bool(self.clip_quantization),
            "per_signal": {},
        }

        if self.quantization_range == "global":
            y_min = float(np.min(y))
            y_max = float(np.max(y))

            q_all, y_all, meta_all = _quantize_uniform_midrise_vector(
                y.reshape(-1),
                bits=int(self.measurement_bits),
                y_min=y_min,
                y_max=y_max,
                clip=self.clip_quantization,
            )

            q_indices = q_all.reshape(y.shape)
            y_tilde = y_all.reshape(y.shape)

            metadata["global"] = meta_all
            return q_indices, y_tilde, metadata

        if self.quantization_range == "fixed":
            y_min = float(self.measurement_quantization_min)
            y_max = float(self.measurement_quantization_max)

            q_all, y_all, meta_all = _quantize_uniform_midrise_vector(
                y.reshape(-1),
                bits=int(self.measurement_bits),
                y_min=y_min,
                y_max=y_max,
                clip=self.clip_quantization,
            )

            q_indices = q_all.reshape(y.shape)
            y_tilde = y_all.reshape(y.shape)

            metadata["fixed"] = meta_all
            return q_indices, y_tilde, metadata

        # per_signal
        for p in range(n_periods):
            for s in range(n_sensors):
                y_vec = y[:, p, s]

                y_min = float(np.min(y_vec))
                y_max = float(np.max(y_vec))

                q_vec, y_vec_tilde, meta = _quantize_uniform_midrise_vector(
                    y_vec,
                    bits=int(self.measurement_bits),
                    y_min=y_min,
                    y_max=y_max,
                    clip=self.clip_quantization,
                )

                q_indices[:, p, s] = q_vec
                y_tilde[:, p, s] = y_vec_tilde

                metadata["per_signal"][f"period_{p}_sensor_{s}"] = meta

        return q_indices, y_tilde, metadata


# =============================================================================
# QUANTIZATION FUNCTIONS
# =============================================================================

def _quantize_uniform_midrise_vector(
    y: np.ndarray,
    bits: int,
    y_min: float,
    y_max: float,
    clip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Uniform mid-rise quantization of a real-valued vector.

    The quantizer maps y to integer indices:

        q in {0, ..., 2^bits - 1}

    and reconstructs each index to the corresponding bin center.

    Parameters
    ----------
    y : np.ndarray
        Real-valued input vector.

    bits : int
        Number of bits.

    y_min : float
        Lower quantization range.

    y_max : float
        Upper quantization range.

    clip : bool
        If True, out-of-range values are clipped. If False, out-of-range values
        raise ValueError.

    Returns
    -------
    q : np.ndarray
        Integer bin indices.

    y_tilde : np.ndarray
        Dequantized bin-center values.

    metadata : dict
        Quantization metadata.
    """

    y = np.asarray(y, dtype=float).reshape(-1)

    if bits < 1:
        raise ValueError("bits must be >= 1.")

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        raise ValueError("Quantization range must be finite.")

    if y_max < y_min:
        raise ValueError("y_max must be >= y_min.")

    num_bins = int(2 ** bits)

    if np.isclose(y_max, y_min):
        # Degenerate range: all measurements are effectively identical.
        q = np.zeros_like(y, dtype=np.int64)
        y_tilde = np.full_like(y, fill_value=y_min, dtype=float)

        metadata = {
            "enabled": True,
            "bits": int(bits),
            "num_bins": int(num_bins),
            "mode": "uniform_midrise",
            "range_mode_effective": "degenerate",
            "y_min": float(y_min),
            "y_max": float(y_max),
            "delta": 0.0,
            "saturated_low": 0,
            "saturated_high": 0,
        }

        return q, y_tilde, metadata

    delta = (y_max - y_min) / num_bins

    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("Invalid quantization delta.")

    below = y < y_min
    above = y > y_max

    if not clip and (np.any(below) or np.any(above)):
        raise ValueError("Input contains values outside quantization range.")

    y_clip = np.clip(y, y_min, y_max)

    q = np.floor((y_clip - y_min) / delta).astype(np.int64)
    q = np.clip(q, 0, num_bins - 1)

    y_tilde = y_min + (q.astype(float) + 0.5) * delta

    metadata = {
        "enabled": True,
        "bits": int(bits),
        "num_bins": int(num_bins),
        "mode": "uniform_midrise",
        "y_min": float(y_min),
        "y_max": float(y_max),
        "delta": float(delta),
        "saturated_low": int(np.sum(below)),
        "saturated_high": int(np.sum(above)),
    }

    return q, y_tilde, metadata


# =============================================================================
# OMP
# =============================================================================

def _omp(
    A: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    tolerance: Optional[float] = None,
) -> Tuple[np.ndarray, list[int], float]:
    """
    Orthogonal Matching Pursuit.

    Solves approximately:

        y = A alpha

    with alpha K-sparse.

    Parameters
    ----------
    A : np.ndarray
        Dictionary matrix with shape (M, N).

    y : np.ndarray
        Measurement vector with shape (M,).

    sparsity : int
        Maximum support size K.

    tolerance : float or None
        Optional stopping criterion on residual norm.

    Returns
    -------
    alpha : np.ndarray
        Estimated coefficient vector with shape (N,).

    support : list[int]
        Selected support.

    residual_norm : float
        Final residual norm.
    """

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    M, N = A.shape

    if y.shape[0] != M:
        raise ValueError(f"y length {y.shape[0]} does not match A rows {M}.")

    K = int(min(max(1, sparsity), M, N))

    residual = np.array(y, dtype=float, copy=True)
    support: list[int] = []
    alpha = np.zeros(N, dtype=float)

    if tolerance is None:
        tolerance = 1e-12

    for _ in range(K):
        correlations = A.T @ residual

        if support:
            correlations[np.asarray(support, dtype=int)] = 0.0

        j = int(np.argmax(np.abs(correlations)))

        if j in support:
            break

        support.append(j)

        A_s = A[:, support]

        coeffs, *_ = np.linalg.lstsq(A_s, y, rcond=None)

        residual = y - A_s @ coeffs
        residual_norm = float(np.linalg.norm(residual))

        if residual_norm <= tolerance:
            break

    if support:
        alpha[np.asarray(support, dtype=int)] = coeffs
        residual_norm = float(np.linalg.norm(y - A[:, support] @ coeffs))
    else:
        residual_norm = float(np.linalg.norm(y))

    return alpha, support, residual_norm


# =============================================================================
# BASIS / DICTIONARY HELPERS
# =============================================================================

def _build_orthonormal_dct_basis(N: int) -> np.ndarray:
    """
    Build orthonormal DCT-II basis matrix Psi.

    The columns of Psi are basis vectors such that:

        x = Psi alpha

    and:

        alpha = Psi.T x

    for orthonormal Psi.
    """

    if N < 1:
        raise ValueError("N must be positive.")

    Psi = np.zeros((N, N), dtype=float)

    n = np.arange(N, dtype=float)

    for k in range(N):
        if k == 0:
            scale = np.sqrt(1.0 / N)
        else:
            scale = np.sqrt(2.0 / N)

        Psi[:, k] = scale * np.cos(
            np.pi * (n + 0.5) * k / N
        )

    return Psi


def _prepare_dictionary_for_omp(
    A: np.ndarray,
    normalize_columns: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optionally normalize dictionary columns for OMP correlation stability.
    """

    A = np.asarray(A, dtype=float)

    if not normalize_columns:
        return A, np.ones(A.shape[1], dtype=float)

    column_norms = np.linalg.norm(A, axis=0)
    column_norms_safe = np.where(column_norms > 0, column_norms, 1.0)

    A_norm = A / column_norms_safe[None, :]

    return A_norm, column_norms_safe


def _unscale_coefficients(
    alpha_scaled: np.ndarray,
    column_norms: np.ndarray,
) -> np.ndarray:
    """
    Convert coefficients from normalized dictionary back to original dictionary.

    If A_norm[:, j] = A[:, j] / norm_j, then:

        A_norm alpha_scaled = A alpha

    requires:

        alpha_j = alpha_scaled_j / norm_j
    """

    alpha_scaled = np.asarray(alpha_scaled, dtype=float)
    column_norms = np.asarray(column_norms, dtype=float)

    return alpha_scaled / column_norms


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _ensure_3d_signal_tensor(x: np.ndarray) -> np.ndarray:
    """
    Ensure x has shape (time, periods, sensors).
    """

    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        return x[:, None, None]

    if x.ndim == 2:
        # Interpret as (time, periods), single sensor.
        return x[:, :, None]

    if x.ndim == 3:
        return x

    raise ValueError(
        "Signal tensor must have shape (time,), (time, periods), "
        "or (time, periods, sensors)."
    )


def _ensure_1d_time_vector(t: np.ndarray) -> np.ndarray:
    """
    Ensure t is a 1D time vector.
    """

    t = np.asarray(t, dtype=float).reshape(-1)

    if t.ndim != 1:
        raise ValueError("t must be a 1D vector.")

    if len(t) < 2:
        raise ValueError("t must contain at least two samples.")

    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")

    return t


__all__ = [
    "CSAcquisitionCore",
    "CSAcquisitionResult",
    "CSReconstructionResult",
    "CSQuantizationMetadata",
]
