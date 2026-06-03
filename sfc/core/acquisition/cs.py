"""
sfc/core/acquisition/cs.py

Compressive Sensing (CS) acquisition core.

Purpose
-------
This module implements a reusable CS acquisition layer that operates on the
source signal before modulation / MAC decisions.

High-level idea
---------------
Given a source signal x in R^N, assume x is sparse or compressible in some
basis Psi:

    x = Psi * alpha

where alpha is sparse or approximately sparse.

The acquisition process computes compressed measurements:

    y = Phi * x
      = Phi * Psi * alpha

where:
- Phi is the sensing matrix
- Psi is the representation basis

Reconstruction is performed here using OMP (Orthogonal Matching Pursuit).

Current scope
-------------
- real-valued signals
- tensor-aware:
      (time, periods, sensors)
- supported sparse bases:
      * "dct"
      * "identity"
- supported sensing matrices:
      * "gaussian"
      * "bernoulli"
- reconstruction:
      * OMP

Design principle
----------------
This acquisition layer is:
- modulation-agnostic
- MAC-agnostic
- source-representation focused

It is intended to become the source-acquisition block for:
    CS + FDMA
before later extending to other stacks if desired.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sfc.core.acquisition.base import (
    AcquisitionCoreBase,
    AcquisitionResult,
    ReconstructionResult,
    ensure_1d_time_vector,
    ensure_3d_signal_tensor,
    validate_time_axis_length,
)


class CSAcquisitionCore(AcquisitionCoreBase):
    """
    Compressive Sensing acquisition core.

    Parameters
    ----------
    n_measurements : int
        Number of compressed measurements M.

    sparsity : int
        Target sparsity level K used by OMP.

    basis : {"dct", "identity"}, optional
        Sparse representation basis.

    sensing_matrix : {"gaussian", "bernoulli"}, optional
        Sensing-matrix family.

    random_state : int or None, optional
        Random seed for sensing-matrix generation.

    normalize_dictionary_columns : bool, optional
        If True, OMP internally normalizes dictionary columns for atom selection.

    store_true_representation : bool, optional
        If True, store the exact basis coefficients of the original signal
        (not used by reconstruction, but useful for diagnostics).
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
        **kwargs
    ):
        super().__init__(
            n_measurements=n_measurements,
            sparsity=sparsity,
            basis=basis,
            sensing_matrix=sensing_matrix,
            random_state=random_state,
            normalize_dictionary_columns=normalize_dictionary_columns,
            store_true_representation=store_true_representation,
            **kwargs,
        )

        if n_measurements < 1:
            raise ValueError("n_measurements must be >= 1.")
        if sparsity < 1:
            raise ValueError("sparsity must be >= 1.")

        self.n_measurements = int(n_measurements)
        self.sparsity = int(sparsity)
        self.basis = str(basis).lower()
        self.sensing_matrix = str(sensing_matrix).lower()
        self.random_state = random_state
        self.normalize_dictionary_columns = bool(normalize_dictionary_columns)
        self.store_true_representation = bool(store_true_representation)

        if self.basis not in {"dct", "identity"}:
            raise ValueError("basis must be one of {'dct', 'identity'}")

        if self.sensing_matrix not in {"gaussian", "bernoulli"}:
            raise ValueError("sensing_matrix must be one of {'gaussian', 'bernoulli'}")

    # =========================================================================
    # MAIN ACQUISITION
    # =========================================================================

    def acquire(
        self,
        x: np.ndarray,
        t: np.ndarray,
        **kwargs
    ) -> AcquisitionResult:
        """
        Acquire compressed measurements from the source signal.

        Parameters
        ----------
        x : np.ndarray
            Continuous-time/source-domain signal tensor with shape:
                (time, periods, sensors)

        t : np.ndarray
            Time grid for the first axis of x.

        Returns
        -------
        AcquisitionResult
            measurements shape:
                (n_measurements, periods, sensors)
        """

        _ = kwargs

        x = ensure_3d_signal_tensor(x).astype(float, copy=False)
        t = ensure_1d_time_vector(t)
        validate_time_axis_length(x, t)

        N, n_periods, n_sensors = x.shape

        if self.n_measurements > N:
            raise ValueError(
                f"n_measurements={self.n_measurements} cannot exceed signal length N={N}"
            )

        rng = np.random.default_rng(self.random_state)

        Psi = _build_basis_matrix(self.basis, N)
        Phi = _build_sensing_matrix(
            family=self.sensing_matrix,
            n_measurements=self.n_measurements,
            signal_length=N,
            rng=rng
        )

        measurements = np.zeros((self.n_measurements, n_periods, n_sensors), dtype=float)
        true_representation = None
        if self.store_true_representation:
            true_representation = np.zeros((N, n_periods, n_sensors), dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                x_vec = x[:, p, s]
                y_vec = Phi @ x_vec
                measurements[:, p, s] = y_vec

                if self.store_true_representation:
                    # Orthonormal basis -> alpha = Psi^T x
                    true_representation[:, p, s] = Psi.T @ x_vec

        return AcquisitionResult(
            measurements=measurements,
            representation=true_representation,
            t=t,
            aux={
                "basis_name": self.basis,
                "sensing_matrix_name": self.sensing_matrix,
                "Phi": Phi,
                "Psi": Psi,
                "dictionary": Phi @ Psi,
                "n_measurements": self.n_measurements,
                "signal_length": N,
                "sparsity": self.sparsity,
                "normalize_dictionary_columns": self.normalize_dictionary_columns,
            },
        )

    # =========================================================================
    # RECONSTRUCTION
    # =========================================================================

    def reconstruct(
        self,
        acquisition_result: AcquisitionResult,
        **kwargs
    ) -> ReconstructionResult:
        """
        Reconstruct the signal from compressed measurements using OMP.

        Parameters
        ----------
        acquisition_result : AcquisitionResult
            Result returned by acquire(...)

        Keyword arguments
        -----------------
        sparsity : int, optional
            Override sparsity level for this reconstruction call.

        Returns
        -------
        ReconstructionResult
            reconstructed_signal shape:
                (time, periods, sensors)
        """

        if acquisition_result.measurements is None:
            raise ValueError("AcquisitionResult.measurements is required for CS reconstruction.")

        measurements = np.asarray(acquisition_result.measurements, dtype=float)
        if measurements.ndim != 3:
            raise ValueError(
                "measurements must have shape (n_measurements, periods, sensors)"
            )

        sparsity = int(kwargs.get("sparsity", self.sparsity))

        Phi = np.asarray(acquisition_result.aux["Phi"], dtype=float)
        Psi = np.asarray(acquisition_result.aux["Psi"], dtype=float)

        A = Phi @ Psi   # dictionary
        N = Psi.shape[0]

        _, n_periods, n_sensors = measurements.shape

        x_hat = np.zeros((N, n_periods, n_sensors), dtype=float)
        alpha_hat = np.zeros((N, n_periods, n_sensors), dtype=float)

        support_sets: List[List[List[int]]] = [
            [[] for _ in range(n_sensors)] for _ in range(n_periods)
        ]
        residual_norms = np.zeros((n_periods, n_sensors), dtype=float)

        for p in range(n_periods):
            for s in range(n_sensors):
                y = measurements[:, p, s]

                alpha_vec, support, residual_norm = _omp(
                    A=A,
                    y=y,
                    sparsity=sparsity,
                    normalize_columns=self.normalize_dictionary_columns
                )

                alpha_hat[:, p, s] = alpha_vec
                x_hat[:, p, s] = np.real(Psi @ alpha_vec)

                support_sets[p][s] = support
                residual_norms[p, s] = residual_norm

        return ReconstructionResult(
            reconstructed_signal=x_hat,
            recovered_representation=alpha_hat,
            aux={
                "support_sets": support_sets,
                "residual_norms": residual_norms,
            },
        )


# =============================================================================
# BASIS CONSTRUCTION
# =============================================================================

def _build_basis_matrix(name: str, N: int) -> np.ndarray:
    """
    Build an orthonormal representation basis matrix Psi such that:

        x = Psi * alpha
        alpha = Psi^T * x

    Supported bases
    ----------------
    - "identity"
    - "dct"
    """

    name = str(name).lower()

    if name == "identity":
        return np.eye(N, dtype=float)

    if name == "dct":
        return _build_orthonormal_dct_basis(N)

    raise ValueError(f"Unsupported basis: {name}")


def _build_orthonormal_dct_basis(N: int) -> np.ndarray:
    """
    Build an orthonormal DCT-II basis matrix of size (N, N).

    Columns are basis atoms.
    """

    Psi = np.zeros((N, N), dtype=float)

    n = np.arange(N, dtype=float)

    for k in range(N):
        if k == 0:
            Psi[:, k] = np.sqrt(1.0 / N)
        else:
            Psi[:, k] = np.sqrt(2.0 / N) * np.cos(np.pi * (n + 0.5) * k / N)

    return Psi


# =============================================================================
# SENSING MATRIX CONSTRUCTION
# =============================================================================

def _build_sensing_matrix(
    family: str,
    n_measurements: int,
    signal_length: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Build a sensing matrix Phi of shape (M, N).

    Family
    ------
    - gaussian:
          entries ~ N(0, 1/M)
    - bernoulli:
          entries in {-1/sqrt(M), +1/sqrt(M)}
    """

    family = str(family).lower()

    M = int(n_measurements)
    N = int(signal_length)

    if family == "gaussian":
        Phi = rng.normal(0.0, 1.0 / np.sqrt(M), size=(M, N))
        return Phi.astype(float)

    if family == "bernoulli":
        signs = rng.integers(0, 2, size=(M, N))
        Phi = (2 * signs - 1) / np.sqrt(M)
        return Phi.astype(float)

    raise ValueError(f"Unsupported sensing matrix family: {family}")


# =============================================================================
# OMP RECONSTRUCTION
# =============================================================================

def _omp(
    A: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    normalize_columns: bool = True
):
    """
    Orthogonal Matching Pursuit.

    Solves approximately:
        y = A * alpha

    Parameters
    ----------
    A : np.ndarray
        Dictionary matrix of shape (M, N)

    y : np.ndarray
        Measurement vector of shape (M,)

    sparsity : int
        Max number of selected atoms.

    normalize_columns : bool
        If True, normalize dictionary columns for atom selection. The least-squares
        fit is still performed on the original selected columns.

    Returns
    -------
    tuple
        (alpha_hat, support, residual_norm)
    """

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    M, N = A.shape
    if y.shape[0] != M:
        raise ValueError(f"OMP mismatch: A.shape[0]={M}, len(y)={len(y)}")

    support: List[int] = []
    residual = y.copy()

    if normalize_columns:
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms == 0.0] = 1.0
        A_sel = A / col_norms
    else:
        col_norms = np.ones(N, dtype=float)
        A_sel = A

    for _ in range(min(sparsity, N)):
        correlations = np.abs(A_sel.T @ residual)

        # avoid re-selecting an already selected atom
        if support:
            correlations[np.array(support, dtype=int)] = -np.inf

        j = int(np.argmax(correlations))

        if j in support:
            break

        support.append(j)

        As = A[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(As, y, rcond=None)
        residual = y - As @ coeffs

        if np.linalg.norm(residual) < 1e-12:
            break

    alpha_hat = np.zeros(N, dtype=float)

    if support:
        As = A[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(As, y, rcond=None)
        alpha_hat[np.array(support, dtype=int)] = coeffs

    residual_norm = float(np.linalg.norm(y - A @ alpha_hat))
    return alpha_hat, support, residual_norm


__all__ = [
    "CSAcquisitionCore",
]
