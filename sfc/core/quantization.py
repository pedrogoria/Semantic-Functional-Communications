"""
quantization.py

Core quantization routines for the SFC PHY.

This module is intentionally based on the existing trusted implementation
currently present in the project's reliable sampling code:

- quantize(x, x_min, x_max, bins)
- quantize_ta_tb(ta, tb, w0, bins)

The quantizer is a uniform mid-rise quantizer with a ceil() mapping and
saturation of bin indices, consistent with the manuscript uniform quantizer
structure (Eq. (1)): Q(x) = ceil((x - x_min)/Delta)*Delta - Delta/2 + x_min. [1](https://github.com/pedrogoria/Semantic-Functional-Communications)

Important behavioral details (kept identical to the trusted code):
- 'bins' is floored to an integer.
- Delta = (x_max - x_min)/bins
- Internal bin index uses ceil((x - x_min)/Delta)
- Saturation: indices <= 0 map to 1, indices > bins map to bins
- Reconstruction level is the bin midpoint (mid-rise): k*Delta - Delta/2 + x_min
- For ta/tb arrays, values equal to 9999 remain 9999 after quantization.

This module does not implement any sampling or channel logic. It only provides
quantization primitives that the PHY core will reuse for RbCP, delta sampling,
and any future representations.

Author: SFC Project
"""

from __future__ import annotations

import numpy as np


SENTINEL_EMPTY = 9999


def quantize(x, x_min, x_max, bins):
    """
    Uniform mid-rise quantization with ceil() binning and saturation.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).
    x_min : float or np.ndarray
        Lower bound of quantization range. Can be scalar or vector (broadcastable to x).
    x_max : float or np.ndarray
        Upper bound of quantization range. Can be scalar or vector (broadcastable to x).
    bins : float or int
        Number of quantization bins. The trusted behavior floors it.

    Returns
    -------
    x_q : float or np.ndarray
        Quantized value(s) at mid-rise reconstruction levels.

    Notes
    -----
    This function is kept behavior-compatible with the trusted implementation:

        bins = floor(bins)
        delta = (x_max - x_min)/bins
        k = ceil((x - x_min)/delta)
        k > bins -> bins
        k <= 0  -> 1
        x_q = k*delta - delta/2 + x_min

    No clipping of x is performed before computing k; saturation is done on k.
    """
    bins = np.floor(bins)

    # Convert to arrays for vectorized operations
    x_arr = np.asarray(x, dtype=float)
    x_min_arr = np.asarray(x_min, dtype=float)
    x_max_arr = np.asarray(x_max, dtype=float)

    delta_bin = (x_max_arr - x_min_arr) / bins

    x_s = np.ceil((x_arr - x_min_arr) / delta_bin)

    # Saturation on the bin index (1..bins)
    x_s = np.where(x_s > bins, bins, x_s)
    x_s = np.where(x_s <= 0, 1, x_s)

    x_q = x_s * delta_bin - delta_bin / 2.0 + x_min_arr

    # Return scalar if input was scalar
    if np.isscalar(x):
        return float(np.asarray(x_q))
    return x_q


def quantize_ta_tb(ta, tb, w0, bins):
    """
    Quantize RbCP phase-shift parameters ta and tb per harmonic.

    The canonical interval for harmonic n is:
        [-pi/(n*w0), +pi/(n*w0)]

    Parameters
    ----------
    ta : np.ndarray
        Array of ta parameters. Supported shapes:
            (periods, harmonics, sensors)
            (periods, harmonics)
            (harmonics,)
    tb : np.ndarray
        Array of tb parameters with same shape rules as ta.
    w0 : float
        Fundamental angular frequency (rad/s).
    bins : float or int
        Number of quantization bins (floored).

    Returns
    -------
    ta_q : np.ndarray
        Quantized ta with same shape as ta.
    tb_q : np.ndarray
        Quantized tb with same shape as tb.

    Notes
    -----
    This function preserves the sentinel value 9999:
    any location where ta == 9999 (or tb == 9999) remains 9999 after quantization.

    This is kept behavior-compatible with the trusted code.
    """
    ta_arr = np.asarray(ta, dtype=float)
    tb_arr = np.asarray(tb, dtype=float)

    n = np.arange(ta_arr.shape[1]) + 1 if ta_arr.ndim >= 2 else np.arange(ta_arr.shape[0]) + 1
    x_min = -np.pi / (n * w0)
    x_max = +np.pi / (n * w0)

    ta_q = np.zeros_like(ta_arr, dtype=float)
    tb_q = np.zeros_like(tb_arr, dtype=float)

    if ta_arr.ndim == 3:
        for p in range(ta_arr.shape[0]):
            for s in range(ta_arr.shape[2]):
                ta_q[p, :, s] = quantize(ta_arr[p, :, s], x_min, x_max, bins)
                tb_q[p, :, s] = quantize(tb_arr[p, :, s], x_min, x_max, bins)
    elif ta_arr.ndim == 2:
        for p in range(ta_arr.shape[0]):
            ta_q[p, :] = quantize(ta_arr[p, :], x_min, x_max, bins)
            tb_q[p, :] = quantize(tb_arr[p, :], x_min, x_max, bins)
    else:
        ta_q = quantize(ta_arr, x_min, x_max, bins)
        tb_q = quantize(tb_arr, x_min, x_max, bins)

    # Preserve sentinel values exactly
    ta_q = np.where(ta_arr == SENTINEL_EMPTY, SENTINEL_EMPTY, ta_q)
    tb_q = np.where(tb_arr == SENTINEL_EMPTY, SENTINEL_EMPTY, tb_q)

    return ta_q, tb_q


def quantize_index(x, x_min, x_max, bins):
    """
    Return the quantization bin index (1..bins) using the same rule as quantize().

    This is useful when the PHY wants to transmit indices/events instead of floats.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).
    x_min : float or np.ndarray
        Lower bound.
    x_max : float or np.ndarray
        Upper bound.
    bins : float or int
        Number of bins (floored).

    Returns
    -------
    idx : int or np.ndarray
        Bin indices in {1, 2, ..., bins}.
    """
    bins = np.floor(bins)

    x_arr = np.asarray(x, dtype=float)
    x_min_arr = np.asarray(x_min, dtype=float)
    x_max_arr = np.asarray(x_max, dtype=float)

    delta_bin = (x_max_arr - x_min_arr) / bins
    idx = np.ceil((x_arr - x_min_arr) / delta_bin)

    idx = np.where(idx > bins, bins, idx)
    idx = np.where(idx <= 0, 1, idx)

    if np.isscalar(x):
        return int(np.asarray(idx))
    return idx.astype(int)


def dequantize_index(idx, x_min, x_max, bins):
    """
    Convert a bin index (1..bins) back into the mid-rise reconstruction value.

    This is the inverse mapping of quantize_index() + quantize() reconstruction.

    Parameters
    ----------
    idx : int or np.ndarray
        Bin index (1..bins).
    x_min : float or np.ndarray
        Lower bound.
    x_max : float or np.ndarray
        Upper bound.
    bins : float or int
        Number of bins (floored).

    Returns
    -------
    x_q : float or np.ndarray
        Mid-rise reconstructed value.
    """
    bins = np.floor(bins)

    idx_arr = np.asarray(idx, dtype=float)
    x_min_arr = np.asarray(x_min, dtype=float)
    x_max_arr = np.asarray(x_max, dtype=float)

    delta_bin = (x_max_arr - x_min_arr) / bins

    # Saturate index to [1, bins]
    idx_arr = np.where(idx_arr > bins, bins, idx_arr)
    idx_arr = np.where(idx_arr <= 0, 1, idx_arr)

    x_q = idx_arr * delta_bin - delta_bin / 2.0 + x_min_arr

    if np.isscalar(idx):
        return float(np.asarray(x_q))
    return x_q


if __name__ == "__main__":
    # Minimal self-test / sanity check.
    # This is useful when running this file directly from an IDE.
    x_min_test = -1.0
    x_max_test = 1.0
    bins_test = 4

    xs = np.array([-1.2, -1.0, -0.7, -0.1, 0.1, 0.7, 1.0, 1.2], dtype=float)

    qv = quantize(xs, x_min_test, x_max_test, bins_test)
    qi = quantize_index(xs, x_min_test, x_max_test, bins_test)
    qd = dequantize_index(qi, x_min_test, x_max_test, bins_test)

    print("x :", xs)
    print("q :", qv)
    print("i :", qi)
    print("d :", qd)
