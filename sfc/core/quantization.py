"""
sfc/core/quantization.py

Trusted core module for quantization in the cosine-phase sampling pipeline.

This file isolates the quantization logic that, in the original trusted
`sfc/sampling.py`, was implemented through the functions:
- quantize(...)
- quantize_ta_tb(...)

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- encapsulation into a dedicated class
- comments and documentation

Forbidden changes:
- changing formulas
- changing clipping behavior
- changing handling of scalar/vector bounds
- changing the handling of the 9999 sentinel
- changing branching behavior based on array dimensionality

The goal is to preserve the original numerical and logical behavior exactly.
"""

import numpy as np


class QuantizationCore:
    """
    Core class for quantization operations used by the cosine-phase pipeline.

    This class preserves the behavior of the trusted standalone quantization
    functions while providing an object-oriented entry point for the core.

    Parameters
    ----------
    T : float, optional
        Signal period. Stored for compatibility with the wider core design.

    harmonics : int, optional
        Number of harmonics. Stored for compatibility.

    sensor_nodes : int, optional
        Number of sensors. Stored for compatibility.

    Notes
    -----
    The quantization logic itself is stateless and could also be implemented
    purely as free functions. The class is provided here only to keep a
    consistent interface across the core package.
    """

    def __init__(self, T=1, harmonics=3, sensor_nodes=5, **options):
        """
        Initialize the quantization core.

        Parameters
        ----------
        T : float, optional
            Signal period.

        harmonics : int, optional
            Number of harmonics.

        sensor_nodes : int, optional
            Number of sensors.

        **options : dict
            Additional options accepted only for compatibility with the overall
            package design. They are not used here.
        """

        # ------------------------------------------------------------------
        # Preserve the same naming style used by other extracted core modules.
        # ------------------------------------------------------------------
        self.name = options.pop('name', 'CPM')

        # ------------------------------------------------------------------
        # Store basic metadata for compatibility, even though quantization
        # formulas below do not depend directly on all of them.
        # ------------------------------------------------------------------
        self.T = T
        self.harmonics = harmonics
        self.sensors = sensor_nodes

        # ------------------------------------------------------------------
        # Preserve the same definition of the fundamental angular frequency
        # used throughout the trusted sampling module.
        # ------------------------------------------------------------------
        self.w0 = 2 * np.pi / T

    # ======================================================================
    # Class-based wrappers around the trusted standalone functions
    # ======================================================================

    def quantize(self, x, x_min, x_max, bins, poss=1 / 2):
        """
        Quantize values exactly as in the trusted standalone `quantize(...)`.

        Parameters
        ----------
        x : np.ndarray or scalar
            Input value(s) to be quantized.

        x_min : np.ndarray or scalar
            Minimum quantization bound(s).

        x_max : np.ndarray or scalar
            Maximum quantization bound(s).

        bins : np.ndarray or scalar
            Number of bins.

        poss : float, optional
            Positioning factor inside the quantization bin.
            The trusted default is 1/2.

        Returns
        -------
        np.ndarray or scalar
            Quantized output with the same broadcasting behavior as the trusted
            implementation.

        Notes
        -----
        This method preserves exactly:
        - bins = floor(bins)
        - delta_bin = (x_max - x_min) / bins
        - x_s = ceil((x - x_min) / delta_bin)
        - clipping of x_s to [1, bins]
        - reconstruction:
              x_s * delta_bin - delta_bin * poss + x_min
        """

        return quantize(x, x_min, x_max, bins, poss=poss)

    def quantize_ta_tb(self, ta, tb, bins, poss_ta=1 / 2, poss_tb=1 / 2):
        """
        Quantize ta and tb exactly as in the trusted standalone `quantize_ta_tb(...)`.

        Parameters
        ----------
        ta : np.ndarray
            ta phase-parameter array.

        tb : np.ndarray
            tb phase-parameter array.

        bins : np.ndarray or scalar
            Number of quantization bins.

        poss_ta : float, optional
            Positioning factor for ta quantization bins.

        poss_tb : float, optional
            Positioning factor for tb quantization bins.

        Returns
        -------
        tuple
            (ta_q, tb_q)

        Notes
        -----
        This method preserves exactly:
        - harmonic-dependent bounds:
              x_min = -pi / (n * w0)
              x_max =  pi / (n * w0)
        - branching based on len(ta.shape)
        - propagation of 9999 sentinels
        """

        return quantize_ta_tb(
            ta,
            tb,
            self.w0,
            bins,
            poss_ta=poss_ta,
            poss_tb=poss_tb
        )


def quantize(x, x_min, x_max, bins, poss=1 / 2):
    """
    Trusted standalone quantization helper extracted from `sfc/sampling.py`.

    Parameters
    ----------
    x : np.ndarray or scalar
        Input values to quantize.

    x_min : np.ndarray or scalar
        Minimum bound(s) of the quantization range.

    x_max : np.ndarray or scalar
        Maximum bound(s) of the quantization range.

    bins : np.ndarray or scalar
        Number of quantization bins.

    poss : float, optional
        Position offset inside the quantization bin.
        The trusted default is 1/2.

    Returns
    -------
    np.ndarray or scalar
        Quantized output.

    Notes
    -----
    This function preserves exactly the trusted logic:

        bins = np.floor(bins)
        delta_bin = (x_max - x_min) / bins

        x_s = np.ceil((x - x_min) / delta_bin)
        x_s[np.where(x_s > bins)] = bins
        x_s[np.where(x_s <= 0)] = 1
        x_s = x_s * delta_bin - delta_bin * poss + x_min

    No numerical safeguards, reinterpretations, or alternative rounding
    strategies are introduced.
    """

    bins = np.floor(bins)
    delta_bin = (x_max - x_min) / bins

    x_s = np.ceil((x - x_min) / delta_bin)
    x_s[np.where(x_s > bins)] = bins
    x_s[np.where(x_s <= 0)] = 1
    x_s = x_s * delta_bin - delta_bin * poss + x_min

    return x_s


def quantize_ta_tb(ta, tb, w0, bins, poss_ta=1 / 2, poss_tb=1 / 2):
    """
    Trusted standalone ta/tb quantization helper extracted from `sfc/sampling.py`.

    Parameters
    ----------
    ta : np.ndarray
        ta phase-parameter array.

    tb : np.ndarray
        tb phase-parameter array.

    w0 : float
        Fundamental angular frequency.

    bins : np.ndarray or scalar
        Number of quantization bins.

    poss_ta : float, optional
        Position offset used for ta quantization.

    poss_tb : float, optional
        Position offset used for tb quantization.

    Returns
    -------
    tuple
        (ta_q, tb_q)

        ta_q : np.ndarray
            Quantized ta.

        tb_q : np.ndarray
            Quantized tb.

    Notes
    -----
    This function preserves exactly the trusted logic:

        n = np.arange(ta.shape[1]) + 1
        x_min = -pi / (n * w0)
        x_max =  pi / (n * w0)

        ta_q = zeros(ta.shape)
        tb_q = zeros(tb.shape)

        if len(ta.shape) == 3:
            ...
        elif len(ta.shape) == 2:
            ...
        else:
            ...

        ta_q[np.where(ta == 9999)] = 9999
        tb_q[np.where(tb == 9999)] = 9999

    The 9999 sentinel handling is preserved exactly.
    """

    bins = bins
    n = np.arange(ta.shape[1]) + 1
    x_min = - np.pi / (n * w0)
    x_max = np.pi / (n * w0)

    ta_q = np.zeros(ta.shape)
    tb_q = np.zeros(tb.shape)

    if len(ta.shape) == 3:
        for indx0 in range(ta.shape[0]):
            for indx2 in range(ta.shape[2]):
                ta_q[indx0, :, indx2] = quantize(
                    ta[indx0, :, indx2],
                    x_min,
                    x_max,
                    bins,
                    poss_ta
                )
                tb_q[indx0, :, indx2] = quantize(
                    tb[indx0, :, indx2],
                    x_min,
                    x_max,
                    bins,
                    poss_tb
                )

    elif len(ta.shape) == 2:
        for indx0 in range(ta.shape[0]):
            ta_q[indx0, :] = quantize(
                ta[indx0, :],
                x_min,
                x_max,
                bins,
                poss_ta
            )
            tb_q[indx0, :] = quantize(
                tb[indx0, :],
                x_min,
                x_max,
                bins,
                poss_tb
            )

    else:
        ta_q = quantize(ta, x_min, x_max, bins, poss_ta)
        tb_q = quantize(tb, x_min, x_max, bins, poss_tb)

    ta_q[np.where(ta == 9999)] = 9999
    tb_q[np.where(tb == 9999)] = 9999

    return ta_q, tb_q