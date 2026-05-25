"""
sfc/core/fourier.py

Trusted core module for Fourier-coefficient computation used by the cosine-phase
sampling pipeline.

This file isolates the Fourier-related logic that, in the original trusted
`sfc/sampling.py`, was distributed mainly across:
- CPSample.calc_an_bn_dft
- calc_an_bn_dft(...)
- cpm_sample(...)

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- encapsulation into a dedicated class
- comments and documentation
- helper method extraction for readability

Forbidden changes:
- changing formulas
- changing normalization logic
- changing shape conventions
- changing default parameter behavior
- changing sampling-period handling

The goal is to preserve the original numerical and logical behavior.
"""

import numpy as np

from sfc.core.phase_cof import calc_ta_tb


class FourierCoefficientCore:
    """
    Core class for Fourier-coefficient computation in the cosine-phase pipeline.

    This class preserves the behavior of the trusted `CPSample.calc_an_bn_dft`
    method, while isolating it from the rest of the sampling module.

    Parameters
    ----------
    T : float, optional
        Signal period / observation window.

    harmonics : int, optional
        Number of harmonics.

    sensor_nodes : int, optional
        Number of sensors.

    dft_signal_periods : int, optional
        Number of repeated periods used in the internal DFT computation.

    Notes
    -----
    The original trusted implementation stores these values inside `CPSample`.
    This class stores only the subset needed for Fourier-coefficient computation.
    """

    def __init__(self, T=1, harmonics=3, sensor_nodes=5, **options):
        """
        Initialize the Fourier core.

        Parameters
        ----------
        T : float, optional
            Signal period.

        harmonics : int, optional
            Number of harmonics.

        sensor_nodes : int, optional
            Number of sensors.

        **options : dict
            Additional options. The following trusted parameter is supported:
            - dft_signal_periods
        """

        # ------------------------------------------------------------------
        # Preserve the trusted public naming default used elsewhere.
        # ------------------------------------------------------------------
        self.name = options.pop('name', 'CPM')

        # ------------------------------------------------------------------
        # Store the observation period and the number of harmonics.
        # ------------------------------------------------------------------
        self.T = T
        self.harmonics = harmonics

        # ------------------------------------------------------------------
        # Store the number of sensors for the shape-conversion logic.
        # ------------------------------------------------------------------
        self.sensors = sensor_nodes

        # ------------------------------------------------------------------
        # Preserve the exact same fundamental angular frequency definition.
        # ------------------------------------------------------------------
        self.w0 = 2 * np.pi / T

        # ------------------------------------------------------------------
        # Preserve the same trusted option controlling the number of periods
        # used in the DFT computation.
        # ------------------------------------------------------------------
        self.dft_signal_periods = int(options.pop('dft_signal_periods', 1))

    # ======================================================================
    # Main class-based Fourier computation
    # ======================================================================

    def calc_an_bn_dft(self, x, Tt, **opt):
        """
        Compute Fourier coefficients a_n and b_n exactly as in the trusted code.

        This method is a structural extraction of `CPSample.calc_an_bn_dft`.

        Parameters
        ----------
        x : np.ndarray
            Input signal. Accepted shapes follow the trusted implementation:
            - 1D: (time,)
            - 2D: (time, periods)
            - 3D: (time, periods, sensors)

        Tt : float
            Time step.

        **opt : dict
            Optional parameters preserved from the trusted implementation:
            - t : explicit time vector
            - normalize : whether to normalize if max(a_n^2 + b_n^2) >= 4
            - norm : normalization target (default = 3.9)

        Returns
        -------
        tuple
            (an, bn, x)

            an : np.ndarray
                Cosine coefficients with shape (periods, harmonics, sensors)

            bn : np.ndarray
                Sine coefficients with shape (periods, harmonics, sensors)

            x : np.ndarray
                Possibly normalized signal, preserving the original behavior

        Notes
        -----
        This method preserves exactly:
        - 1D -> 2D -> 3D shape promotion
        - construction of the time vector
        - DFT computation using repeated periods
        - coefficient extraction from FX
        - optional normalization loop
        """

        # ------------------------------------------------------------------
        # Preserve trusted shape-promotion behavior exactly.
        # ------------------------------------------------------------------
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(x.shape) == 2:
            xx = np.zeros(np.append(x.shape, self.sensors))
            xx[:, :, 0] = x
            x = xx

        # ------------------------------------------------------------------
        # Preserve the default time-vector construction exactly.
        # ------------------------------------------------------------------
        t = opt.pop('t', np.arange(x.shape[0] * self.dft_signal_periods) * Tt)

        # ------------------------------------------------------------------
        # Preserve the original FX allocation and dtype.
        # ------------------------------------------------------------------
        FX = np.zeros((x.shape[1], self.harmonics + 1, self.sensors)) * 1j

        # ------------------------------------------------------------------
        # Preserve the original nested-loop DFT logic exactly.
        # ------------------------------------------------------------------
        for iii in range(self.sensors):
            for ii in range(x.shape[1]):
                for i in range(0, self.harmonics + 1):
                    FX[ii, i, iii] = (
                        Tt
                        * np.sum(
                            np.tile(x[:, ii, iii], self.dft_signal_periods)
                            * np.exp(-1j * i * self.w0 * t)
                        )
                        / self.T
                    )
                    FX[ii, i, iii] = FX[ii, i, iii] / self.dft_signal_periods

        # ------------------------------------------------------------------
        # Preserve the exact coefficient extraction.
        # ------------------------------------------------------------------
        an = 2 * np.real(FX[:, 1:, :])
        bn = -2 * np.imag(FX[:, 1:, :])

        # ------------------------------------------------------------------
        # Preserve the optional normalization loop exactly.
        # ------------------------------------------------------------------
        if opt.pop('normalize', False):
            nor_x = opt.pop('norm', 3.9)

            for iii in range(self.sensors):
                for ii in range(x.shape[1]):
                    aux = max(an[ii, :, iii] ** 2 + bn[ii, :, iii] ** 2)

                    while aux >= 4:
                        x[:, ii, iii] = x[:, ii, iii] * np.sqrt(nor_x / aux)

                        for i in range(0, self.harmonics + 1):
                            FX[ii, i, iii] = (
                                Tt
                                * np.sum(
                                    np.tile(x[:, ii, iii], self.dft_signal_periods)
                                    * np.exp(-1j * i * self.w0 * t)
                                )
                                / self.T
                            )
                            FX[ii, i, iii] = FX[ii, i, iii] / self.dft_signal_periods

                        an[ii, :, iii] = 2 * np.real(FX[ii, 1:, iii])
                        bn[ii, :, iii] = -2 * np.imag(FX[ii, 1:, iii])
                        aux = max(an[ii, :, iii] ** 2 + bn[ii, :, iii] ** 2)

        return an, bn, x


def calc_an_bn_dft(x, Tt, T, NN, **opt):
    """
    Legacy global helper for Fourier-coefficient computation.

    This function preserves exactly the standalone trusted implementation from
    the original `sfc/sampling.py`.

    Parameters
    ----------
    x : np.ndarray
        1D signal.

    Tt : float
        Time step.

    T : float
        Signal period.

    NN : int
        Number of harmonics.

    **opt : dict
        Optional parameters preserved from the trusted implementation:
        - normalize : whether to normalize if max(a_n^2 + b_n^2) >= 4
        - norm : normalization target (default = 3.9)

    Returns
    -------
    tuple
        (an, bn, FX)

        an : np.ndarray
            Cosine coefficients.

        bn : np.ndarray
            Sine coefficients.

        FX : np.ndarray
            Complex Fourier coefficients including the DC index.

    Notes
    -----
    This function preserves exactly:
    - time-vector construction
    - DFT computation loop
    - optional normalization loop
    - return values and order
    """

    t = np.arange(len(x)) * Tt
    w0 = 2 * np.pi / T

    FX = np.zeros(NN + 1) * 1j

    for i in range(0, NN + 1):
        FX[i] = Tt * np.sum(x * np.exp(-1j * i * w0 * t)) / T

    an = 2 * np.real(FX[1:])
    bn = -2 * np.imag(FX[1:])

    if opt.pop('normalize', True):
        aux = max(an ** 2 + bn ** 2)
        nor_x = opt.pop('norm', 3.9)

        while aux >= 4:
            x = x * np.sqrt(nor_x / aux)
            FX = np.zeros(NN + 1) * 1j

            for i in range(0, NN + 1):
                FX[i] = Tt * np.sum(x * np.exp(-1j * i * w0 * t)) / T

            an = 2 * np.real(FX[1:])
            bn = -2 * np.imag(FX[1:])
            aux = max(an ** 2 + bn ** 2)

    return an, bn, FX


def cpm_sample(x, Tt, T, NN):
    """
    Trusted global helper that computes ta/tb from a 1D signal.

    This preserves exactly the original `cpm_sample(...)` helper:
    1. compute Fourier coefficients with `calc_an_bn_dft`
    2. compute phase parameters with the global `calc_ta_tb(...)`

    Parameters
    ----------
    x : np.ndarray
        1D signal.

    Tt : float
        Time step.

    T : float
        Signal period.

    NN : int
        Number of harmonics.

    Returns
    -------
    tuple
        (ta, tb)
    """

    an, bn, f = calc_an_bn_dft(x, Tt, T, NN)
    return calc_ta_tb(an, bn, np.arange(1, NN + 1), 2 * np.pi / T)