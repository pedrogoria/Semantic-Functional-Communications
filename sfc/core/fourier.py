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

Design decision
---------------
The helper `cpm_sample(...)` must now obtain ta/tb ONLY through the trusted
complex-logarithm implementation exposed by `sfc.core.phase_cof.calc_ta_tb(...)`.
"""

import numpy as np

from sfc.core.phase_cof import calc_ta_tb


class FourierCoefficientCore:
    """
    Core class for Fourier-coefficient computation in the cosine-phase pipeline.
    """

    def __init__(self, T=1, harmonics=3, sensor_nodes=5, **options):
        self.name = options.pop('name', 'CPM')
        self.T = T
        self.harmonics = harmonics
        self.sensors = sensor_nodes
        self.w0 = 2 * np.pi / T
        self.dft_signal_periods = int(options.pop('dft_signal_periods', 1))

    def calc_an_bn_dft(self, x, Tt, **opt):
        """
        Compute Fourier coefficients a_n and b_n exactly as in the trusted code.
        """

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(x.shape) == 2:
            xx = np.zeros(np.append(x.shape, self.sensors))
            xx[:, :, 0] = x
            x = xx

        t = opt.pop('t', np.arange(x.shape[0] * self.dft_signal_periods) * Tt)

        FX = np.zeros((x.shape[1], self.harmonics + 1, self.sensors)) * 1j

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

        an = 2 * np.real(FX[:, 1:, :])
        bn = -2 * np.imag(FX[:, 1:, :])

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

    This helper computes:
    1. Fourier coefficients with `calc_an_bn_dft(...)`
    2. Phase parameters with the trusted complex-logarithm helper from
       `sfc.core.phase_cof.calc_ta_tb(...)`

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
        (ta, tb) as REAL 1D arrays.
    """

    an, bn, _ = calc_an_bn_dft(x, Tt, T, NN)
    return calc_ta_tb(an, bn, np.arange(1, NN + 1), 2 * np.pi / T)
