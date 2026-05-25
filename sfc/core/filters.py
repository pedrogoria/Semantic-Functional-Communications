"""
sfc/core/filters.py

Trusted core module for filtering and spectral visualization utilities.

This file isolates the filter-related logic extracted from the trusted
`sfc/sampling.py` implementation.

Currently included:
- sinc_filter(...)
- plot_fft(...)
- filter_periodic(...)

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- comments and documentation
- clearer organization

Forbidden changes:
- changing formulas
- changing convolution indexing
- changing FFT scaling
- changing frequency-axis construction
- changing real/complex handling
- changing the periodic-filtering reconstruction logic

The goal is to preserve the original numerical and logical behavior exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


def sinc_filter(x, BW=10, Tt=0.001, Tth=10, x_clones=20):
    """
    Apply the trusted sinc-based low-pass filtering routine.

    Parameters
    ----------
    x : np.ndarray
        Input signal array.

        Supported shapes (preserved from the trusted implementation):
        - 1D: (time,)
        - 2D: (time, periods)
        - 3D: (time, periods, sensors)

    BW : float, optional
        Signal bandwidth parameter used to define the sampling interval
        Ts = 1 / (2 * BW).

    Tt : float, optional
        Time step.

    Tth : float, optional
        Half-support of the sinc kernel over which the impulse response is built.

    x_clones : int, optional
        Number of tiled copies of the signal used to emulate periodic extension
        during convolution.

    Returns
    -------
    np.ndarray
        Filtered output with the same shape as the trusted implementation.

    Notes
    -----
    This preserves exactly the trusted logic:

        Ts = 1 / (2 * BW)
        th = np.arange(-Tth, Tth, Tt)
        h = np.sinc(th / Ts)

        convolution is performed over tiled copies of x
        central segment is extracted using:
            H + int(x_clones / 2) * N : H + int(1 + x_clones / 2) * N

        final output is normalized by np.sum(h)

    No changes were introduced to:
    - kernel definition
    - tiling strategy
    - slicing indices
    - normalization
    """

    Ts = 1 / (2 * BW)
    th = np.arange(-Tth, Tth, Tt)
    h = np.sinc(th / Ts)

    N = x.shape[0]
    H = int(len(h) / 2)

    xf = np.zeros(x.shape)

    if len(x.shape) == 3:
        for ind1 in range(x.shape[1]):
            for ind2 in range(x.shape[2]):
                xf[:, ind1, ind2] = np.convolve(
                    h,
                    np.tile(x[:, ind1, ind2], x_clones)
                )[
                    H + int(x_clones / 2) * N : H + int(1 + x_clones / 2) * N
                ]

        xf = xf / np.sum(h)

    elif len(x.shape) == 2:
        for ind1 in range(x.shape[1]):
            xf[:, ind1] = np.convolve(
                h,
                np.tile(x[:, ind1], x_clones)
            )[
                H + int(x_clones / 2) * N : H + int(1 + x_clones / 2) * N
            ]

        xf = xf / np.sum(h)

    elif len(x.shape) == 1:
        xf = np.convolve(
            h,
            np.tile(x, x_clones)
        )[
            H + int(x_clones / 2) * N : H + int(1 + x_clones / 2) * N
        ]

        xf = xf / np.sum(h)

    else:
        print('x.shape is no valid')

    return xf


def plot_fft(x, Tt=0.001, p_log=True):
    """
    Plot the FFT magnitude of a 1D signal exactly as in the trusted code.

    Parameters
    ----------
    x : np.ndarray
        Input 1D signal.

    Tt : float, optional
        Time step.

    p_log : bool, optional
        If True, use semilog-y plotting for the magnitude spectrum.
        If False, use linear plotting.

    Returns
    -------
    None

    Notes
    -----
    This function preserves exactly:
    - scipy.fftpack.fft(x)
    - magnitude scaling by 1 / N
    - frequency-axis definition:
          np.linspace(0.0, 1.0 / (2.0 * Tt), N // 2)
    - plotting only the positive half-spectrum

    This is intentionally kept as a plotting utility inside the extracted
    filters module because it lived in the trusted sampling file and is
    closely tied to spectral inspection of filtering/signal preparation.
    """

    N = len(x)
    yf = scipy.fftpack.fft(x)
    yf = (np.abs(yf)) / N

    xf = np.linspace(0.0, 1.0 / (2.0 * Tt), N // 2)

    fig, ax = plt.subplots()

    if p_log:
        plt.semilogy(xf, yf[:N // 2])
    else:
        plt.plot(xf, yf[:N // 2])

    plt.show()


def filter_periodic(x, W, Tt, tau):
    """
    Apply the trusted periodic Fourier-domain filtering/reconstruction routine.

    Parameters
    ----------
    x : np.ndarray
        Input 1D signal defined over a window of length `tau`.

    W : float
        Bandwidth parameter used to determine the number of Fourier terms kept.

    Tt : float
        Time step.

    tau : float
        Signal period / observation window.

    Returns
    -------
    np.ndarray
        Real-valued reconstructed signal after periodic-bandlimited filtering.

    Notes
    -----
    This function preserves exactly the trusted logic:

        Ft = 1 / Tt
        t = np.arange(-tau / 2, tau / 2, Tt)
        f = Ft / len(t) * np.arange(-len(t) / 2, len(t) / 2, 1)

        FX = zeros(int(2 * floor(tau * W / 2) + 1)) * 1j
        w0 = 2 * pi / tau
        L = int((len(FX) - 1) / 2)

        for each Fourier index:
            FX[w] = Tt * sum(x * exp(-j * w0 * (w - L) * t))

        reconstruct:
            x_f = sum(FX[w] * exp(j * w0 * (w - L) * t))

        return real(x_f)

    Even seemingly unused intermediate variables (such as `f`) are preserved
    because they were present in the trusted implementation.
    """

    Ft = 1 / Tt
    t = np.arange(-tau / 2, tau / 2, Tt)
    f = Ft / len(t) * np.arange(-len(t) / 2, len(t) / 2, 1)

    FX = 1j * np.zeros(int(2 * np.floor(tau * W / 2) + 1))
    w0 = 2 * np.pi / tau
    L = int((len(FX) - 1) / 2)

    for w in range(len(FX)):
        FX[w] = Tt * np.sum(x * np.exp(- 1j * w0 * (w - L) * t))

    x_f = np.zeros(len(t))

    for w in range(len(FX)):
        x_f = x_f + FX[w] * np.exp(1j * w0 * (w - L) * t)

    return np.real(x_f)


__all__ = [
    "sinc_filter",
    "plot_fft",
    "filter_periodic",
]
