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
                                    H + int(x_clones / 2) * N: H + int(1 + x_clones / 2) * N
                                    ]

        xf = xf / np.sum(h)

    elif len(x.shape) == 2:
        for ind1 in range(x.shape[1]):
            xf[:, ind1] = np.convolve(
                h,
                np.tile(x[:, ind1], x_clones)
            )[
                          H + int(x_clones / 2) * N: H + int(1 + x_clones / 2) * N
                          ]

        xf = xf / np.sum(h)

    elif len(x.shape) == 1:
        xf = np.convolve(
            h,
            np.tile(x, x_clones)
        )[
             H + int(x_clones / 2) * N: H + int(1 + x_clones / 2) * N
             ]

        xf = xf / np.sum(h)

    else:
        print('x.shape is no valid')

    return xf


def sinc_reconstruct_from_samples(
        x_samples,
        t_samples,
        t_eval,
        tau,
        sample_period=None,
        periodic_replicas=10,
):
    """
    Reconstruct a dense-time signal from uniformly spaced samples using
    truncated periodic sinc interpolation.

    Purpose
    -------
    This function is the project-wide core implementation for sample-based
    continuous-time reconstruction. Pipelines and debug scripts should use this
    helper instead of implementing local sinc interpolation logic.

    Mathematical model
    ------------------
    Given samples:

        x[k] = x(t_k)

    with uniform sampling period:

        T_s = t_{k+1} - t_k,

    the ideal bandlimited interpolation is:

        x_hat(t) = sum_k x[k] sinc((t - t_k) / T_s).

    To follow the periodic reconstruction convention used in the simulations,
    this function includes shifted copies of the sample times:

        t_k + m * tau,

    for:

        m = -periodic_replicas, ..., +periodic_replicas.

    Therefore, the implemented reconstruction is:

        x_hat(t) =
            sum_{m=-R}^{R}
            sum_k x[k] sinc((t - (t_k + m tau)) / T_s),

    where:

        R = periodic_replicas.

    Default convention
    ------------------
    The default is:

        periodic_replicas = 10

    which corresponds to using periodic copies from:

        -10 * tau  to  +10 * tau.

    This matches the convention used elsewhere in the simulation stack.

    Parameters
    ----------
    x_samples : np.ndarray
        Sample values.

        Supported shapes:
        - 1D:
              (samples,)

        - 2D:
              (samples, periods)

        - 3D:
              (samples, periods, sensors)

        More generally, the first axis must correspond to sample index, and all
        trailing axes are preserved.

    t_samples : np.ndarray
        1D sample-time vector with length equal to x_samples.shape[0].

        The samples are expected to be uniformly spaced. If sample_period is not
        provided, the function infers it from t_samples and validates uniform
        spacing.

    t_eval : np.ndarray
        1D dense time grid where the reconstruction is evaluated.

    tau : float
        Observation period / signal period.

    sample_period : float or None, optional
        Sampling period T_s.

        If None, T_s is inferred as:

            t_samples[1] - t_samples[0].

        If provided, the supplied value is used directly after validation.

    periodic_replicas : int, optional
        Number of periodic replicas on each side of the central interval.

        periodic_replicas = 10 means:
            replicas m = -10, ..., 0, ..., +10.

    Returns
    -------
    np.ndarray
        Reconstructed dense-time signal.

        Output shape:
        - if x_samples is 1D:
              (len(t_eval),)

        - if x_samples is 2D:
              (len(t_eval), periods)

        - if x_samples is 3D:
              (len(t_eval), periods, sensors)

        More generally:
              (len(t_eval),) + x_samples.shape[1:]

    Notes
    -----
    This function intentionally does not call sinc_filter(...).

    Reason:
        sinc_filter(...) is a trusted periodic low-pass filtering routine that
        operates on an already dense signal by convolving tiled copies with a
        sinc kernel.

        sinc_reconstruct_from_samples(...) solves a different problem:
        reconstructing a dense signal directly from sparse/uniform samples.

    Both functions belong in sfc.core.filters, but they serve different roles.

    Important implementation detail
    -------------------------------
    np.sinc(z) in NumPy is defined as:

        sinc(z) = sin(pi z) / (pi z).

    Therefore, the interpolation kernel is correctly written as:

        np.sinc((t - t_k) / T_s).

    Local logic policy
    ------------------
    This function centralizes the sinc sample-reconstruction logic in the core.
    Pipelines, scripts, and debug files should import and use this function
    instead of implementing their own sinc reconstruction loops.
    """
    x_samples = np.asarray(x_samples, dtype=float)
    t_samples = np.asarray(t_samples, dtype=float).reshape(-1)
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

    if tau <= 0:
        raise ValueError("tau must be positive.")

    if periodic_replicas < 0:
        raise ValueError("periodic_replicas must be nonnegative.")

    periodic_replicas = int(periodic_replicas)

    if x_samples.ndim < 1:
        raise ValueError("x_samples must have at least one dimension.")

    if x_samples.shape[0] != len(t_samples):
        raise ValueError(
            "x_samples.shape[0] must match len(t_samples). "
            f"Got x_samples.shape[0]={x_samples.shape[0]} and "
            f"len(t_samples)={len(t_samples)}."
        )

    if len(t_samples) < 1:
        raise ValueError("t_samples must contain at least one sample.")

    if len(t_eval) < 1:
        raise ValueError("t_eval must contain at least one point.")

    if len(t_samples) == 1:
        if sample_period is None:
            raise ValueError(
                "sample_period must be provided when only one sample is given."
            )

        Ts = float(sample_period)

    else:
        if np.any(np.diff(t_samples) <= 0):
            raise ValueError("t_samples must be strictly increasing.")

        if sample_period is None:
            Ts = float(t_samples[1] - t_samples[0])

            diffs = np.diff(t_samples)

            if not np.allclose(diffs, Ts):
                raise ValueError(
                    "t_samples must be uniformly spaced when sample_period is "
                    "not provided."
                )
        else:
            Ts = float(sample_period)

    if Ts <= 0:
        raise ValueError("sample_period must be positive.")

    # -------------------------------------------------------------------------
    # Preserve trailing dimensions.
    #
    # Internally, reshape:
    #
    #     (samples, periods, sensors)
    #
    # or any:
    #
    #     (samples, ...)
    #
    # into:
    #
    #     (samples, n_streams)
    #
    # so that the same interpolation loop works for 1D, 2D, and 3D inputs.
    # -------------------------------------------------------------------------
    trailing_shape = x_samples.shape[1:]
    x_flat = x_samples.reshape(x_samples.shape[0], -1)

    y_flat = np.zeros((len(t_eval), x_flat.shape[1]), dtype=float)

    replica_ids = np.arange(
        -periodic_replicas,
        periodic_replicas + 1,
        dtype=int,
    )

    # -------------------------------------------------------------------------
    # Truncated periodic sinc interpolation.
    #
    # For each periodic replica m and each sample k:
    #
    #     shifted_time = t_samples[k] + m * tau
    #     kernel       = sinc((t_eval - shifted_time) / Ts)
    #
    # Then add:
    #
    #     x_samples[k] * kernel
    #
    # to every stream.
    # -------------------------------------------------------------------------
    for m in replica_ids:
        shift = float(m) * float(tau)

        for k in range(len(t_samples)):
            shifted_time = t_samples[k] + shift
            kernel = np.sinc((t_eval - shifted_time) / Ts)

            y_flat += kernel[:, None] * x_flat[k, :][None, :]

    if len(trailing_shape) == 0:
        return y_flat[:, 0]

    return y_flat.reshape((len(t_eval),) + trailing_shape)


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
    "sinc_reconstruct_from_samples",
    "plot_fft",
    "filter_periodic",
]

