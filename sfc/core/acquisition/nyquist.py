"""
sfc/core/acquisition/nyquist.py

Trusted acquisition module for Nyquist sampling, quantization, and sinc-based
reconstruction.

This file isolates the `Nyquist` class extracted from the trusted
`sfc/sampling.py` implementation.

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- comments and documentation
- organization for readability

Forbidden changes:
- changing formulas
- changing defaults
- changing shape handling
- changing quantization logic
- changing capacity calculations
- changing reconstruction indexing
- changing sinc reconstruction logic

The goal is to preserve the original numerical and logical behavior exactly.
"""

import numpy as np


class Nyquist:
    """
    Trusted Nyquist sampling, quantization, and reconstruction class.

    This class preserves the exact behavior of the original implementation
    extracted from `sfc/sampling.py`.

    Responsibilities
    ----------------
    - sample a continuous-time/discrete-grid signal at the Nyquist rate;
    - optionally quantize the sampled values;
    - reconstruct the signal using sinc interpolation.

    Notes
    -----
    This class is intentionally kept faithful to the trusted implementation,
    including:
    - the channel-capacity-derived bit allocation;
    - the use of `np.floor` in time-index generation;
    - the exact clipping rules during quantization;
    - the tiled convolution strategy used in reconstruction.
    """

    def __init__(self, T=1, Tt=0.001, sampling_rate=10, sensor_nodes=5, bandwidth=100, **options):
        """
        Initialize the Nyquist sampling model.

        Parameters
        ----------
        T : float, optional
            Signal period / observation window.

        Tt : float, optional
            Time step of the discrete-time representation.

        sampling_rate : float, optional
            Sampling rate used by the Nyquist sampler.

        sensor_nodes : int, optional
            Number of sensors.

        bandwidth : float, optional
            Total channel bandwidth.

        **options : dict
            Additional trusted options:
            - name
            - T_sinc
            - snr_dB
            - bits_codeword
            - header
            - x_clones

        Notes
        -----
        The trusted implementation derives the number of quantization bits
        from the per-sensor channel capacity:

            P_N0bw = 10 ** (snr_dB / 10)
            sensor_channel_capacity = sensor_bw * log2(1 + P_N0bw)
            bits_codeword = floor(sensor_channel_capacity / sampling_rate)

        This behavior is preserved exactly.
        """

        # ------------------------------------------------------------------
        # Preserve the original public naming default.
        # ------------------------------------------------------------------
        self.name = options.pop('name', 'Nyquist')

        # ------------------------------------------------------------------
        # Store the fundamental Nyquist/signal parameters.
        # ------------------------------------------------------------------
        self.sampling_rate = sampling_rate
        self.sensor_nodes = sensor_nodes
        self.bandwidth = bandwidth
        self.T = T

        # ------------------------------------------------------------------
        # Preserve the sinc reconstruction support parameter.
        # ------------------------------------------------------------------
        self.T_sinc = options.pop('T_sinc', 10)

        # ------------------------------------------------------------------
        # Preserve the original per-sensor bandwidth split.
        # ------------------------------------------------------------------
        self.sensor_bw = bandwidth / sensor_nodes

        # ------------------------------------------------------------------
        # Preserve the original SNR parameterization in dB.
        # ------------------------------------------------------------------
        self.snr_dB = options.pop('snr_dB', 3)

        # ------------------------------------------------------------------
        # Convert SNR from dB to linear scale exactly as in the trusted code.
        # ------------------------------------------------------------------
        P_N0bw = (10 ** (self.snr_dB / 10))

        # ------------------------------------------------------------------
        # Preserve the original per-sensor Shannon capacity expression.
        # ------------------------------------------------------------------
        self.sensor_channel_capacity = self.sensor_bw * np.log2(1 + P_N0bw)

        # ------------------------------------------------------------------
        # Preserve the original default logic for bits_codeword.
        # ------------------------------------------------------------------
        self.bits_codeword = options.pop(
            'bits_codeword',
            int(np.floor(self.sensor_channel_capacity / self.sampling_rate))
        )

        # ------------------------------------------------------------------
        # Preserve the original hard cap of 1000 bits.
        # ------------------------------------------------------------------
        if self.bits_codeword > 1000:
            self.bits_codeword = 1000

        # ------------------------------------------------------------------
        # Preserve the original capacity assertion exactly.
        # ------------------------------------------------------------------
        assert self.bits_codeword * self.sampling_rate <= self.sensor_channel_capacity, \
            'bits per message exceeds the channel capacity'

        # ------------------------------------------------------------------
        # Preserve the original header logic exactly.
        # ------------------------------------------------------------------
        self.header = options.pop('header', False)

        if self.header:
            self.bits_msg = int(self.bits_codeword - np.ceil(np.log2(sensor_nodes)))
        else:
            self.bits_msg = self.bits_codeword

        # ------------------------------------------------------------------
        # Preserve the original bin count rule exactly.
        # ------------------------------------------------------------------
        self.bins = max([2 ** self.bits_msg, 1])

        # ------------------------------------------------------------------
        # Store time step and the number of tiled copies used in sinc
        # reconstruction.
        # ------------------------------------------------------------------
        self.Tt = Tt
        self.x_clones = options.pop('x_clones', 20)

    # ======================================================================
    # Sampling and optional quantization
    # ======================================================================

    def __call__(self, x, t, **options):
        """
        Sample the signal and optionally quantize the samples.

        Parameters
        ----------
        x : np.ndarray
            Input signal array.

            Supported shapes (preserved from the trusted implementation):
            - 1D: (time,)
            - 2D: (time, periods)
            - 3D: (time, periods, sensors)

        t : np.ndarray
            Time vector. Preserved as an argument for compatibility with the
            original method signature, although it is not explicitly used in the
            original sampling computation.

        **options : dict
            Optional trusted parameters:
            - quantize : bool (default = True)
            - peak2peak : 'sample' or a numeric range value

        Returns
        -------
        np.ndarray
            Sampled (and optionally quantized) signal.

        Notes
        -----
        This method preserves exactly:
        - sample-index construction:
              t_s = floor(arange(T * sampling_rate) * (1 / Tt) / sampling_rate)
        - branching by input dimensionality
        - optional quantization using the exact in-method logic
        """

        quantize = options.pop('quantize', True)

        # ------------------------------------------------------------------
        # Preserve the original sampling-index construction exactly.
        # ------------------------------------------------------------------
        t_s = np.floor(np.arange(self.T * self.sampling_rate) * (1 / self.Tt) / self.sampling_rate)

        # ------------------------------------------------------------------
        # Preserve the original input-shape assertion exactly.
        # ------------------------------------------------------------------
        assert len(x.shape) < 4, 'input shape'

        # ------------------------------------------------------------------
        # Preserve the original branching by dimensionality exactly.
        # ------------------------------------------------------------------
        if len(x.shape) == 3:
            x_s = np.zeros((len(t_s), x.shape[1], x.shape[2]))
            for ind1 in range(x.shape[1]):
                for ind2 in range(x.shape[2]):
                    x_s[:, ind1, ind2] = x[t_s.astype(int), ind1, ind2]

        elif len(x.shape) == 2:
            x_s = np.zeros((len(t_s), x.shape[1]))
            for ind1 in range(x.shape[1]):
                x_s[:, ind1] = x[t_s.astype(int), ind1]

        else:
            x_s = x[t_s.astype(int)]

        # ------------------------------------------------------------------
        # Preserve the optional quantization logic exactly.
        # ------------------------------------------------------------------
        if quantize:
            peak2peak = options.pop('peak2peak', 'sample')

            if peak2peak == 'sample':
                x_max = np.max(x, 0)
                x_min = np.min(x, 0)
                delta_bin = (x_max - x_min) / self.bins
            else:
                x_max = peak2peak / 2
                x_min = -peak2peak / 2
                delta_bin = (x_max - x_min) / self.bins

            x_s = np.ceil((x_s - x_min) / delta_bin)
            x_s[np.where(x_s > self.bins)] = self.bins
            x_s[np.where(x_s <= 0)] = 1
            x_s = x_s * delta_bin - delta_bin / 2 + x_min

        return x_s

    # ======================================================================
    # Standalone quantization of already-sampled data
    # ======================================================================

    def quantize(self, x, **options):
        """
        Quantize input values exactly as in the trusted implementation.

        Parameters
        ----------
        x : np.ndarray
            Input sampled values.

        **options : dict
            Optional trusted parameter:
            - peak2peak : 'sample' or a numeric range value

        Returns
        -------
        np.ndarray
            Quantized values.

        Notes
        -----
        This method preserves exactly the original logic and clipping rules.
        """

        peak2peak = options.pop('peak2peak', 'sample')

        if peak2peak == 'sample':
            x_max = np.max(x, 0)
            x_min = np.min(x, 0)
            delta_bin = (x_max - x_min) / self.bins
        else:
            x_max = peak2peak / 2
            x_min = -peak2peak / 2
            delta_bin = (x_max - x_min) / self.bins

        x_s = np.ceil((x - x_min) / delta_bin)
        x_s[np.where(x_s > self.bins)] = self.bins
        x_s[np.where(x_s <= 0)] = 1
        return x_s * delta_bin - delta_bin / 2 + x_min

    # ======================================================================
    # Sinc reconstruction
    # ======================================================================

    def recover_signal(self, x_s):
        """
        Reconstruct the signal from Nyquist samples using sinc interpolation.

        Parameters
        ----------
        x_s : np.ndarray
            Sampled input array.

            Supported shapes (preserved from the trusted implementation):
            - 1D: (samples,)
            - 2D: (samples, periods)
            - 3D: (samples, periods, sensors)

        Returns
        -------
        np.ndarray
            Reconstructed signal on the dense time grid.

        Notes
        -----
        This method preserves exactly:
        - sinc kernel construction:
              th = arange(-T_sinc, T_sinc, Tt)
              h = sinc(th * sampling_rate)
        - sample-position construction:
              t_s = floor(arange(T * sampling_rate) * (1 / Tt) / sampling_rate)
        - tiled convolution reconstruction
        - extraction of the central segment using the same index formulas
        """

        # ------------------------------------------------------------------
        # Preserve the original sinc kernel construction exactly.
        # ------------------------------------------------------------------
        th = np.arange(-int(self.T_sinc), int(self.T_sinc), self.Tt)
        h = np.sinc(th * self.sampling_rate)
        H = int(len(h) / 2)

        # ------------------------------------------------------------------
        # Preserve the trusted sample-position reconstruction grid exactly.
        # ------------------------------------------------------------------
        t_s = np.floor(np.arange(self.T * self.sampling_rate) * (1 / self.Tt) / self.sampling_rate)

        # ------------------------------------------------------------------
        # Preserve the dense reconstruction length exactly.
        # ------------------------------------------------------------------
        N = int(self.T / self.Tt)

        # ------------------------------------------------------------------
        # Preserve the original input-shape assertion.
        # ------------------------------------------------------------------
        assert len(x_s.shape) < 4, 'input shape'

        # ------------------------------------------------------------------
        # Preserve the original branching by dimensionality exactly.
        # ------------------------------------------------------------------
        if len(x_s.shape) == 3:
            xr = np.zeros((N, x_s.shape[1], x_s.shape[2]))
            x_s_t = np.zeros(N)

            for ind1 in range(x_s.shape[1]):
                for ind2 in range(x_s.shape[2]):
                    x_s_t[t_s.astype(int)] = x_s[:, ind1, ind2]
                    x_r = np.convolve(h, np.tile(x_s_t, self.x_clones))
                    xr[:, ind1, ind2] = x_r[
                        H + int(self.x_clones / 2) * N : H + int(1 + self.x_clones / 2) * N
                    ]

        elif len(x_s.shape) == 2:
            xr = np.zeros((N, x_s.shape[1]))
            x_s_t = np.zeros(N)

            for ind1 in range(x_s.shape[1]):
                x_s_t[t_s.astype(int)] = x_s[:, ind1]
                x_r = np.convolve(h, np.tile(x_s_t, self.x_clones))
                xr[:, ind1] = x_r[
                    H + int(self.x_clones / 2) * N : H + int(1 + self.x_clones / 2) * N
                ]

        else:
            x_s_t = np.zeros(N)
            x_s_t[t_s.astype(int)] = x_s
            x_r = np.convolve(h, np.tile(x_s_t, self.x_clones))
            xr = x_r[
                H + int(self.x_clones / 2) * N : H + int(1 + self.x_clones / 2) * N
            ]

        return xr


__all__ = ["Nyquist"]
