"""
sfc/core/reconstruction.py

Trusted core module for signal reconstruction in the cosine-phase sampling pipeline.

This file isolates the reconstruction logic that, in the original trusted
`sfc/sampling.py`, was distributed across:
- CPSample.recover_signal
- recover_signal(ta, tb, t, w0)

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- encapsulation into a dedicated class
- comments and documentation

Forbidden changes:
- changing formulas
- changing the handling of the 9999 sentinel
- changing loop order
- changing output shapes
- changing the harmonic indexing convention

The goal is to preserve the original numerical and logical behavior exactly.
"""

import numpy as np


class ReconstructionCore:
    """
    Core class for signal reconstruction from cosine-phase parameters.

    This class preserves the behavior of the trusted `CPSample.recover_signal`
    method while isolating it from the rest of the sampling module.

    Parameters
    ----------
    T : float, optional
        Signal period / observation window.

    harmonics : int, optional
        Number of harmonics used in the reconstruction.

    sensor_nodes : int, optional
        Number of sensors.

    Notes
    -----
    The original trusted implementation stores these parameters inside
    `CPSample`. This class stores only the subset needed for reconstruction.
    """

    def __init__(self, T=1, harmonics=3, sensor_nodes=5, **options):
        """
        Initialize the reconstruction core.

        Parameters
        ----------
        T : float, optional
            Signal period.

        harmonics : int, optional
            Number of harmonics.

        sensor_nodes : int, optional
            Number of sensors.

        **options : dict
            Additional options accepted only for compatibility with the
            overall core design. They are not used here.
        """

        # ------------------------------------------------------------------
        # Preserve the trusted public naming default used elsewhere.
        # ------------------------------------------------------------------
        self.name = options.pop('name', 'CPM')

        # ------------------------------------------------------------------
        # Store signal-period and harmonic configuration exactly as scalars.
        # ------------------------------------------------------------------
        self.T = T
        self.harmonics = harmonics
        self.sensors = sensor_nodes

        # ------------------------------------------------------------------
        # Preserve the exact definition of the fundamental angular frequency.
        # ------------------------------------------------------------------
        self.w0 = 2 * np.pi / T

    # ======================================================================
    # Class-based reconstruction
    # ======================================================================

    def recover_signal(self, ta, tb, t, **options):
        """
        Reconstruct the signal from ta and tb exactly as in the trusted code.

        Parameters
        ----------
        ta : np.ndarray
            Phase parameter ta with shape:
                (n_periods, harmonics, sensors)

        tb : np.ndarray
            Phase parameter tb with shape:
                (n_periods, harmonics, sensors)

        t : np.ndarray
            Time vector used for reconstruction.

        **options : dict
            Additional keyword arguments accepted for compatibility.
            They are not used here.

        Returns
        -------
        np.ndarray
            Reconstructed signal array with shape:
                (len(t), n_periods, sensors)

        Notes
        -----
        This method preserves exactly:
        - initialization of xr with shape (len(t), ta.shape[0], sensors)
        - per-sensor and per-period loops
        - construction of an intermediate x array with shape
          (harmonics, len(t))
        - use of the 9999 sentinel to skip absent harmonics
        - summation over the harmonic dimension
        """

        xr = np.zeros((len(t), ta.shape[0], int(self.sensors)))

        for iii in range(self.sensors):
            for ii in range(ta.shape[0]):
                x = np.zeros((self.harmonics, len(t)))

                for n in range(self.harmonics):
                    if ta[ii, n, iii] != 9999:
                        x[n, :] = np.cos((n + 1) * self.w0 * (t - ta[ii, n, iii]))

                    if tb[ii, n, iii] != 9999:
                        x[n, :] = x[n, :] + np.cos((n + 1) * self.w0 * (t - tb[ii, n, iii]))

                x = np.sum(x, 0)
                xr[:, ii, iii] = x

        return xr


def recover_signal(ta, tb, t, w0):
    """
    Legacy global helper for signal reconstruction.

    This function preserves exactly the standalone trusted implementation from
    the original `sfc/sampling.py`.

    Parameters
    ----------
    ta : np.ndarray
        Phase parameter ta with shape (NN,) for a single reconstructed signal.

    tb : np.ndarray
        Phase parameter tb with shape (NN,) for a single reconstructed signal.

    t : np.ndarray
        Time vector used for reconstruction.

    w0 : float
        Fundamental angular frequency.

    Returns
    -------
    np.ndarray
        Reconstructed 1D signal.

    Notes
    -----
    This function preserves exactly:
    - the allocation of x with shape (NN, len(t))
    - the harmonic loop
    - the sum of the two cosine terms
    - the final sum across the harmonic dimension
    """

    NN = len(ta)
    x = np.zeros((NN, len(t)))

    for n in range(NN):
        x[n, :] = (
                np.cos((n + 1) * w0 * (t - ta[n]))
                + np.cos((n + 1) * w0 * (t - tb[n]))
        )

    return np.sum(x, 0)
