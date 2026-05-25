"""
sfc/core/propagation.py

Trusted core module for propagation and channel impulse computation.

This file isolates the propagation logic that was originally embedded inside
the channel class. The goal is to preserve the exact mathematical behavior
while improving separation of responsibilities.

This module currently provides:
- path-length computation;
- path-phase computation;
- propagation amplitude computation;
- full complex envelope / impulse computation.

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- encapsulation into a dedicated class
- comments and documentation
- helper methods for readability

Forbidden changes:
- changing formulas
- changing default constants
- changing output shapes
- adding silent numerical protections that alter the original behavior

In particular, this module intentionally preserves the original behavior
when a sensor node is exactly at the base-station position, which may lead
to division by zero in the amplitude formula. That behavior must not be
silently changed here.
"""

import numpy as np


class PropagationModel:
    """
    Propagation model for sensor-to-base-station links.

    This class encapsulates the exact propagation logic from the original
    trusted implementation, namely the `get_impulse` routine.

    Parameters
    ----------
    base_station : np.ndarray, optional
        Base-station position as a NumPy array of shape (1, 2).
        The default is np.array([[0, 0]]).

    carrier_frequency : float, optional
        Carrier frequency in Hz.

    path_loss_exp : float, optional
        Path-loss exponent used exactly in the original amplitude formula.

    Notes
    -----
    The original trusted code computes the envelope as:

        a1 = (self.sn_positions - self.base_station)
        path_length = reshape(sqrt(diag(a1.dot(a1.T))), (len(a1), 1))
        path_phase = mod(-path_length * self.carrier_frequency / c0, 2*pi)
        amplitude = (c0 / (4*pi*self.carrier_frequency*path_length)) ** self.path_loss_exp
        envelope = amplitude * exp(1j * path_phase)

    This class preserves the same formulas exactly, while moving them to a
    dedicated propagation module.
    """

    def __init__(
        self,
        base_station=np.array([[0, 0]]),
        carrier_frequency=2.4e9,
        path_loss_exp=0
    ):
        """
        Initialize the propagation model.

        Parameters
        ----------
        base_station : np.ndarray, optional
            Base-station position stored exactly as provided.

        carrier_frequency : float, optional
            Carrier frequency in Hz.

        path_loss_exp : float, optional
            Path-loss exponent.
        """

        # ------------------------------------------------------------------
        # Preserve the provided base-station representation as an attribute.
        # No shape correction is performed here, because shape validation and
        # fallback behavior belonged to the calling channel class in the
        # original implementation.
        # ------------------------------------------------------------------
        self.base_station = base_station

        # ------------------------------------------------------------------
        # Store carrier frequency exactly as a scalar attribute.
        # ------------------------------------------------------------------
        self.carrier_frequency = carrier_frequency

        # ------------------------------------------------------------------
        # Store path-loss exponent exactly as a scalar attribute.
        # ------------------------------------------------------------------
        self.path_loss_exp = path_loss_exp

    # ======================================================================
    # Helper computations
    # ======================================================================

    def compute_relative_positions(self, sn_positions):
        """
        Compute the position of each sensor node relative to the base station.

        Parameters
        ----------
        sn_positions : np.ndarray
            Sensor-node positions with shape (N, 2).

        Returns
        -------
        np.ndarray
            Relative-position array of shape (N, 2).

        Notes
        -----
        This preserves the exact original operation:

            a1 = (self.sn_positions - self.base_station)
        """

        return sn_positions - self.base_station

    def compute_path_length(self, sn_positions):
        """
        Compute the propagation path length from each sensor node to the base station.

        Parameters
        ----------
        sn_positions : np.ndarray
            Sensor-node positions with shape (N, 2).

        Returns
        -------
        np.ndarray
            Column vector of shape (N, 1) containing path lengths.

        Notes
        -----
        This preserves the exact original expression:

            a1 = (self.sn_positions - self.base_station)
            path_length = np.reshape(np.sqrt(np.diag(a1.dot(a1.T))), (len(a1), 1))

        Although a simpler formulation could be used, this exact algebraic form
        is preserved intentionally to match the trusted code.
        """

        a1 = self.compute_relative_positions(sn_positions)
        path_length = np.reshape(np.sqrt(np.diag(a1.dot(a1.T))), (len(a1), 1))
        return path_length

    def compute_path_phase(self, path_length, c0=3e8):
        """
        Compute the propagation phase term.

        Parameters
        ----------
        path_length : np.ndarray
            Column vector of path lengths with shape (N, 1).

        c0 : float, optional
            Propagation speed constant. The default is 3e8.

        Returns
        -------
        np.ndarray
            Column vector of phase values with shape (N, 1).

        Notes
        -----
        This preserves the exact original expression:

            path_phase = np.mod(-path_length * self.carrier_frequency / c0, 2 * np.pi)
        """

        path_phase = np.mod(-path_length * self.carrier_frequency / c0, 2 * np.pi)
        return path_phase

    def compute_amplitude(self, path_length, c0=3e8):
        """
        Compute the propagation amplitude term.

        Parameters
        ----------
        path_length : np.ndarray
            Column vector of path lengths with shape (N, 1).

        c0 : float, optional
            Propagation speed constant. The default is 3e8.

        Returns
        -------
        np.ndarray
            Column vector of propagation amplitudes with shape (N, 1).

        Notes
        -----
        This preserves the exact original expression:

            amplitude = (c0 / (4 * np.pi * self.carrier_frequency * path_length)) ** self.path_loss_exp

        Important:
        ----------
        No division-by-zero protection is introduced here, because the original
        trusted implementation did not introduce one either.
        """

        amplitude = (
            c0 / (4 * np.pi * self.carrier_frequency * path_length)
        ) ** self.path_loss_exp

        return amplitude

    # ======================================================================
    # Main public API
    # ======================================================================

    def get_impulse(self, sn_positions, c0=3e8):
        """
        Compute the complex propagation envelope (impulse coefficient).

        Parameters
        ----------
        sn_positions : np.ndarray
            Sensor-node positions with shape (N, 2).

        c0 : float, optional
            Propagation speed constant. The default is 3e8.

        Returns
        -------
        np.ndarray
            Complex envelope array with shape (N, 1).

        Notes
        -----
        This method preserves the exact original logic:

            a1 = (self.sn_positions - self.base_station)
            path_length = np.reshape(np.sqrt(np.diag(a1.dot(a1.T))), (len(a1), 1))
            path_phase = np.mod(-path_length * self.carrier_frequency / c0, 2 * np.pi)
            amplitude = (c0 / (4 * np.pi * self.carrier_frequency * path_length)) ** self.path_loss_exp
            envelope = amplitude * np.exp(1j * path_phase)

        The result is a complex-valued propagation coefficient for each sensor.
        """

        path_length = self.compute_path_length(sn_positions)
        path_phase = self.compute_path_phase(path_length, c0=c0)
        amplitude = self.compute_amplitude(path_length, c0=c0)

        envelope = amplitude * np.exp(1j * path_phase)
        return envelope
