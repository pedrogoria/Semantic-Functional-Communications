"""
sfc/core/channel/physical_channel.py

Physical channel block for SFC.

Physical interpretation
-----------------------
This module models the SFC received resource-time frame after map placement and
collision/superposition.

The input frame is:

    signal.shape = (rx_slots_total, R)

and the output frame is:

    y.shape = (rx_slots_total, R)

The SFC physical-power convention is:

    P = average transmit power per sensor
    tau = signal period
    N = number of harmonics
    L = number of active chips/rows per SFC map

Each sensor transmits 2N semantic events per period. Each event map has L active
chips. Therefore, each sensor transmits:

    2 N L

active chips per period.

To enforce average transmit power P per sensor over tau, the total energy per
sensor per period is:

    E_sensor = P tau

and the energy per active SFC chip is:

    E_chip = P tau / (2 N L)

The current SFC channel works at the matched-filter/resource-output level.
Therefore, the signal multiplier is:

    sfc_signal_level = sqrt(E_chip)
                     = sqrt(P tau / (2 N L))

This is the matched-filter equivalent of the manuscript pulse-amplitude formula:

    A = sqrt(tau P B / (4 L R N))

because, with chip duration:

    T_chip = 2R / B

we have:

    sqrt(E_chip) = A sqrt(T_chip)

Noise model
-----------
The additive noise is modeled as circular complex Gaussian:

    n ~ CN(0, N0)

implemented as:

    n = sqrt(N0/2) (n_I + j n_Q)

with:

    n_I, n_Q ~ N(0, 1)

so that:

    E[|n|^2] = N0

Important note
--------------
The scalar:

    SNR_reference = P / (B N0)

may still be reported by system_parameters.py as a classical wideband reference
quantity. However, it is not the direct SFC event-detection SNR. The relevant
SFC signal level at the matched-filter output is governed by E_chip.
"""

from __future__ import annotations

import numpy as np

from sfc.core.system_parameters import build_derived_system_parameters


class PhysicalChannel:
    """
    Physical channel model for SFC.

    Supported modes
    ---------------
    - clean
    - awgn

    Input / Output shape
    --------------------
    Input:
        signal : np.ndarray
            Shape:
                (rx_slots_total, R)

    Output:
        y : np.ndarray
            Shape:
                (rx_slots_total, R), complex-valued
    """

    def __init__(self, cfg):
        """
        Initialize the physical channel.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        """

        self.cfg = cfg

        # ------------------------------------------------------------------
        # Centralized derived parameters
        # ------------------------------------------------------------------
        self.params = build_derived_system_parameters(cfg)

        self.P = float(self.params.P)
        self.B = float(self.params.B)
        self.N0 = float(self.params.N0)
        self.tau = float(self.params.tau)
        self.L = int(self.params.L)
        self.R = int(self.params.R)

        # Classical wideband reference SNR, useful for diagnostics only.
        self.SNR = float(self.params.SNR)
        self.SNR_dB = float(self.params.SNR_dB)

        # ------------------------------------------------------------------
        # Number of harmonics N.
        #
        # Prefer explicit signal.N_override when available because the SFC
        # phase/event representation in pipelines often fixes N this way.
        # Otherwise, use the centrally derived params.N.
        # ------------------------------------------------------------------
        signal_cfg = cfg.get("signal", {})

        if signal_cfg.get("N_override", None) is not None:
            self.N = int(signal_cfg["N_override"])
        else:
            self.N = int(self.params.N)

        if self.N < 1:
            raise ValueError("SFC PhysicalChannel requires N >= 1.")

        if self.L < 1:
            raise ValueError("SFC PhysicalChannel requires L >= 1.")

        if self.P <= 0:
            raise ValueError("SFC PhysicalChannel requires P > 0.")

        if self.tau <= 0:
            raise ValueError("SFC PhysicalChannel requires tau > 0.")

        if self.N0 <= 0:
            raise ValueError("SFC PhysicalChannel requires N0 > 0.")

        # ------------------------------------------------------------------
        # SFC matched-filter/resource-output energy normalization.
        #
        # Each sensor transmits 2N events per period.
        # Each event map has L active chips.
        #
        # E_sensor = P * tau
        # E_chip   = E_sensor / (2 * N * L)
        #
        # Since this channel operates at matched-filter output level:
        #
        # y_clean = sqrt(E_chip) * signal
        # ------------------------------------------------------------------
        self.E_sensor = self.P * self.tau
        self.num_events_per_sensor = 2 * self.N
        self.num_active_chips_per_sensor = self.num_events_per_sensor * self.L

        self.E_event = self.E_sensor / self.num_events_per_sensor
        self.E_chip = self.E_sensor / self.num_active_chips_per_sensor

        self.sfc_signal_level = float(np.sqrt(self.E_chip))

        # Manuscript physical pulse amplitude.
        #
        # A = sqrt(tau P B / (4 L R N))
        #
        # This is not directly used in the matched-filter frame, but it is kept
        # for diagnostics and consistency checks.
        self.sfc_pulse_amplitude = float(
            np.sqrt((self.tau * self.P * self.B) / (4.0 * self.L * self.R * self.N))
        )

        # ------------------------------------------------------------------
        # Backward-compatible aliases.
        #
        # Some existing modules may still inspect E_tot, E_s, or signal_level.
        # For SFC, signal_level should now mean the matched-filter chip level.
        # ------------------------------------------------------------------
        self.E_tot = self.E_sensor
        self.E_s = self.E_chip
        self.signal_level = self.sfc_signal_level

        # ------------------------------------------------------------------
        # Channel type
        # ------------------------------------------------------------------
        self.mode = cfg.get("channel", {}).get("type", "awgn")

    def transmit(self, signal):
        """
        Transmit the final SFC resource-time frame through the physical channel.

        Parameters
        ----------
        signal : np.ndarray
            Final channel frame with shape:

                (rx_slots_total, R)

            This matrix is interpreted as the structured resource-time frame
            before physical-layer matched-filter scaling and noise.

        Returns
        -------
        np.ndarray
            Complex-valued received frame with the same shape as input.
        """

        signal = np.asarray(signal)

        assert len(signal.shape) == 2, \
            "signal must have shape (rx_slots_total, R)"

        if self.mode == "clean":
            return self._clean(signal)

        if self.mode == "awgn":
            return self._awgn(signal)

        raise ValueError(f"Unknown channel type: {self.mode}")

    def _clean(self, signal):
        """
        Clean channel without additive noise.

        Matched-filter/resource-output model:

            y = sqrt(E_chip) * signal

        where:

            E_chip = P tau / (2 N L)
        """

        transmitted = self.sfc_signal_level * signal.astype(complex)
        return transmitted

    def _awgn(self, signal):
        """
        Apply complex AWGN to the final SFC resource-time frame.

        Matched-filter/resource-output model:

            y = sqrt(E_chip) * signal + n

        where:

            E_chip = P tau / (2 N L)

        and:

            n ~ CN(0, N0)
        """

        transmitted = self.sfc_signal_level * signal.astype(complex)

        noise = np.sqrt(self.N0 / 2.0) * (
            np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
        )

        return transmitted + noise

    def diagnostics(self):
        """
        Return physical-channel diagnostic quantities.
        """

        return {
            "P": self.P,
            "B": self.B,
            "N0": self.N0,
            "tau": self.tau,
            "R": self.R,
            "L": self.L,
            "N": self.N,
            "SNR_reference": self.SNR,
            "SNR_reference_dB": self.SNR_dB,
            "E_sensor": self.E_sensor,
            "num_events_per_sensor": self.num_events_per_sensor,
            "num_active_chips_per_sensor": self.num_active_chips_per_sensor,
            "E_event": self.E_event,
            "E_chip": self.E_chip,
            "sfc_signal_level": self.sfc_signal_level,
            "sfc_pulse_amplitude": self.sfc_pulse_amplitude,
            "mode": self.mode,
        }
