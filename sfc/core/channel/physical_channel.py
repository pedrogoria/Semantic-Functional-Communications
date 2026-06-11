"""
sfc/core/channel/physical_channel.py

Physical channel utilities and SFC physical channel block.

This module contains two related pieces:

1. Generic AWGN helpers
   --------------------
   Project-wide AWGN functions that should be used by simulations whenever a
   waveform/resource-level additive Gaussian channel is needed.

   The generic convention is:

       real AWGN:
           n ~ N(0, N0)

       complex circular AWGN:
           n ~ CN(0, N0)
           n = sqrt(N0/2) * (n_I + j n_Q)

   with:

       n_I, n_Q ~ N(0, 1)

   Therefore:

       E[|n|^2] = N0

   IMPORTANT
   ---------
   These generic AWGN helpers do NOT multiply N0 by B, B_s, or any bandwidth.

   The interpretation is that the input signal is already represented at the
   appropriate matched-filter / resource-output / sampled-domain level, and N0
   is the corresponding noise variance parameter at that level.

   If a simulation is designed with fixed SNR while sweeping bandwidth, then the
   pipeline/builder must first resolve the appropriate N0(B). After that, the
   channel simulation uses only the resolved N0.

   In other words:

       SNR is diagnostic / capacity-related.
       N0 is the direct noise parameter used by the physical simulation.

2. PhysicalChannel
   ---------------
   SFC-specific physical channel model for the SFC resource-time frame.

   This class applies SFC matched-filter/resource-output signal normalization:

       y_clean = sqrt(E_chip) * signal

   where:

       E_chip = P tau / (2 N L)

   and then applies AWGN using the generic complex AWGN helper:

       n ~ CN(0, N0)

Physical interpretation for SFC
-------------------------------
This module models the SFC received resource-time frame after map placement and
collision/superposition.

The input frame is:

    signal.shape = (rx_slots_total, R)

and the output frame is:

    y.shape = (rx_slots_total, R)

The SFC physical-power convention is:

    P   = average transmit power per sensor
    tau = signal period
    N   = number of harmonics
    L   = number of active chips/rows per SFC map

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

Important note
--------------
The scalar:

    SNR_reference = P / (B N0)

may still be reported by system_parameters.py as a classical wideband reference
quantity. However, it is not the direct SFC event-detection SNR. The relevant
SFC signal level at the matched-filter output is governed by E_chip, and the
resource-output noise variance is governed directly by N0.
"""

from __future__ import annotations

import numpy as np

from sfc.core.system_parameters import build_derived_system_parameters


# =============================================================================
# GENERIC AWGN HELPERS
# =============================================================================

def generate_awgn(
    shape,
    N0,
    complex_noise=True,
    rng=None,
    dtype=None,
):
    """
    Generate additive white Gaussian noise using N0 directly as the noise
    variance parameter.

    This is the project-wide generic AWGN noise generator.

    Physical convention
    -------------------
    This helper does NOT use bandwidth.

    It does NOT compute:

        B * N0

    and it does NOT compute:

        B_s * N0

    The reason is that this helper is intended to operate at the waveform,
    matched-filter, sampled, or resource-output level where the signal has
    already been mapped into the domain used by the simulation. In that domain,
    N0 is interpreted as the direct noise variance parameter.

    If an experiment requires a fixed SNR while sweeping B, the pipeline must
    first resolve N0(B). Then this function should be called with that resolved
    N0.

    Noise definitions
    -----------------
    If complex_noise=True:

        n ~ CN(0, N0)

    implemented as:

        n = sqrt(N0/2) * (n_I + j n_Q)

    with:

        n_I, n_Q ~ N(0, 1)

    so that:

        E[|n|^2] = N0

    If complex_noise=False:

        n ~ N(0, N0)

    implemented as:

        n = sqrt(N0) * n_I

    with:

        n_I ~ N(0, 1)

    so that:

        E[n^2] = N0

    Parameters
    ----------
    shape : tuple
        Desired noise shape.

    N0 : float
        Noise variance parameter at the simulation domain.

    complex_noise : bool, optional
        If True, generate circular complex Gaussian noise.
        If False, generate real Gaussian noise.

    rng : np.random.Generator or None, optional
        Random number generator.

        If None, this function uses np.random.default_rng().

    dtype : data-type or None, optional
        Optional dtype cast for the generated noise. If None, the default is:
        - complex for complex_noise=True
        - float for complex_noise=False

    Returns
    -------
    np.ndarray
        Generated AWGN noise array.
    """
    N0 = float(N0)

    if N0 < 0:
        raise ValueError("N0 must be nonnegative.")

    if rng is None:
        rng = np.random.default_rng()

    if N0 == 0.0:
        if complex_noise:
            noise = np.zeros(shape, dtype=complex)
        else:
            noise = np.zeros(shape, dtype=float)

        if dtype is not None:
            noise = noise.astype(dtype)

        return noise

    if complex_noise:
        noise = np.sqrt(N0 / 2.0) * (
            rng.normal(0.0, 1.0, size=shape)
            + 1j * rng.normal(0.0, 1.0, size=shape)
        )
    else:
        noise = np.sqrt(N0) * rng.normal(0.0, 1.0, size=shape)

    if dtype is not None:
        noise = noise.astype(dtype)

    return noise


def apply_awgn(
    signal,
    N0,
    complex_noise=None,
    rng=None,
):
    """
    Add AWGN to an input signal using N0 directly.

    This is the generic project-wide AWGN channel helper.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    N0 : float
        Noise variance parameter at the simulation domain.

    complex_noise : bool or None, optional
        If True, add complex circular AWGN.
        If False, add real AWGN.
        If None, this helper chooses automatically:
            - complex AWGN if signal is complex-valued
            - real AWGN if signal is real-valued

    rng : np.random.Generator or None, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Noisy signal with the same shape as input.

    Notes
    -----
    This helper does not scale the input signal.

    If a method needs power normalization before AWGN, that normalization must be
    performed before calling this function. For example, SFC uses
    sqrt(E_chip), while FDMA waveform methods may use FDMACore power
    normalization.
    """
    signal = np.asarray(signal)

    if complex_noise is None:
        complex_noise = np.iscomplexobj(signal)

    noise = generate_awgn(
        shape=signal.shape,
        N0=N0,
        complex_noise=bool(complex_noise),
        rng=rng,
    )

    return signal + noise


def apply_scaled_awgn(
    signal,
    signal_level,
    N0,
    complex_noise=True,
    rng=None,
):
    """
    Scale a signal and then add AWGN using N0 directly.

    This helper implements:

        y = signal_level * signal + n

    where:

        n ~ CN(0, N0), if complex_noise=True
        n ~ N(0, N0),  if complex_noise=False

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    signal_level : float
        Multiplicative signal level applied before noise.

    N0 : float
        Noise variance parameter at the simulation domain.

    complex_noise : bool, optional
        If True, generate complex circular Gaussian noise.
        If False, generate real Gaussian noise.

    rng : np.random.Generator or None, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Scaled noisy signal.
    """
    signal = np.asarray(signal)
    signal_level = float(signal_level)

    if signal_level < 0:
        raise ValueError("signal_level must be nonnegative.")

    transmitted = signal_level * signal

    if complex_noise:
        transmitted = transmitted.astype(complex)

    return apply_awgn(
        signal=transmitted,
        N0=N0,
        complex_noise=complex_noise,
        rng=rng,
    )


# =============================================================================
# SFC PHYSICAL CHANNEL
# =============================================================================

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
                (rx_slots_total, R)

            Complex-valued for awgn mode.
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
        # Centralized derived parameters.
        # ------------------------------------------------------------------
        self.params = build_derived_system_parameters(cfg)

        self.P = float(self.params.P)
        self.B = float(self.params.B)
        self.N0 = float(self.params.N0)
        self.tau = float(self.params.tau)
        self.L = int(self.params.L)
        self.R = int(self.params.R)

        # Classical wideband reference SNR, useful for diagnostics only.
        # This is not used directly for noise generation.
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

        if self.R < 1:
            raise ValueError("SFC PhysicalChannel requires R >= 1.")

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

        if self.sfc_signal_level <= 0:
            raise ValueError("SFC PhysicalChannel requires sfc_signal_level > 0.")

        # ------------------------------------------------------------------
        # Manuscript physical pulse amplitude.
        #
        # A = sqrt(tau P B / (4 L R N))
        #
        # This is not directly used in the matched-filter frame, but it is kept
        # for diagnostics and consistency checks.
        # ------------------------------------------------------------------
        self.sfc_pulse_amplitude = float(
            np.sqrt((self.tau * self.P * self.B) / (4.0 * self.L * self.R * self.N))
        )

        # ------------------------------------------------------------------
        # Backward-compatible aliases.
        #
        # Some existing modules may still inspect E_tot, E_s, or signal_level.
        # For SFC, signal_level now means the matched-filter chip level.
        # ------------------------------------------------------------------
        # self.E_tot = self.E_sensor
        # self.E_s = self.E_chip
        # self.signal_level = self.sfc_signal_level

        # ------------------------------------------------------------------
        # Channel type and RNG.
        #
        # If cfg["channel"]["seed"] is provided, the physical channel is
        # reproducible independently of global np.random state.
        #
        # Otherwise, use cfg["reproducibility"]["seed"] if available.
        #
        # The RNG is created once in __init__, not inside transmit(...), so the
        # noise sequence advances across repeated channel uses.
        # ------------------------------------------------------------------
        channel_cfg = cfg.get("channel", {})

        self.mode = str(channel_cfg.get("type", "awgn")).lower()

        valid_modes = {"clean", "awgn"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Unknown channel type: {self.mode}. "
                f"Expected one of {sorted(valid_modes)}."
            )

        channel_seed = channel_cfg.get("seed", None)

        if channel_seed is None:
            channel_seed = cfg.get("reproducibility", {}).get("seed", None)

        if channel_seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(int(channel_seed))

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
            Received frame with the same shape as input.
        """
        signal = np.asarray(signal)

        if signal.ndim != 2:
            raise ValueError("signal must have shape (rx_slots_total, R).")

        if signal.shape[1] != self.R:
            raise ValueError(
                f"signal.shape[1] must equal R={self.R}, got {signal.shape[1]}."
            )

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

        implemented by the generic core AWGN helper:

            apply_scaled_awgn(..., complex_noise=True)

        Important
        ---------
        The noise variance is N0 directly.

        This method does NOT use:

            B * N0

        and does NOT use:

            B_s * N0

        because the SFC channel operates at the matched-filter/resource-output
        level, where N0 is the direct noise variance parameter.
        """
        return apply_scaled_awgn(
            signal=signal,
            signal_level=self.sfc_signal_level,
            N0=self.N0,
            complex_noise=True,
            rng=self.rng,
        )

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


__all__ = [
    "generate_awgn",
    "apply_awgn",
    "apply_scaled_awgn",
    "PhysicalChannel",
]
