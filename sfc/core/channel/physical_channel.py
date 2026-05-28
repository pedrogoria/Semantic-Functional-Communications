"""
sfc/core/channel/physical_channel.py

Physical channel block for SFC.

PRINCIPLE ENFORCED HERE
-----------------------
We denote the signal-to-noise ratio by:

    SNR = P / (B * N0)

where:
- P  : average transmit power per sensor
- B  : channel bandwidth
- N0 : noise parameter used by the SFC physical channel

In the software:
- cfg["system"]["SNR_dB"] represents this SNR in dB
- the same P, B, and N0 are used consistently across models
- P is also the same parameter used in the Benchmark capacity calculation

CENTRALIZATION RULE
-------------------
This module does NOT recompute local physical quantities such as:
- SNR_linear
- N0
- E_tot
- E_s
- signal_level
- default_threshold

Instead, all these quantities are derived centrally through:

    build_derived_system_parameters(cfg)

so that the physical model remains consistent across:
- physical_channel.py
- detection.py
- pipelines
- future channel modules

CURRENT INTERPRETATION
----------------------
At this stage of the implementation:

1. The total available energy per cycle is:
       E_tot = P * tau

2. The energy per transmitted symbol/resource is:
       E_s = E_tot / L = (P * tau) / L

3. The output is interpreted as the matched-filter output per resource:
       y = sqrt(E_s) * signal + n

4. The additive noise is complex Gaussian:
       n ~ CN(0, N0)

5. Using the manuscript relation:
       SNR = P / (B * N0)
   we derive:
       N0 = P / (B * SNR_linear)

IMPORTANT
---------
This means that the physical channel returns a COMPLEX tensor.

Therefore, downstream detection should use |y|, correlation, or another
matched-filter-compatible decision rule, rather than a plain real-valued
threshold directly on y.

Future work may refine:
- actual pulse shape
- explicit matched filter in continuous/discrete time
- fading
- phase-aware collision superposition
"""

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
        signal : (num_time_slots, L, R)

    Output:
        y : (num_time_slots, L, R), complex-valued
    """

    def __init__(self, cfg):
        """
        Initialize the physical channel.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.

        Notes
        -----
        The following quantities are obtained centrally from
        `build_derived_system_parameters(cfg)`:

        - SNR_linear
        - N0
        - E_tot = P * tau
        - E_s   = (P * tau) / L
        - signal_level = sqrt(E_s)
        """

        self.cfg = cfg

        # ------------------------------------------------------------------
        # Centralized derived parameters
        # ------------------------------------------------------------------
        self.params = build_derived_system_parameters(cfg)

        self.P = self.params.P
        self.B = self.params.B
        self.SNR_dB = self.params.SNR_dB
        self.SNR = self.params.SNR

        self.N0 = self.params.N0
        self.E_tot = self.params.E_tot
        self.E_s = self.params.E_s
        self.signal_level = self.params.signal_level

        # ------------------------------------------------------------------
        # Channel type
        # ------------------------------------------------------------------
        self.mode = cfg.get("channel", {}).get("type", "awgn")

    def transmit(self, signal):
        """
        Transmit the aggregate tensor through the physical channel.

        PRINCIPLE
        ---------
        This function enforces:

            SNR = P / (B * N0)

        via the centrally derived parameter:

            N0 = P / (B * SNR_linear)

        and models the matched-filter output as:

            y = sqrt(E_s) * signal + n

        with:
            E_s = (P * tau) / L
            n ~ CN(0, N0)

        Parameters
        ----------
        signal : np.ndarray
            Aggregate transmitted tensor with shape:
                (num_time_slots, L, R)

            This tensor is interpreted as the structured resource activation
            BEFORE physical-layer amplitude/noise effects.

        Returns
        -------
        np.ndarray
            Complex-valued received tensor with the same shape as the input.
        """

        if self.mode == "clean":
            return self._clean(signal)

        elif self.mode == "awgn":
            return self._awgn(signal)

        else:
            raise ValueError(f"Unknown channel type: {self.mode}")

    def _clean(self, signal):
        """
        Clean channel (no additive noise).

        PRINCIPLE
        ---------
        The clean-channel output is still interpreted as the matched-filter
        output for the transmitted resource symbols:

            y = sqrt(E_s) * signal

        where:
            E_s = (P * tau) / L

        Parameters
        ----------
        signal : np.ndarray
            Input tensor with shape:
                (num_time_slots, L, R)

        Returns
        -------
        np.ndarray
            Complex-valued clean output tensor.
        """

        transmitted = self.signal_level * signal.astype(complex)
        return transmitted

    def _awgn(self, signal):
        """
        Apply complex AWGN at the matched-filter output.

        PRINCIPLE
        ---------
        This function enforces:

            SNR = P / (B * N0)

        through the centrally derived parameter:

            N0 = P / (B * SNR_linear)

        and uses the current matched-filter-output approximation:

            y = sqrt(E_s) * signal + n

        where:
            E_s = (P * tau) / L

        and:
            n ~ CN(0, N0)

        Complex-noise generation
        ------------------------
        If:
            n = n_I + j n_Q

        then:
            n_I ~ N(0, N0/2)
            n_Q ~ N(0, N0/2)

        so that:
            E[|n|^2] = N0

        Parameters
        ----------
        signal : np.ndarray
            Input tensor with shape:
                (num_time_slots, L, R)

        Returns
        -------
        np.ndarray
            Complex-valued noisy output tensor.

        Notes
        -----
        This is the correct place where the manuscript SNR affects the SFC:
        - not as the SFC channel SNR itself,
        - but through the derived N0, which sets the AWGN level.
        """

        transmitted = self.signal_level * signal.astype(complex)

        noise = np.sqrt(self.N0 / 2.0) * (
            np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
        )
        return transmitted + noise
