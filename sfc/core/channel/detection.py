"""
sfc/core/channel/detection.py

Detection block for SFC.

CurrentCurrent role
---------------
1. strict:
       score == L

2. loose:
       score > 0

3. threshold:
       score >= score_threshold

Here, score is the number of active reference-map cells that are detected in
the binarized received window.

Default score behavior
----------------------
If cfg["channel"]["score_threshold"] is not provided, then:

    score_threshold = L

This means the default threshold-mode behavior requires all L active chips of a
map to be detected.
"""

from __future__ import annotations

import numpy as np

from sfc.core.system_parameters import build_derived_system_parameters


class MapDetector:
    """
    Recover event-related map hypotheses from the final received channel frame.
    """

    def __init__(self, cfg):
        """
        Initialize detector configuration.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        """

        self.cfg = cfg
        det_cfg = cfg.get("channel", {})

        # ------------------------------------------------------------------
        # Detection mode:
        # - strict
        # - loose
        # - threshold
        # ------------------------------------------------------------------
        self.mode = det_cfg.get("detection_mode", "threshold")

        # Optional score threshold in map-matching domain.
        # If absent, use L at runtime.
        self.score_threshold = det_cfg.get("score_threshold", None)

        # ------------------------------------------------------------------
        # Centralized physical/system-derived parameters
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
        # Prefer explicit signal.N_override when available because SFC phase
        # and event representation in the pipelines often fixes N this way.
        # Otherwise, use the centrally derived params.N.
        # ------------------------------------------------------------------
        signal_cfg = cfg.get("signal", {})

        if signal_cfg.get("N_override", None) is not None:
            self.N = int(signal_cfg["N_override"])
        else:
            self.N = int(self.params.N)

        if self.N < 1:
            raise ValueError("MapDetector requires N >= 1.")

        if self.L < 1:
            raise ValueError("MapDetector requires L >= 1.")

        if self.P <= 0:
            raise ValueError("MapDetector requires P > 0.")

        if self.tau <= 0:
            raise ValueError("MapDetector requires tau > 0.")

        if self.N0 <= 0:
            raise ValueError("MapDetector requires N0 > 0.")

        # ------------------------------------------------------------------
        # SFC matched-filter/resource-output energy normalization.
        #
        # Same convention as physical_channel.py:
        #
        #   E_sensor = P * tau
        #   events per sensor = 2N
        #   active chips per sensor = 2NL
        #   E_chip = P*tau/(2NL)
        #   sfc_signal_level = sqrt(E_chip)
        # ------------------------------------------------------------------
        self.E_sensor = self.P * self.tau
        self.num_events_per_sensor = 2 * self.N
        self.num_active_chips_per_sensor = self.num_events_per_sensor * self.L

        self.E_event = self.E_sensor / self.num_events_per_sensor
        self.E_chip = self.E_sensor / self.num_active_chips_per_sensor

        self.sfc_signal_level = float(np.sqrt(self.E_chip))

        # Manuscript physical pulse amplitude, kept for diagnostics.
        self.sfc_pulse_amplitude = float(
            np.sqrt((self.tau * self.P * self.B) / (4.0 * self.L * self.R * self.N))
        )

        # ------------------------------------------------------------------
        # Threshold configuration.
        #
        # If explicit threshold is provided, use it exactly.
        # Otherwise use:
        #
        #   threshold = threshold_factor * sqrt(E_chip)
        #
        # This matches the new physical_channel.py scaling.
        # ------------------------------------------------------------------
        self.threshold_factor = float(det_cfg.get("threshold_factor", 0.5))

        if "threshold" in det_cfg:
            self.threshold = float(det_cfg["threshold"])
            self.threshold_source = "explicit"
        else:
            self.threshold = float(self.threshold_factor * self.sfc_signal_level)
            self.threshold_source = "threshold_factor_times_sfc_signal_level"

    def detect(self, y, reference_maps=None):
        """
        Detect maps from the final received channel frame.

        Parameters
        ----------
        y : np.ndarray
            Final received channel frame with shape:

                (rx_slots_total, R)

            May be complex-valued.

        reference_maps : np.ndarray or None, optional
            Fixed codebook with shape:

                (num_event_ids, L, R)

        Returns
        -------
        np.ndarray
            If reference_maps is None:

                binary thresholded frame with shape:
                    (rx_slots_total, R)

            If reference_maps is provided:

                estimated per-event maps with shape:
                    (event_slots_total, num_event_ids, L, R)

        Detection logic
        ---------------
        For each possible event-start slot t0:

        1. Threshold the received magnitude:

               y_bin = abs(y) > threshold

        2. Extract an L-row window:

               rec_win = y_bin[t0:t0+L, :]

        3. Compare the window with each reference map.

        4. Compute:

               score = sum(rec_win * reference_map)

           Since each valid reference map has exactly L active cells, score is
           the number of detected active chips for that candidate map.

        5. Accept or reject according to detection mode.
        """

        y = np.asarray(y)

        assert len(y.shape) == 2, \
            "y must have shape (rx_slots_total, R)"

        # ------------------------------------------------------------------
        # Threshold the magnitude of the complex matched-filter output.
        # ------------------------------------------------------------------
        y_bin = (np.abs(y) > self.threshold).astype(float)

        if reference_maps is None:
            return y_bin

        reference_maps = np.asarray(reference_maps, dtype=float)

        assert len(reference_maps.shape) == 3, \
            "reference_maps must have shape (num_event_ids, L, R)"

        num_event_ids, L, R = reference_maps.shape

        assert y_bin.shape[1] == R, \
            "y.shape[1] must match reference_maps.shape[2]"

        rx_slots_total = y_bin.shape[0]
        event_slots_total = rx_slots_total - L + 1

        if event_slots_total <= 0:
            raise ValueError(
                "Invalid dimensions: rx_slots_total - L + 1 must be positive."
            )

        score_threshold = (
            int(self.score_threshold)
            if self.score_threshold is not None
            else int(L)
        )

        maps_est = np.zeros(
            (event_slots_total, num_event_ids, L, R),
            dtype=float,
        )

        for t0 in range(event_slots_total):
            rec_win = y_bin[t0:t0 + L, :]  # shape: (L, R)

            for event_id in range(num_event_ids):
                ref = reference_maps[event_id]

                score = float(np.sum(rec_win * ref))

                if self.mode == "strict":
                    if score == L:
                        maps_est[t0, event_id] = ref

                elif self.mode == "loose":
                    if score > 0:
                        maps_est[t0, event_id] = ref

                elif self.mode == "threshold":
                    if score >= score_threshold:
                        maps_est[t0, event_id] = ref

                else:
                    raise ValueError(f"Unknown detection mode: {self.mode}")

        return maps_est

    def diagnostics(self):
        """
        Return detector diagnostic quantities.
        """

        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "threshold_factor": self.threshold_factor,
            "score_threshold": self.score_threshold,
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
        }
