"""
sfc/core/channel/detection.py

Detection block for SFC.

CURRENT ROLE
------------
Recover event-related map hypotheses from the received aggregate signal.

IMPORTANT
---------
The physical channel returns the complex matched-filter output.

Therefore detection must be based on the magnitude of the received signal:

    abs(y) > threshold

PHYSICAL PRINCIPLE
------------------
The physical channel uses:

    SNR = P / (B * N0)

and the matched-filter-output approximation:

    y = sqrt(E_s) * signal + n

with:

    E_s = P / B

Therefore the received pulse amplitude is not 1 in general.
The threshold must be consistent with:

    sqrt(E_s)

CURRENT THRESHOLD RULE
----------------------
If cfg["channel"]["threshold"] is explicitly provided, use it.

Otherwise derive:

    threshold = threshold_factor * sqrt(E_s)

with:
    threshold_factor = cfg["channel"].get("threshold_factor", 0.5)

This ensures that clean-channel pulses are detectable even when P/B < 1.

Detection modes
---------------
1. strict:
       score == L

2. loose:
       score > 0

3. threshold:
       score >= score_threshold

Default behavior
----------------
If score_threshold is not provided, then:
    score_threshold = L
"""

import numpy as np


class MapDetector:
    """
    Recover maps from received aggregate signal.
    """

    def __init__(self, cfg):
        """
        Initialize detector configuration.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        """

        det_cfg = cfg.get("channel", {})
        sys_cfg = cfg["system"]

        # ------------------------------------------------------------------
        # Physical parameters needed to derive the default threshold
        # ------------------------------------------------------------------
        self.P = sys_cfg["P"]
        self.B = sys_cfg["B"]

        # matched-filter symbol energy
        self.E_s = self.P / self.B
        self.signal_level = np.sqrt(self.E_s)

        # ------------------------------------------------------------------
        # Threshold configuration
        #
        # If explicit threshold is not given, derive it from sqrt(E_s)
        # ------------------------------------------------------------------
        self.threshold_factor = det_cfg.get("threshold_factor", 0.5)

        if "threshold" in det_cfg:
            self.threshold = det_cfg["threshold"]
        else:
            self.threshold = self.threshold_factor * self.signal_level

        # ------------------------------------------------------------------
        # Detection mode:
        # - strict
        # - loose
        # - threshold
        # ------------------------------------------------------------------
        self.mode = det_cfg.get("detection_mode", "threshold")

        # optional score threshold; if absent, use L at runtime
        self.score_threshold = det_cfg.get("score_threshold", None)

    def detect(self, y, reference_maps=None):
        """
        Detect maps from received aggregate signal.

        Parameters
        ----------
        y : np.ndarray
            Shape:
                (num_time_slots, L, R)

            May be complex-valued.

        reference_maps : np.ndarray or None, optional
            Fixed codebook with shape:
                (num_event_ids, L, R)

        Returns
        -------
        np.ndarray
            If reference_maps is None:
                binary aggregate maps, shape (num_time_slots, L, R)

            If reference_maps is provided:
                estimated per-event maps, shape (num_time_slots, num_event_ids, L, R)
        """

        assert len(y.shape) == 3, \
            "y must have shape (num_time_slots, L, R)"

        # ------------------------------------------------------------------
        # Detect active resource cells based on the magnitude of the received
        # matched-filter output.
        # ------------------------------------------------------------------
        y_bin = (np.abs(y) > self.threshold).astype(float)

        if reference_maps is None:
            return y_bin

        num_time_slots = y_bin.shape[0]
        num_event_ids = reference_maps.shape[0]
        L = reference_maps.shape[1]

        score_threshold = self.score_threshold if self.score_threshold is not None else L

        maps_est = np.zeros((num_time_slots, num_event_ids, y_bin.shape[1], y_bin.shape[2]))

        for t in range(num_time_slots):
            rec = y_bin[t]

            for event_id in range(num_event_ids):
                ref = reference_maps[event_id]

                score = np.sum(rec * ref)

                if self.mode == "strict":
                    if score == L:
                        maps_est[t, event_id] = ref

                elif self.mode == "loose":
                    if score > 0:
                        maps_est[t, event_id] = ref

                elif self.mode == "threshold":
                    if score >= score_threshold:
                        maps_est[t, event_id] = ref

                else:
                    raise ValueError(f"Unknown detection mode: {self.mode}")

        return maps_est
