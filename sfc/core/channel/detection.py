"""
sfc/core/channel/detection.py

Detection block for SFC.

CURRENT ROLE
------------
Recover event-related map hypotheses from the final received channel frame.

IMPORTANT
---------
The physical channel now returns the final complex-valued channel frame with
shape:

    y.shape = (rx_slots_total, R)

where:
- rx_slots_total = event_slots_total + L - 1
- R is the number of sub-carriers/resources

Therefore detection must be based on:
1. the magnitude of the received frame:
       abs(y)
2. sliding windows of length L:
       y[t0:t0+L, :]
   for each possible event-start slot t0

PHYSICAL PRINCIPLE
------------------
The physical channel uses the manuscript-consistent relation:

    SNR = P / (B * N0)

and the matched-filter-output approximation:

    y = sqrt(E_s) * signal + n

with:
    E_s = (P * tau) / L

The default threshold is derived centrally through:

    build_derived_system_parameters(cfg)

so that the detector remains consistent with:
- physical_channel.py
- the system-level power/bandwidth model

DETECTION MODES
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

INPUT / OUTPUT CONVENTION
-------------------------
Input:
    y : (rx_slots_total, R)

reference_maps:
    (num_event_ids, L, R)

Output:
    maps_est : (event_slots_total, num_event_ids, L, R)

where:
    event_slots_total = rx_slots_total - L + 1

This output format is compatible with:
    EventMapper.maps_to_events(maps_est)
"""

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

        Notes
        -----
        Threshold logic:
        - if cfg["channel"]["threshold"] is explicitly provided, use it
        - otherwise use the centralized default threshold derived from:
              threshold_factor * signal_level
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
        self.threshold_factor = det_cfg.get("threshold_factor", 0.5)

        self.params = build_derived_system_parameters(
            cfg,
            threshold_factor=self.threshold_factor
        )

        # ------------------------------------------------------------------
        # Threshold configuration
        # ------------------------------------------------------------------
        if "threshold" in det_cfg:
            self.threshold = det_cfg["threshold"]
        else:
            self.threshold = self.params.default_threshold

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
        - extract the L-row window:
              rec_win = y_bin[t0:t0+L, :]
        - compare it to each reference map
        - decide according to the chosen detection mode
        """

        assert len(y.shape) == 2, \
            "y must have shape (rx_slots_total, R)"

        # ------------------------------------------------------------------
        # Threshold the magnitude of the complex matched-filter output
        # ------------------------------------------------------------------
        y_bin = (np.abs(y) > self.threshold).astype(float)

        if reference_maps is None:
            return y_bin

        # reference_maps shape: (num_event_ids, L, R)
        assert len(reference_maps.shape) == 3, \
            "reference_maps must have shape (num_event_ids, L, R)"

        num_event_ids, L, R = reference_maps.shape

        assert y_bin.shape[1] == R, \
            "y.shape[1] must match reference_maps.shape[2]"

        rx_slots_total = y_bin.shape[0]
        event_slots_total = rx_slots_total - L + 1

        if event_slots_total <= 0:
            raise ValueError(
                "Invalid dimensions: rx_slots_total - L + 1 must be positive"
            )

        score_threshold = self.score_threshold if self.score_threshold is not None else L

        maps_est = np.zeros((event_slots_total, num_event_ids, L, R))

        for t0 in range(event_slots_total):
            rec_win = y_bin[t0:t0 + L, :]   # shape: (L, R)

            for event_id in range(num_event_ids):
                ref = reference_maps[event_id]

                score = np.sum(rec_win * ref)

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
