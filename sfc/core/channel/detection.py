"""
sfc/core/channel/detection.py

Detection block for SFC.

CURRENT ROLE
------------
Recover event-related map hypotheses from the received aggregate signal.

IMPORTANT
---------
The physical channel returns the complex matched-filter output.

Therefore detection is based on the magnitude of the received signal:

    abs(y) > threshold

PHYSICAL PRINCIPLE
------------------
The physical channel uses the manuscript-consistent relation:

    SNR = P / (B * N0)

with:
- P  : average transmit power per sensor
- B  : channel bandwidth
- N0 : noise parameter

The same system-level parameters are also used by the Benchmark model, but
for the SFC channel they determine the received symbol level and the noise.

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

    build_derived_system_parameters(cfg, threshold_factor=...)

so that the physical model remains consistent across:
- physical_channel.py
- detection.py
- pipelines
- future channel modules

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

Threshold rule
--------------
If cfg["channel"]["threshold"] is explicitly provided, use it.

Otherwise derive:

    threshold = default_threshold

where default_threshold is built centrally from:

    threshold_factor * signal_level
"""

import numpy as np

from sfc.core.system_parameters import build_derived_system_parameters


class MapDetector:
    """
    Recover maps from the received aggregate signal.
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
        The detector threshold is determined as follows:

        1. If cfg["channel"]["threshold"] is explicitly provided,
           use that value directly.

        2. Otherwise, derive the threshold centrally from:
               threshold_factor * signal_level
           where signal_level is built from the shared physical model.
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

        # Optional score threshold in map-score domain.
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
        #
        # If explicit threshold is not given, derive it centrally from:
        #     threshold_factor * signal_level
        # ------------------------------------------------------------------
        if "threshold" in det_cfg:
            self.threshold = det_cfg["threshold"]
        else:
            self.threshold = self.params.default_threshold

    def detect(self, y, reference_maps=None):
        """
        Detect maps from the received aggregate signal.

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
                estimated per-event maps, shape:
                    (num_time_slots, num_event_ids, L, R)
        """

        assert len(y.shape) == 3, \
            "y must have shape (num_time_slots, L, R)"

        # ------------------------------------------------------------------
        # Detect active resource cells based on magnitude of the complex
        # matched-filter output.
        # ------------------------------------------------------------------
        y_bin = (np.abs(y) > self.threshold).astype(float)

        if reference_maps is None:
            return y_bin

        num_time_slots = y_bin.shape[0]
        num_event_ids = reference_maps.shape[0]
        L = reference_maps.shape[1]

        score_threshold = self.score_threshold if self.score_threshold is not None else L

        maps_est = np.zeros(
            (num_time_slots, num_event_ids, y_bin.shape[1], y_bin.shape[2])
        )

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
