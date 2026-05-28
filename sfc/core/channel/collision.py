"""
sfc/core/channel/collision.py

Collision / superposition model for SFC.

IMPORTANT
---------
✅ NEW LOGIC
-----------
This module now performs the temporal placement of each event map into the
final channel frame.

Input shape
-----------
maps:
    (event_slots_total, num_event_ids, L, R)

Interpretation:
- axis 0 = possible event-start slot
- axis 1 = event ID
- axis 2 = internal row of the map (temporal structure)
- axis 3 = sub-carrier index

Output shape
------------
frame:
    (event_slots_total + L - 1, R)

Interpretation:
- axis 0 = received discrete-time slot index
- axis 1 = sub-carrier index

For each event starting at slot t0:
    frame[t0:t0+L, :] += map_of_that_event

This is the correct manuscript-consistent temporal superposition.
"""

import numpy as np


class CollisionModel:
    """
    Multi-event superposition model.

    Current supported modes
    -----------------------
    - "sum"    : linear sum in the final channel frame
    - "binary" : occupancy clipping after superposition
    """

    def __init__(self, cfg):
        self.mode = cfg.get("channel", {}).get("collision_mode", "sum")

    def apply(self, maps):
        """
        Apply temporal placement + collision/superposition.

        Parameters
        ----------
        maps : np.ndarray
            Shape:
                (event_slots_total, num_event_ids, L, R)

        Returns
        -------
        np.ndarray
            Final channel frame with shape:
                (event_slots_total + L - 1, R)
        """

        assert len(maps.shape) == 4, \
            "maps must have shape (event_slots_total, num_event_ids, L, R)"

        event_slots_total, num_event_ids, L, R = maps.shape

        # ------------------------------------------------------------------
        # Build final channel frame
        # ------------------------------------------------------------------
        frame = np.zeros((event_slots_total + L - 1, R), dtype=maps.dtype)

        # ------------------------------------------------------------------
        # Temporal placement:
        # if an event starts at slot t0 and its map has L rows, then it
        # occupies the rows:
        #   t0, t0+1, ..., t0+L-1
        # in the final received frame.
        # ------------------------------------------------------------------
        for t0 in range(event_slots_total):
            for event_id in range(num_event_ids):
                map_e = maps[t0, event_id]

                # Skip empty maps quickly
                if np.any(map_e):
                    frame[t0:t0 + L, :] += map_e

        # ------------------------------------------------------------------
        # Collision / superposition mode
        # ------------------------------------------------------------------
        if self.mode == "sum":
            return frame

        elif self.mode == "binary":
            frame_bin = np.copy(frame)
            frame_bin[frame_bin > 1] = 1
            return frame_bin

        else:
            raise ValueError(f"Unknown collision mode: {self.mode}")
