"""
sfc/core/channel/collision.py

Collision / superposition model, :] += map_of_that_eventCollision / superposition model for SFC.

This implements the manuscript-consistent temporal placement of SFC maps.

Physical-power convention
-------------------------
This module does not apply physical amplitude scaling and does not add noise.

The output frame is a dimensionless resource-time occupancy/superposition
frame. The physical channel later maps this frame to the matched-filter /
resource-output level using:

    y = sqrt(E_chip) * frame + n

where:

    E_chip = P tau / (2 N L)

This separation keeps this module responsible only for structural collision and
keeps physical energy/noise modeling inside physical_channel.py.

Supported collision modes
-------------------------
1. sum:
       Linear superposition. Colliding active chips add.

2. binary:
       Occupancy clipping after superposition. Any value larger than one is
       clipped to one.

For physically meaningful amplitude/power accounting, "sum" is the preferred
mode because it preserves collision multiplicity before physical scaling.
"""

from __future__ import annotations

import numpy as np


class CollisionModel:
    """
    Multi-event temporal placement and superposition model.

    Current supported modes
    -----------------------
    - "sum"    : linear sum in the final channel frame
    - "binary" : occupancy clipping after superposition
    """

    def __init__(self, cfg):
        """
        Initialize collision model.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        """

        self.cfg = cfg
        self.mode = cfg.get("channel", {}).get("collision_mode", "sum")

    def apply(self, maps):
        """
        Apply temporal placement and collision/superposition.

        Parameters
        ----------
        maps : np.ndarray
            Map tensor with shape:

                (event_slots_total, num_event_ids, L, R)

            Each nonzero map at maps[t0, event_id] is interpreted as an event
            starting at slot t0. Its L rows are placed into the final received
            frame over rows:

                t0, t0+1, ..., t0+L-1

        Returns
        -------
        np.ndarray
            Final dimensionless channel frame with shape:

                (event_slots_total + L - 1, R)
        """

        maps = np.asarray(maps)

        assert len(maps.shape) == 4, \
            "maps must have shape (event_slots_total, num_event_ids, L, R)"

        event_slots_total, num_event_ids, L, R = maps.shape

        if event_slots_total < 1:
            raise ValueError("event_slots_total must be >= 1.")

        if num_event_ids < 1:
            raise ValueError("num_event_ids must be >= 1.")

        if L < 1:
            raise ValueError("L must be >= 1.")

        if R < 1:
            raise ValueError("R must be >= 1.")

        # ------------------------------------------------------------------
        # Build final channel frame.
        # ------------------------------------------------------------------
        frame = np.zeros(
            (event_slots_total + L - 1, R),
            dtype=maps.dtype,
        )

        # ------------------------------------------------------------------
        # Temporal placement:
        #
        # if an event starts at slot t0 and its map has L rows, then it
        # occupies:
        #
        #   frame[t0:t0+L, :]
        #
        # in the final received frame.
        # ------------------------------------------------------------------
        for t0 in range(event_slots_total):
            for event_id in range(num_event_ids):
                map_e = maps[t0, event_id]

                # Skip empty maps quickly.
                if np.any(map_e):
                    frame[t0:t0 + L, :] += map_e

        # ------------------------------------------------------------------
        # Collision / superposition mode.
        # ------------------------------------------------------------------
        if self.mode == "sum":
            return frame

        if self.mode == "binary":
            frame_bin = np.array(frame, copy=True)
            frame_bin[frame_bin > 1] = 1
            return frame_bin

        raise ValueError(f"Unknown collision mode: {self.mode}")

    def diagnostics(self):
        """
        Return collision-model diagnostics.
        """

        return {
            "mode": self.mode,
        }
