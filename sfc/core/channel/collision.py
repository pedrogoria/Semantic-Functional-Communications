"""
sfc/core/channel/collision.py

Collision / superposition model for SFC.

CURRENT LOGIC  ✅ NEW
--------------------
Input maps shape:
    (num_time_slots, num_event_ids, L, R)

Output shape:
    (num_time_slots, L, R)

This module combines simultaneous event maps before the physical channel.
"""

import numpy as np


class CollisionModel:
    """
    Multi-event superposition model.
    """

    def __init__(self, cfg):
        self.mode = cfg.get("channel", {}).get("collision_mode", "sum")

    def apply(self, maps):
        """
        Apply collision / superposition.

        Parameters
        ----------
        maps : np.ndarray
            Shape:
                (num_time_slots, num_event_ids, L, R)

        Returns
        -------
        np.ndarray
            Shape:
                (num_time_slots, L, R)
        """

        assert len(maps.shape) == 4, \
            "maps must have shape (num_time_slots, num_event_ids, L, R)"

        # ------------------------------------------------------------------
        # Mode 1: linear superposition
        # ------------------------------------------------------------------
        if self.mode == "sum":
            return np.sum(maps, axis=1)

        # ------------------------------------------------------------------
        # Mode 2: binary occupancy
        # If multiple maps activate the same cell, clip to 1
        # ------------------------------------------------------------------
        elif self.mode == "binary":
            y = np.sum(maps, axis=1)
            y[y > 1] = 1
            return y

        else:
            raise ValueError(f"Unknown collision mode: {self.mode}")
