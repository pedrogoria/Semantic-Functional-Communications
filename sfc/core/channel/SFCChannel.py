"""
sfc/core/channel/SFCChannel.py

Orchestrator for SFC channel.

Pipeline:

events → maps → channel → maps_est → events_est

Key design decisions:
- maps are FIXED per sensor (do not change during simulation)
- mapping is deterministic once initialized
- modular structure (mapping, collision, channel, detection)
"""

import numpy as np

from sfc.core.channel.mapping import EventMapper
from sfc.core.channel.physical_channel import PhysicalChannel
from sfc.core.channel.detection import MapDetector
from sfc.core.channel.collision import CollisionModel   # ✅ NEW MODULE


class SFCChannel:
    """
    SFC communication channel.

    Responsibilities:
    - convert events to maps
    - apply collision + superposition
    - apply physical channel
    - detect maps
    - recover events
    """

    # ---------------------------------------------------------------------
    # INIT
    # ---------------------------------------------------------------------
    def __init__(self, cfg):

        self.cfg = cfg

        # system parameters
        self.S = cfg["system"]["S"]
        self.R = cfg["system"]["R"]
        self.L = cfg["system"]["L"]

        # ------------------------------------------------------------
        # MODULES
        # ------------------------------------------------------------
        self.mapper = EventMapper(cfg)
        self.collision = CollisionModel(cfg)         # ✅ NEW
        self.channel = PhysicalChannel(cfg)
        self.detector = MapDetector(cfg)

        # ------------------------------------------------------------
        # FIXED MAPS PER SENSOR
        # ------------------------------------------------------------
        self.maps_library = self._generate_maps()

    # ---------------------------------------------------------------------
    # MAP GENERATION (FIXED)
    # ---------------------------------------------------------------------
    def _generate_maps(self):
        """
        Generate fixed maps per sensor.

        IMPORTANT:
        - each sensor has a unique map
        - maps DO NOT change during simulation

        Current implementation:
        - random binary patterns (future: structured codes)

        Returns:
            maps_library: (S, R, L)
        """

        rng = np.random.default_rng(self.cfg["reproducibility"]["seed"])

        maps = rng.integers(0, 2, size=(self.S, self.R, self.L))

        print("[INFO] Maps generated (fixed per sensor)")

        return maps

    # ---------------------------------------------------------------------
    # MAIN CALL
    # ---------------------------------------------------------------------
    def __call__(self, events):
        """
        Run full SFC pipeline.

        Parameters
        ----------
        events : np.ndarray
            Shape: (S, slots)

        Returns
        -------
        events_est : np.ndarray
            Estimated events after transmission
        """

        # ------------------------------------------------------------
        # EVENTS → MAPS
        # ------------------------------------------------------------
        maps = self._events_to_maps(events)

        # ------------------------------------------------------------
        # COLLISION + SUPERPOSITION
        # ------------------------------------------------------------
        maps_tx = self.collision.apply(maps)

        # ------------------------------------------------------------
        # PHYSICAL CHANNEL
        # ------------------------------------------------------------
        y = self.channel.transmit(maps_tx)

        # ------------------------------------------------------------
        # DETECTION
        # ------------------------------------------------------------
        maps_est = self.detector.detect(y)

        # ------------------------------------------------------------
        # MAPS → EVENTS
        # ------------------------------------------------------------
        events_est = self._maps_to_events(maps_est)

        return events_est

    # ---------------------------------------------------------------------
    # INTERNAL: EVENTS → MAPS
    # ---------------------------------------------------------------------
    def _events_to_maps(self, events):
        """
        Apply fixed map of each sensor when event occurs.

        Logic:
        - if event(s, k) == 1 → add map of sensor s at instant k

        Output:
            maps: (S, R, L)
        """

        S, slots = events.shape

        maps = np.zeros((S, self.R, self.L))

        for s in range(S):
            for k in range(slots):
                if events[s, k] == 1:
                    maps[s] += self.maps_library[s]   # ✅ FIXED MAP

        return maps

    # ---------------------------------------------------------------------
    # INTERNAL: MAPS → EVENTS
    # ---------------------------------------------------------------------
    def _maps_to_events(self, maps_est):
        """
        Recover events from estimated maps.

        Current logic:
        - if energy in map > threshold → event = 1

        NOTE:
        This is a simplified logic.
        Future:
        - pattern matching
        - correlation detection
        """

        S, R, L = maps_est.shape

        events_est = np.zeros((S, R * L))

        for s in range(S):
            energy = np.sum(maps_est[s])

            if energy > 0:
                events_est[s, :] = 1   # ✅ simple approximation

        return events_est
