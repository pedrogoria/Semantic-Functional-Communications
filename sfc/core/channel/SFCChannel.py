"""
sfc/core/channel/SFCChannel.py

Orchestrator for the Semantic-Functional Communication (SFC) channel.

Pipeline
--------
events -> maps -> channel -> maps_est -> events_est

IMPORTANT
---------
This class uses the NEW map-generation logic explicitly authorized:

NEW MAP LOGIC
-------------
- each map has exactly one '1' per row;
- map assignment to event IDs is uniformly distributed;
- once generated, the map assigned to an event ID never changes.

The rest of the architecture is modular:
- mapping
- collision / superposition
- physical channel
- detection
"""

import itertools
import numpy as np

from sfc.core.channel.mapping import EventMapper
from sfc.core.channel.collision import CollisionModel
from sfc.core.channel.physical_channel import PhysicalChannel
from sfc.core.channel.detection import MapDetector


class SFCChannel:
    """
    SFC channel orchestrator.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # ------------------------------------------------------------------
        # System dimensions
        # ------------------------------------------------------------------
        self.S = cfg["system"]["S"]   # number of sensors / users
        self.R = cfg["system"]["R"]   # number of resources / columns
        self.L = cfg["system"]["L"]   # number of rows / temporal resources

        # ------------------------------------------------------------------
        # Optional sensor-event association matrix
        # ------------------------------------------------------------------
        self.sensor_x_event = cfg.get("channel", {}).get("sensor_x_event", [])

        # ------------------------------------------------------------------
        # Lazy state
        # ------------------------------------------------------------------
        self.maps_library = None
        self.num_event_ids = None
        self.mapper = None

        # ------------------------------------------------------------------
        # Channel submodules
        # ------------------------------------------------------------------
        self.collision = CollisionModel(cfg)
        self.channel = PhysicalChannel(cfg)
        self.detector = MapDetector(cfg)

    # ======================================================================
    # MAIN CALL
    # ======================================================================

    def __call__(self, events, return_intermediates=False):
        """
        Run the full SFC pipeline.

        Parameters
        ----------
        events : np.ndarray
            Shape:
                (num_time_slots, num_event_ids)

        return_intermediates : bool, optional
            If True, also return intermediate tensors.

        Returns
        -------
        np.ndarray or dict
        """

        self._initialize_if_needed(events)

        # EVENTS -> MAPS
        maps_tx = self.mapper.events_to_maps(events)

        # COLLISION / SUPERPOSITION
        superposed = self.collision.apply(maps_tx)

        # PHYSICAL CHANNEL
        y = self.channel.transmit(superposed)

        # DETECTION
        maps_est = self.detector.detect(y, reference_maps=self.maps_library)

        # MAPS -> EVENTS
        events_est = self.mapper.maps_to_events(maps_est)

        if return_intermediates:
            return {
                "events": events,
                "maps_tx": maps_tx,
                "superposed": superposed,
                "y": y,
                "maps_est": maps_est,
                "events_est": events_est,
            }

        return events_est

    # ======================================================================
    # LAZY INITIALIZATION
    # ======================================================================

    def _initialize_if_needed(self, events):
        """
        Initialize fixed codebook and mapper only once.
        """

        if self.maps_library is not None and self.mapper is not None:
            return

        assert len(events.shape) == 2, \
            "events must have shape (num_time_slots, num_event_ids)"

        self.num_event_ids = events.shape[1]

        if len(self.sensor_x_event) == 0:
            self.sensor_x_event = self._build_default_sensor_x_event(self.num_event_ids)

        self.maps_library = self._generate_maps(self.num_event_ids)

        self.mapper = EventMapper(
            self.maps_library,
            sensor_x_event=self.sensor_x_event
        )

        print("[INFO] SFCChannel initialized with fixed maps library")

    def _build_default_sensor_x_event(self, num_event_ids):
        """
        Build default sensor-event association when valid.
        """

        if num_event_ids != self.S:
            raise AssertionError(
                "error in sensor_x_event map: No sensor_x_event map is set and "
                "a default identity(S) is invalid because num_event_ids != S. "
                "Please provide cfg['channel']['sensor_x_event'] explicitly."
            )

        return np.identity(self.S)

    # ======================================================================
    # NEW MAP GENERATION LOGIC
    # ======================================================================

    def _generate_maps(self, num_event_ids):
        """
        Generate fixed maps with the NEW authorized logic:

        - exactly one '1' per row;
        - unique maps per event ID;
        - uniform assignment across the valid codebook.
        """

        codebook = self._build_valid_map_codebook()

        total_valid_maps = codebook.shape[0]

        if num_event_ids > total_valid_maps:
            raise AssertionError(
                f"Not enough unique valid maps available. "
                f"Requested num_event_ids={num_event_ids}, "
                f"but only {total_valid_maps} valid maps exist for R={self.R}, L={self.L}."
            )

        rng = np.random.default_rng(self.cfg["reproducibility"]["seed"])
        selected_indices = rng.choice(total_valid_maps, size=num_event_ids, replace=False)

        maps = codebook[selected_indices]

        invalid_or_duplicates = self._unique_maps(maps)

        if len(invalid_or_duplicates) > 0:
            raise AssertionError(
                f"Generated maps are invalid or duplicated: {invalid_or_duplicates}"
            )

        print(
            f"[INFO] Generated {num_event_ids} fixed maps uniformly from "
            f"{total_valid_maps} valid maps"
        )

        return maps

    def _build_valid_map_codebook(self):
        """
        Build the full codebook of valid maps.

        A valid map has:
        - shape (L, R)
        - exactly one '1' per row
        """

        all_row_choices = itertools.product(range(self.R), repeat=self.L)

        codebook = np.zeros((self.R ** self.L, self.L, self.R), dtype=float)

        for idx, row_choice_tuple in enumerate(all_row_choices):
            for row, col in enumerate(row_choice_tuple):
                codebook[idx, row, col] = 1.0

        return codebook

    # ======================================================================
    # MAP VALIDATION
    # ======================================================================

    def _unique_maps(self, maps):
        """
        Check invalid or duplicate maps.

        A valid map must have exactly one '1' per row.
        """

        x = []

        for i in range(0, maps.shape[0]):
            if not np.all(np.sum(maps[i], axis=1) == 1):
                x.append((i, i))
            else:
                for j in range(i + 1, maps.shape[0]):
                    if np.array(maps[i] == maps[j]).all():
                        x.append((i, j))

        return x
