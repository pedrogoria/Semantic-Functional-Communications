"""
sfc/core/channel/SFCChannel.py

Orchestrator for the Semantic-Functional Communication (SFC) channel.

Pipeline
--------
events -> maps -> collision/superposition -> physical channel -> maps_est -> events_est

Module responsibilities
-----------------------
This class orchestrates the SFC channel blocks:

1. mapping:
       events -> maps_tx

2. collision / temporal superposition:
       maps_tx -> superposed

3. physical channel:
       superposed -> y

4. detection:
       y -> maps_est

5. inverse mapping:
       maps_est -> events_est

Map-generation logic
--------------------
This class uses the authorized fixed-map codebook logic:

- each valid map has shape (L, R);
- each valid map has exactly one active cell per row;
- map assignment to event IDs is uniformly sampled from the valid codebook;
- once generated, the map assigned to an event ID remains fixed.

Physical-power convention
-------------------------
This class does not directly scale power and does not add noise.

Physical scaling is handled by:

    sfc/core/channel/physical_channel.py

using the matched-filter/resource-output SFC convention:

    y = sqrt(E_chip) * superposed + n

where:

    E_chip = P tau / (2 N L)

The collision module returns a dimensionless resource-time superposition frame.
The physical channel maps that dimensionless frame to the physical
matched-filter output level.

Input / output
--------------
Input:

    events.shape = (event_slots_total, num_event_ids)

Output if return_intermediates=False:

    events_est.shape = (event_slots_total, num_event_ids)

Output if return_intermediates=True:

    dict with:
        events
        maps_tx
        superposed
        y
        maps_est
        events_est
        diagnostics
"""

from __future__ import annotations

import itertools
from typing import Any, Dict

import numpy as np

from sfc.core.channel.mapping import EventMapper
from sfc.core.channel.collision import CollisionModel
from sfc.core.channel.physical_channel import PhysicalChannel
from sfc.core.channel.detection import MapDetector


class SFCChannel:
    """
    SFC channel orchestrator.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the SFC channel orchestrator.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        """

        self.cfg = cfg

        # ------------------------------------------------------------------
        # System dimensions.
        # ------------------------------------------------------------------
        self.S = int(cfg["system"]["S"])
        self.R = int(cfg["system"]["R"])
        self.L = int(cfg["system"]["L"])

        if self.S < 1:
            raise ValueError("system.S must be >= 1.")

        if self.R < 1:
            raise ValueError("system.R must be >= 1.")

        if self.L < 1:
            raise ValueError("system.L must be >= 1.")

        # ------------------------------------------------------------------
        # Optional sensor-event association matrix.
        #
        # Expected shape when provided:
        #     (S, num_event_ids)
        #
        # In the fair-methods pipeline, this is usually:
        #     sensor s owns event IDs [2sN, ..., 2(s+1)N - 1]
        # ------------------------------------------------------------------
        self.sensor_x_event = cfg.get("channel", {}).get("sensor_x_event", [])

        # ------------------------------------------------------------------
        # Lazy state.
        # ------------------------------------------------------------------
        self.maps_library = None
        self.num_event_ids = None
        self.mapper = None

        # ------------------------------------------------------------------
        # Channel submodules.
        # ------------------------------------------------------------------
        self.collision = CollisionModel(cfg)
        self.channel = PhysicalChannel(cfg)
        self.detector = MapDetector(cfg)

    # ======================================================================
    # MAIN CALL
    # ======================================================================

    def __call__(self, events, return_intermediates: bool = False):
        """
        Run the full SFC channel pipeline.

        Parameters
        ----------
        events : np.ndarray
            Event matrix with shape:

                (event_slots_total, num_event_ids)

        return_intermediates : bool, optional
            If True, return a dictionary containing intermediate tensors and
            diagnostics. If False, return only events_est.

        Returns
        -------
        np.ndarray or dict
            If return_intermediates is False:

                events_est

            If return_intermediates is True:

                {
                    "events": events,
                    "maps_tx": maps_tx,
                    "superposed": superposed,
                    "y": y,
                    "maps_est": maps_est,
                    "events_est": events_est,
                    "diagnostics": diagnostics,
                }
        """

        events = np.asarray(events, dtype=float)

        assert len(events.shape) == 2, \
            "events must have shape (event_slots_total, num_event_ids)"

        self._initialize_if_needed(events)

        # ------------------------------------------------------------------
        # EVENTS -> MAPS
        # ------------------------------------------------------------------
        maps_tx = self.mapper.events_to_maps(events)

        # ------------------------------------------------------------------
        # COLLISION / TEMPORAL SUPERPOSITION
        # ------------------------------------------------------------------
        superposed = self.collision.apply(maps_tx)

        # ------------------------------------------------------------------
        # PHYSICAL CHANNEL
        # ------------------------------------------------------------------
        y = self.channel.transmit(superposed)

        # ------------------------------------------------------------------
        # DETECTION
        # ------------------------------------------------------------------
        maps_est = self.detector.detect(
            y,
            reference_maps=self.maps_library,
        )

        # ------------------------------------------------------------------
        # MAPS -> EVENTS
        # ------------------------------------------------------------------
        events_est = self.mapper.maps_to_events(maps_est)

        if return_intermediates:
            return {
                "events": events,
                "maps_tx": maps_tx,
                "superposed": superposed,
                "y": y,
                "maps_est": maps_est,
                "events_est": events_est,
                "diagnostics": self.diagnostics(),
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
            "events must have shape (event_slots_total, num_event_ids)"

        self.num_event_ids = int(events.shape[1])

        if self.num_event_ids < 1:
            raise ValueError("num_event_ids must be >= 1.")

        if len(self.sensor_x_event) == 0:
            self.sensor_x_event = self._build_default_sensor_x_event(
                self.num_event_ids
            )
        else:
            self.sensor_x_event = np.asarray(self.sensor_x_event, dtype=float)

            if self.sensor_x_event.shape != (self.S, self.num_event_ids):
                raise ValueError(
                    "sensor_x_event must have shape "
                    f"(S, num_event_ids)=({self.S}, {self.num_event_ids}), "
                    f"got {self.sensor_x_event.shape}."
                )

        self.maps_library = self._generate_maps(self.num_event_ids)

        self.mapper = EventMapper(
            self.maps_library,
            sensor_x_event=self.sensor_x_event,
        )

        print("[INFO] SFCChannel initialized with fixed maps library")

    def _build_default_sensor_x_event(self, num_event_ids):
        """
        Build default sensor-event association when valid.

        The default identity association is only valid when:

            num_event_ids == S

        For the usual SFC Fourier/phase representation, num_event_ids is
        generally:

            2 N S

        and therefore a sensor_x_event matrix must be provided explicitly.
        """

        if num_event_ids != self.S:
            raise AssertionError(
                "error in sensor_x_event map: No sensor_x_event map is set and "
                "a default identity(S) is invalid because num_event_ids != S. "
                "Please provide cfg['channel']['sensor_x_event'] explicitly."
            )

        return np.identity(self.S)

    # ======================================================================
    # MAP GENERATION LOGIC
    # ======================================================================

    def _generate_maps(self, num_event_ids):
        """
        Generate fixed maps.

        Logic
        -----
        - build the full valid codebook;
        - each valid map has exactly one active cell per row;
        - sample num_event_ids unique maps uniformly without replacement;
        - keep the selected assignment fixed for the life of the channel object.
        """

        codebook = self._build_valid_map_codebook()

        total_valid_maps = int(codebook.shape[0])

        if num_event_ids > total_valid_maps:
            raise AssertionError(
                f"Not enough unique valid maps available. "
                f"Requested num_event_ids={num_event_ids}, "
                f"but only {total_valid_maps} valid maps exist "
                f"for R={self.R}, L={self.L}."
            )

        seed = self.cfg.get("reproducibility", {}).get(
            "seed",
            self.cfg.get("monte_carlo", {}).get("seed", 12345),
        )

        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(
            total_valid_maps,
            size=num_event_ids,
            replace=False,
        )

        maps = codebook[selected_indices]

        invalid_or_duplicates = self._unique_maps(maps)

        if len(invalid_or_duplicates) > 0:
            raise AssertionError(
                f"Generated maps are invalid or duplicated: "
                f"{invalid_or_duplicates}"
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

        - shape (L, R);
        - exactly one active cell per row.

        Number of valid maps:

            R^L
        """

        total_valid_maps = self.R ** self.L

        all_row_choices = itertools.product(
            range(self.R),
            repeat=self.L,
        )

        codebook = np.zeros(
            (total_valid_maps, self.L, self.R),
            dtype=float,
        )

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

        A valid map must have exactly one active cell per row.

        Parameters
        ----------
        maps : np.ndarray
            Shape:

                (num_maps, L, R)

        Returns
        -------
        list[tuple[int, int]]
            List of invalid or duplicate map index pairs.
        """

        maps = np.asarray(maps)

        if len(maps.shape) != 3:
            raise ValueError("maps must have shape (num_maps, L, R).")

        invalid_or_duplicates = []

        for i in range(maps.shape[0]):
            if not np.all(np.sum(maps[i], axis=1) == 1):
                invalid_or_duplicates.append((i, i))
                continue

            for j in range(i + 1, maps.shape[0]):
                if np.array_equal(maps[i], maps[j]):
                    invalid_or_duplicates.append((i, j))

        return invalid_or_duplicates

    # ======================================================================
    # DIAGNOSTICS
    # ======================================================================

    def diagnostics(self):
        """
        Return diagnostics from the orchestrator and submodules.
        """

        diagnostics = {
            "S": self.S,
            "R": self.R,
            "L": self.L,
            "num_event_ids": self.num_event_ids,
            "maps_library_initialized": self.maps_library is not None,
            "mapper_initialized": self.mapper is not None,
        }

        if self.sensor_x_event is not None and len(self.sensor_x_event) != 0:
            diagnostics["sensor_x_event_shape"] = tuple(
                np.asarray(self.sensor_x_event).shape
            )

        if hasattr(self.collision, "diagnostics"):
            diagnostics["collision"] = self.collision.diagnostics()

        if hasattr(self.channel, "diagnostics"):
            diagnostics["physical_channel"] = self.channel.diagnostics()

        if hasattr(self.detector, "diagnostics"):
            diagnostics["detector"] = self.detector.diagnostics()

        return diagnostics
