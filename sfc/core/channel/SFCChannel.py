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
This class uses a theorem-consistent fixed-map codebook.

A valid SFC map has shape:

    (L, R)

and exactly one active cell per row.

However, to guarantee unambiguous partially overlapping transmissions under the
orthogonal-vector construction, row resources are partitioned into L disjoint
subsets:

    D_1, D_2, ..., D_L

such that:

    D_i cap D_j = empty, for i != j
    D_1 union ... union D_L = D_perp

In the one-hot resource-vector implementation, D_perp is represented by the R
canonical binary vectors. Therefore, the R resource indices are partitioned into
L disjoint row-resource groups.

Each row l of each map is only allowed to activate a resource from group D_l.

Thus, the number of theorem-valid maps is:

    |D_1| * |D_2| * ... * |D_L|

For example, with R=12 and L=4, the balanced partition is:

    3, 3, 3, 3

and the maximum number of theorem-valid maps is:

    3^4 = 81

Therefore, configurations requiring more than 81 event IDs, e.g.

    2 * N * S = 2 * 5 * 12 = 120

are infeasible with R=12, L=4 under this theorem-consistent construction.

For S=12, N=5, L=4, one needs at least R=16, because:

    4^4 = 256 >= 120

Physical-power convention
-------------------------
This class does not directly scale power and does not add noise.

Physical scaling and AWGN are handled by:

    sfc/core/channel/physical_channel.py

using the matched-filter/resource-output SFC convention:

    y = sqrt(E_chip) * superposed + n

where:

    E_chip = P tau / (2 N L)

and the simulated noise uses the resolved noise parameter N0 directly at the
matched-filter/resource-output level:

    n ~ CN(0, N0)

This orchestrator must not generate noise from:

    B * N0

or:

    B_s * N0

and must not use SNR as an operational noise parameter. SNR remains a diagnostic
and channel-capacity quantity handled by system/theory helpers.

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
import math
from typing import Any, Dict, List

import numpy as np

from sfc.core.channel.collision import CollisionModel
from sfc.core.channel.detection import MapDetector
from sfc.core.channel.mapping import EventMapper
from sfc.core.channel.physical_channel import PhysicalChannel


class SFCChannel:
    """
    SFC channel orchestrator.

    This class coordinates mapping, collision/superposition, physical channel,
    detection, and inverse mapping. It intentionally delegates physical scaling
    and AWGN generation to PhysicalChannel.
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

        if self.R < self.L:
            raise ValueError(
                "The theorem-consistent SFC map construction requires R >= L, "
                f"got R={self.R}, L={self.L}."
            )

        # ------------------------------------------------------------------
        # Optional sensor-event association matrix.
        #
        # Expected shape when provided:
        #     (S, num_event_ids)
        #
        # In fair-methods pipelines, this is usually:
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
        # Theorem-consistent row-resource groups.
        #
        # These groups are fixed for the lifetime of the SFCChannel object.
        # ------------------------------------------------------------------
        self.row_resource_groups = self._build_row_resource_groups()
        self.total_theorem_valid_maps = self._count_theorem_valid_maps(
            self.row_resource_groups
        )

        # ------------------------------------------------------------------
        # Channel submodules.
        #
        # CollisionModel:
        #     dimensionless temporal/resource superposition.
        #
        # PhysicalChannel:
        #     SFC matched-filter/resource-output scaling and AWGN.
        #
        # MapDetector:
        #     detection of transmitted maps from physical channel output.
        # ------------------------------------------------------------------
        self.collision = CollisionModel(cfg)
        self.channel = PhysicalChannel(cfg)
        self.detector = MapDetector(cfg)

    # =========================================================================
    # MAIN CALL
    # =========================================================================

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

        if events.ndim != 2:
            raise ValueError(
                "events must have shape (event_slots_total, num_event_ids)."
            )

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
        #
        # PhysicalChannel is the only block responsible for:
        #   - SFC signal-level scaling;
        #   - AWGN generation;
        #   - N0-based physical noise convention.
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

    # =========================================================================
    # LAZY INITIALIZATION
    # =========================================================================

    def _initialize_if_needed(self, events):
        """
        Initialize fixed codebook and mapper only once.
        """
        if self.maps_library is not None and self.mapper is not None:
            return

        events = np.asarray(events, dtype=float)

        if events.ndim != 2:
            raise ValueError(
                "events must have shape (event_slots_total, num_event_ids)."
            )

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
            raise ValueError(
                "error in sensor_x_event map: No sensor_x_event map is set and "
                "a default identity(S) is invalid because num_event_ids != S. "
                "Please provide cfg['channel']['sensor_x_event'] explicitly."
            )

        return np.identity(self.S)

    # =========================================================================
    # THEOREM-CONSISTENT MAP GENERATION LOGIC
    # =========================================================================

    def _build_row_resource_groups(self) -> List[List[int]]:
        """
        Partition the R resources into L disjoint row-resource groups.

        The groups implement the theorem condition:

            D_i cap D_j = empty, for i != j
            D_1 union ... union D_L = D_perp

        In this implementation, D_perp is represented by the R one-hot
        canonical resource vectors.

        The partition is balanced:

        Example:
            R=12, L=4 -> [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]

            R=14, L=4 -> sizes [4,4,3,3]
        """
        base_size = self.R // self.L
        remainder = self.R % self.L

        groups = []
        start = 0

        for row in range(self.L):
            group_size = base_size + 1 if row < remainder else base_size
            stop = start + group_size
            groups.append(list(range(start, stop)))
            start = stop

        if start != self.R:
            raise RuntimeError(
                f"Internal partition error: consumed {start} resources, "
                f"but R={self.R}."
            )

        for row, group in enumerate(groups):
            if len(group) == 0:
                raise ValueError(
                    f"Invalid row-resource partition: row {row} received "
                    "an empty resource group."
                )

        return groups

    @staticmethod
    def _count_theorem_valid_maps(row_resource_groups: List[List[int]]) -> int:
        """
        Count the number of theorem-valid maps:

            product_l |D_l|
        """
        return int(math.prod(len(group) for group in row_resource_groups))

    def _generate_maps(self, num_event_ids):
        """
        Generate fixed theorem-consistent maps.

        Logic
        -----
        - build the full theorem-valid codebook;
        - each valid map has exactly one active cell per row;
        - row l may only use resources from the l-th row-resource group;
        - sample num_event_ids unique maps uniformly without replacement;
        - keep the selected assignment fixed for the life of the channel object.
        """
        codebook = self._build_valid_map_codebook()

        total_valid_maps = int(codebook.shape[0])

        if num_event_ids > total_valid_maps:
            raise ValueError(
                "Not enough unique theorem-valid SFC maps available. "
                f"Requested num_event_ids={num_event_ids}, but only "
                f"{total_valid_maps} theorem-valid maps exist for "
                f"R={self.R}, L={self.L}, row_resource_group_sizes="
                f"{[len(g) for g in self.row_resource_groups]}. "
                "Increase system.R, reduce system.S, reduce signal.N_override, "
                "or reduce system.L."
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

        invalid_or_duplicates = self._validate_maps(maps)

        if len(invalid_or_duplicates) > 0:
            raise ValueError(
                "Generated maps are invalid, duplicated, or violate "
                f"the row-resource partition: {invalid_or_duplicates}"
            )

        print(
            f"[INFO] Generated {num_event_ids} fixed theorem-valid maps "
            f"uniformly from {total_valid_maps} valid maps"
        )
        print(
            f"[INFO] SFC row-resource group sizes = "
            f"{[len(g) for g in self.row_resource_groups]}"
        )

        return maps

    def _build_valid_map_codebook(self):
        """
        Build the full theorem-valid codebook of SFC maps.

        A valid map has:

        - shape (L, R);
        - exactly one active cell per row;
        - row l only activates resources from D_l;
        - D_l are disjoint row-resource groups.

        Number of theorem-valid maps:

            |D_1| * |D_2| * ... * |D_L|

        This replaces the old R^L construction.
        """
        total_valid_maps = self.total_theorem_valid_maps

        row_choices_iterator = itertools.product(*self.row_resource_groups)

        codebook = np.zeros(
            (total_valid_maps, self.L, self.R),
            dtype=float,
        )

        for idx, row_choice_tuple in enumerate(row_choices_iterator):
            for row, col in enumerate(row_choice_tuple):
                codebook[idx, row, col] = 1.0

        return codebook

    # =========================================================================
    # MAP VALIDATION
    # =========================================================================

    def _validate_maps(self, maps):
        """
        Check invalid, duplicate, or theorem-inconsistent maps.

        A valid theorem-consistent map must:

        - have shape (L, R);
        - have exactly one active cell per row;
        - activate, in row l, only a resource from row_resource_groups[l];
        - be unique within the selected map set.

        Parameters
        ----------
        maps : np.ndarray
            Shape:

                (num_maps, L, R)

        Returns
        -------
        list
            List of validation problems.
        """
        maps = np.asarray(maps)

        if maps.ndim != 3:
            raise ValueError("maps must have shape (num_maps, L, R).")

        if maps.shape[1] != self.L or maps.shape[2] != self.R:
            raise ValueError(
                f"maps must have shape (num_maps, L, R)=(*, {self.L}, {self.R}), "
                f"got {maps.shape}."
            )

        problems = []

        row_resource_sets = [
            set(group) for group in self.row_resource_groups
        ]

        for i in range(maps.shape[0]):
            # --------------------------------------------------------------
            # Check exactly one active cell per row.
            # --------------------------------------------------------------
            row_sums = np.sum(maps[i], axis=1)

            if not np.all(row_sums == 1):
                problems.append(
                    {
                        "map_i": i,
                        "problem": "not_exactly_one_active_cell_per_row",
                        "row_sums": row_sums.tolist(),
                    }
                )
                continue

            # --------------------------------------------------------------
            # Check row-resource group consistency.
            # --------------------------------------------------------------
            for row in range(self.L):
                active_cols = np.where(maps[i, row, :] > 0)[0]

                if len(active_cols) != 1:
                    problems.append(
                        {
                            "map_i": i,
                            "row": row,
                            "problem": "row_not_one_hot",
                            "active_cols": active_cols.tolist(),
                        }
                    )
                    continue

                col = int(active_cols[0])

                if col not in row_resource_sets[row]:
                    problems.append(
                        {
                            "map_i": i,
                            "row": row,
                            "problem": "resource_not_in_row_group",
                            "col": col,
                            "allowed_group": sorted(row_resource_sets[row]),
                        }
                    )

            # --------------------------------------------------------------
            # Check duplicate maps.
            # --------------------------------------------------------------
            for j in range(i + 1, maps.shape[0]):
                if np.array_equal(maps[i], maps[j]):
                    problems.append(
                        {
                            "map_i": i,
                            "map_j": j,
                            "problem": "duplicate_map",
                        }
                    )

        return problems

    def _unique_maps(self, maps):
        """
        Backward-compatible wrapper.

        Historically this method checked only one-hot validity and duplicate
        maps. It now delegates to the theorem-consistent validator.
        """
        return self._validate_maps(maps)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

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
            "row_resource_groups": self.row_resource_groups,
            "row_resource_group_sizes": [
                len(group) for group in self.row_resource_groups
            ],
            "total_theorem_valid_maps": self.total_theorem_valid_maps,
        }

        if self.sensor_x_event is not None and len(self.sensor_x_event) != 0:
            diagnostics["sensor_x_event_shape"] = tuple(
                np.asarray(self.sensor_x_event).shape
            )

        if self.maps_library is not None:
            diagnostics["maps_library_shape"] = tuple(self.maps_library.shape)

        if hasattr(self.collision, "diagnostics"):
            diagnostics["collision"] = self.collision.diagnostics()

        if hasattr(self.channel, "diagnostics"):
            diagnostics["physical_channel"] = self.channel.diagnostics()

        if hasattr(self.detector, "diagnostics"):
            diagnostics["detector"] = self.detector.diagnostics()

        return diagnostics


__all__ = [
    "SFCChannel",
]
