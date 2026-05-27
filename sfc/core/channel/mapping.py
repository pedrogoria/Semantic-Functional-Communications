"""
sfc/core/channel/mapping.py

Event <-> Map conversion using a fixed codebook.

Design assumptions
------------------
- maps_library is fixed during the simulation;
- each event ID is associated with one map;
- sensor_x_event optionally defines which sensor is responsible for which event IDs.

Current logic
-------------
- events are represented as a binary matrix with shape:
      (num_time_slots, num_event_ids)

- maps_library has shape:
      (num_event_ids, L, R)

- each active event copies its associated fixed map into the transmitted tensor.

IMPORTANT
---------
This file does NOT generate maps.
It only converts:
    events -> maps
    maps_est -> events
"""

import numpy as np


class EventMapper:
    """
    Fixed-codebook event / map converter.
    """

    def __init__(self, maps_library, sensor_x_event=None):
        """
        Parameters
        ----------
        maps_library : np.ndarray
            Fixed maps with shape:
                (num_event_ids, L, R)

        sensor_x_event : np.ndarray or None, optional
            Sensor-event association matrix with shape:
                (S, num_event_ids)
        """

        self.maps_library = maps_library
        self.num_event_ids, self.L, self.R = maps_library.shape

        if sensor_x_event is None:
            self.sensor_x_event = []
            self.S = None
        else:
            self.sensor_x_event = sensor_x_event
            self.S = sensor_x_event.shape[0]

    # ======================================================================
    # EVENTS -> MAPS
    # ======================================================================

    def events_to_maps(self, events):
        """
        Convert event activations to transmitted maps.

        Parameters
        ----------
        events : np.ndarray
            Event matrix with shape:
                (num_time_slots, num_event_ids)

        Returns
        -------
        np.ndarray
            Map tensor with shape:
                (num_time_slots, num_event_ids, L, R)
        """

        assert len(events.shape) == 2, \
            "events must have shape (num_time_slots, num_event_ids)"

        num_time_slots, num_event_ids = events.shape

        assert num_event_ids == self.num_event_ids, \
            "events.shape[1] must match maps_library.shape[0]"

        maps = np.zeros((num_time_slots, num_event_ids, self.L, self.R))

        for t in range(num_time_slots):
            active_ids = np.where(events[t] == 1)[0]

            for event_id in active_ids:
                maps[t, event_id] = self.maps_library[event_id]

        return maps

    # ======================================================================
    # MAPS -> EVENTS
    # ======================================================================

    def maps_to_events(self, maps_est):
        """
        Recover event activations from estimated maps.

        Parameters
        ----------
        maps_est : np.ndarray
            Estimated map tensor.

            Supported shapes:
            - (num_time_slots, L, R)
            - (num_time_slots, num_event_ids, L, R)

        Returns
        -------
        np.ndarray
            Recovered event matrix with shape:
                (num_time_slots, num_event_ids)

        CURRENT LOGIC  ✅ NEW
        ---------------------
        Correlate each estimated map with the codebook.
        """

        # ------------------------------------------------------------------
        # Case A: one aggregate detected map per time slot
        # shape = (T, L, R)
        # ------------------------------------------------------------------
        if len(maps_est.shape) == 3:
            num_time_slots = maps_est.shape[0]
            events_est = np.zeros((num_time_slots, self.num_event_ids))

            for t in range(num_time_slots):
                rec = maps_est[t].flatten()

                for event_id in range(self.num_event_ids):
                    ref = self.maps_library[event_id].flatten()
                    score = np.dot(ref, rec)

                    if score > 0:
                        events_est[t, event_id] = 1

            return events_est

        # ------------------------------------------------------------------
        # Case B: one detected map hypothesis per event ID
        # shape = (T, E, L, R)
        # ------------------------------------------------------------------
        elif len(maps_est.shape) == 4:
            num_time_slots = maps_est.shape[0]
            events_est = np.zeros((num_time_slots, self.num_event_ids))

            for t in range(num_time_slots):
                for event_id in range(self.num_event_ids):
                    rec = maps_est[t, event_id].flatten()
                    ref = self.maps_library[event_id].flatten()

                    score = np.dot(ref, rec)

                    if score > 0:
                        events_est[t, event_id] = 1

            return events_est

        else:
            raise ValueError(
                "maps_est must have shape "
                "(num_time_slots, L, R) or (num_time_slots, num_event_ids, L, R)"
            )
