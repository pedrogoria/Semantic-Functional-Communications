"""
sfc/core/mapping.py

Trusted core module for SFC transmission mapping.

This file isolates the mapping logic originally implemented inside the
SFC channel class. It preserves exactly:

- map generation (tx_maps)
- uniqueness verification (unique_maps)

IMPORTANT
---------
This is a direct structural extraction of trusted code.

NO changes in:
- indexing logic
- iteration structure
- combinatorial generation process

Allowed changes:
- encapsulation into a dedicated class
- comments and documentation
- minor structuring for readability
- compatibility helpers for allocation and sensor-event assignment

Forbidden changes:
- modifying the mapping algorithm
- simplifying loops
- altering index update rules
"""

import numpy as np


class SFCMapping:
    """
    Generate and validate SFC transmission maps.

    This class preserves the original combinatorial mapping structure used
    for event encoding in the SFC system.

    Attributes
    ----------
    maps : np.ndarray or None
        Array of shape (num_events, n_sub_symbol, resource), or None if the
        map tensor has not yet been allocated.

    n_sub_symbol : int
        Number of sub-symbols (L).

    resource : int
        Number of available resources (R).

    sensor_x_event : np.ndarray or list
        Mapping between sensors and events. This is not modified by the map
        generation algorithm, but its presence is required exactly as in the
        original implementation.

    map_class : str
        Compatibility attribute preserved from earlier refactoring choices.
    """

    def __init__(self, num_events=None, n_sub_symbol=6, resource=7, sensor_x_event=None):
        """
        Initialize the mapping structure.

        Parameters
        ----------
        num_events : int or None, optional
            Number of events (IDs) to be mapped. If None, the map tensor is not
            allocated at initialization time and must later be created through
            `allocate_maps(...)`.

        n_sub_symbol : int, optional
            Number of sub-symbols (L).

        resource : int, optional
            Number of resource units (R).

        sensor_x_event : np.ndarray or None, optional
            Optional sensor-event association matrix.
        """

        self.n_sub_symbol = n_sub_symbol
        self.resource = resource

        if sensor_x_event is None:
            self.sensor_x_event = []
        else:
            self.sensor_x_event = sensor_x_event

        # ------------------------------------------------------------------
        # Preserve a compatibility attribute that may be useful for future
        # integration, even though the trusted tx_maps logic is deterministic.
        # ------------------------------------------------------------------
        self.map_class = 'random'

        # ------------------------------------------------------------------
        # Allocate maps only if the number of events is already known.
        # Otherwise, the caller may allocate later through allocate_maps(...).
        # ------------------------------------------------------------------
        if num_events is None:
            self.maps = None
        else:
            self.maps = np.zeros((num_events, n_sub_symbol, resource))

    # ======================================================================
    # COMPATIBILITY HELPERS
    # ======================================================================

    def set_sensor_event_map(self, sensor_x_event):
        """
        Set or replace the sensor-event association matrix.

        Parameters
        ----------
        sensor_x_event : np.ndarray
            Sensor-to-event mapping matrix.

        Notes
        -----
        This helper does not change any scientific logic. It only provides a
        clean way to configure the mapping object before calling tx_maps().
        """

        self.sensor_x_event = sensor_x_event

    def allocate_maps(self, num_events):
        """
        Allocate the map tensor.

        Parameters
        ----------
        num_events : int
            Number of event IDs to be represented.

        Returns
        -------
        np.ndarray
            Allocated map tensor with shape:
                (num_events, n_sub_symbol, resource)

        Notes
        -----
        This helper does not change the trusted mapping logic. It only makes
        allocation explicit and compatible with staged initialization.
        """

        self.maps = np.zeros((num_events, self.n_sub_symbol, self.resource))
        return self.maps

    # ======================================================================
    # UNIQUE MAP CHECK (EXACT LOGIC)
    # ======================================================================

    def unique_maps(self):
        """
        Check for invalid or duplicate maps.

        Returns
        -------
        list of tuple
            List of index pairs indicating:
            - invalid maps (rows that do not sum to 1)
            - duplicate maps

        Notes
        -----
        Exact preservation of original logic:

        - Each row must have exactly one "1"
        - No two maps should be identical
        """

        if self.maps is None:
            raise AssertionError('error in maps: No map tensor is allocated')

        x = []

        for i in range(0, self.maps.shape[0]):

            # --------------------------------------------------------------
            # Check whether each row of the current map sums exactly to 1.
            # This reproduces the original validity criterion.
            # --------------------------------------------------------------
            if not np.all(np.sum(self.maps[i], axis=1) == 1):
                x.append((i, i))

            else:
                # ----------------------------------------------------------
                # Compare against all following maps to detect duplicates.
                # This preserves the original nested-loop structure exactly.
                # ----------------------------------------------------------
                for j in range(i + 1, self.maps.shape[0]):
                    if np.array(self.maps[i] == self.maps[j]).all():
                        x.append((i, j))

        return x

    # ======================================================================
    # MAP GENERATION (EXACT ALGORITHM)
    # ======================================================================

    def tx_maps(self):
        """
        Generate transmission maps exactly as in the original implementation.

        Returns
        -------
        list
            Output of unique_maps() for validation.

        Notes
        -----
        This method implements a combinatorial enumeration of resource indices.

        The logic is preserved EXACTLY, including:

        - definition of L_aux
        - definition of groupSize
        - nested update logic using carry mechanism
        - uniqueness warning condition
        """

        assert len(self.sensor_x_event) != 0, \
            'error in sensor_x_event map: No sensor_x_event map is set'

        if self.maps is None:
            raise AssertionError('error in maps: No map tensor is allocated')

        # --------------------------------------------------------------
        # Initial index selection (L_aux), preserved exactly.
        # Note:
        # np.floor(...) is intentionally kept, preserving the original
        # floating-step behavior before casting indices with int(...).
        # --------------------------------------------------------------
        L_aux = np.arange(
            0,
            self.resource,
            np.floor(self.resource / self.n_sub_symbol)
        )

        # --------------------------------------------------------------
        # Group-size definition, preserved exactly.
        # --------------------------------------------------------------
        groupSize = np.arange(
            np.floor(self.resource / self.n_sub_symbol) - 1,
            self.resource,
            np.floor(self.resource / self.n_sub_symbol)
        )

        groupSize[-1] = self.resource - 1

        # --------------------------------------------------------------
        # Capacity check (unchanged behavior).
        # This reproduces the original warning condition exactly.
        # --------------------------------------------------------------
        if self.maps.shape[0] > (
            (np.floor(self.resource / self.n_sub_symbol) ** (self.n_sub_symbol - 1)) *
            (groupSize[-1] - groupSize[-2])
        ):
            print('Not every ID is unique')

        # --------------------------------------------------------------
        # Main combinatorial assignment loop, preserved exactly.
        # --------------------------------------------------------------
        for index in range(self.maps.shape[0]):

            # ----------------------------------------------------------
            # Assign one active resource position per row.
            # ----------------------------------------------------------
            for index1 in range(self.maps.shape[1]):
                self.maps[index, index1, int(L_aux[index1])] = 1

            # ----------------------------------------------------------
            # Carry mechanism for the next combination.
            # This is the exact logic used in the trusted code.
            # ----------------------------------------------------------
            aux = 0
            L_aux[aux] = L_aux[aux] + 1

            while L_aux[aux] > groupSize[aux]:

                L_aux[aux] = L_aux[aux] - np.floor(self.resource / self.n_sub_symbol)
                aux = aux + 1

                if aux < self.n_sub_symbol:
                    L_aux[aux] = L_aux[aux] + 1
                else:
                    L_aux = np.arange(
                        0,
                        self.resource,
                        np.floor(self.resource / self.n_sub_symbol)
                    )
                    aux = 0

        # --------------------------------------------------------------
        # Return uniqueness check exactly as in the original behavior.
        # --------------------------------------------------------------
        return self.unique_maps()


__all__ = ["SFCMapping"]