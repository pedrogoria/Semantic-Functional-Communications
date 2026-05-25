"""
sfc/core/channel.py

Trusted core implementation of the Semantic-Functional Communication (SFC) channel.

This module integrates:
- positioning dynamics (PositionSensorNodes)
- propagation model (PropagationModel)
- SFC mapping (SFCMapping)
- transmission + reception logic

IMPORTANT
---------
This is a STRICT structural refactor of the trusted implementation.

NO mathematical or algorithmic changes were introduced.

Preserved exactly:
- signal construction
- looping structure
- noise model
- detection rule
- thresholding
- map usage

Only changes:
- modularization
- external dependencies isolated into core modules
- improved documentation
- explicit synchronization with the mapping object
"""

import numpy as np

from sfc.core.positioning import PositionSensorNodes
from sfc.core.propagation import PropagationModel
from sfc.core.mapping import SFCMapping


class SFCChannel(PositionSensorNodes):
    """
    Semantic-Functional Communication channel.

    This class preserves the exact behavior of the original implementation,
    while delegating:
        - propagation -> PropagationModel
        - mapping -> SFCMapping
        - positioning -> PositionSensorNodes

    The core transmission and detection logic remains unchanged.
    """

    def __init__(
        self,
        base_station=np.array([[0, 0]]),
        sensor_nodes=64,
        carrier_frequency=2.4e9,
        n_sub_symbol=6,
        resource=7,
        **options
    ):
        """
        Initialize the SFC channel.

        Parameters
        ----------
        base_station : np.ndarray, optional
            Base-station position. If the shape is not (1, 2), the trusted
            fallback behavior is to replace it with [[0, 0]].

        sensor_nodes : int, optional
            Number of sensor nodes.

        carrier_frequency : float, optional
            Carrier frequency in Hz.

        n_sub_symbol : int, optional
            Number of rows in the SFC transmission matrix.

        resource : int, optional
            Number of columns/resources in the SFC transmission matrix.

        **options : dict
            Additional trusted options used by the original implementation.
        """

        # ------------------------------------------------------------------
        # BASIC PARAMETERS (preserved)
        # ------------------------------------------------------------------
        self.name = options.pop('name', 'Generic')
        self.base_station = base_station

        self.carrier_frequency = options.pop('carrier_frequency', carrier_frequency)
        self.path_loss_exp = options.pop('path_loss_exp', 0)

        self.bandwidth = options.pop('bandwidth', 1000)
        self.resource = resource
        self.n_sub_symbol = n_sub_symbol

        # ------------------------------------------------------------------
        # TIME SLOT DEFINITION (preserved)
        # ------------------------------------------------------------------
        self.energy_slot_time = 2 / (self.bandwidth / resource)

        # ------------------------------------------------------------------
        # MOVEMENT TYPE (preserved)
        # ------------------------------------------------------------------
        self.sn_movement_type = options.pop('sn_movement_type', 'module')

        # ------------------------------------------------------------------
        # SENSOR / EVENT STRUCTURE
        # ------------------------------------------------------------------
        self.sensor_nodes = sensor_nodes
        self.sensor_x_event = options.pop('sensor_x_event', [])

        # ------------------------------------------------------------------
        # NOISE POWER
        # ------------------------------------------------------------------
        self.N0 = options.pop('N0', 0)

        # ------------------------------------------------------------------
        # OPTIONAL MOVEMENT STD (preserved)
        # ------------------------------------------------------------------
        if self.sn_movement_type == 'gaussian':
            self.sn_mov_std = options.pop('sn_mov_std', 10)

        # ------------------------------------------------------------------
        # INITIALIZE POSITIONING (unchanged inheritance logic)
        # ------------------------------------------------------------------
        super().__init__(
            sensor_nodes=sensor_nodes,
            energy_slot_time=self.energy_slot_time,
            sn_movement_type=self.sn_movement_type,
            **options
        )

        # ------------------------------------------------------------------
        # BASE STATION SHAPE FIX (preserved behavior)
        # ------------------------------------------------------------------
        if self.base_station.shape != (1, 2):
            self.base_station = np.array([[0, 0]])

        # ------------------------------------------------------------------
        # PROPAGATION MODEL
        # ------------------------------------------------------------------
        self.propagation = PropagationModel(
            base_station=self.base_station,
            carrier_frequency=self.carrier_frequency,
            path_loss_exp=self.path_loss_exp
        )

        self.path_loss = self.propagation.get_impulse(self.sn_positions)

        # ------------------------------------------------------------------
        # TRANSMISSION AMPLITUDE (preserved)
        # ------------------------------------------------------------------
        self.tx_amplitude = options.pop('tx_amplitude', 10 * np.ones((sensor_nodes, 1)))

        if self.tx_amplitude.shape != (sensor_nodes, 1):
            self.tx_amplitude = 10 * np.ones((sensor_nodes, 1))
            self.dc_threshold = options.pop('dc_threshold', 5)

        if options.pop('power_at_receiver', False):
            self.tx_amplitude = self.tx_amplitude / np.abs(self.path_loss)

        # ------------------------------------------------------------------
        # THRESHOLD (preserved)
        # ------------------------------------------------------------------
        self.dc_threshold = options.pop(
            'dc_threshold',
            0.5 * np.min(np.sqrt(self.tx_amplitude * np.abs(self.path_loss)))
        )

        # ------------------------------------------------------------------
        # MAPPING MODULE
        # ------------------------------------------------------------------
        self.mapping = SFCMapping(
            n_sub_symbol=self.n_sub_symbol,
            resource=self.resource,
            sensor_x_event=self.sensor_x_event
        )

    # ======================================================================
    # INTERNAL SYNCHRONIZATION
    # ======================================================================

    def _sync_mapping(self):
        """
        Synchronize the mapping object with the channel-level sensor-event map.

        Why this exists
        ----------------
        The channel stores `self.sensor_x_event`, and the mapping object also
        stores a copy/reference of that information. To avoid any accidental
        divergence between the two objects, this helper explicitly synchronizes
        them before map generation.

        Important
        ---------
        This does NOT change the scientific behavior of the model.
        It only ensures internal consistency between composed objects.
        """

        self.mapping.set_sensor_event_map(self.sensor_x_event)

    # ======================================================================
    # MAIN CHANNEL OPERATION
    # ======================================================================

    def __call__(self, events, **options):
        """
        Execute transmission and detection.

        This method preserves EXACTLY the original logic.

        Parameters
        ----------
        events : np.ndarray
            Event matrix with shape:
                (N, num_events)

        **options : dict
            Optional trusted parameters:
            - sensor_x_event
            - update
            - n_steps

        Returns
        -------
        tuple
            (rx_events, rx_map, received_signal)
        """

        # ------------------------------------------------------------------
        # Setup sensor-event mapping if not provided.
        # This preserves the original behavior:
        # use the identity mapping when no sensor_x_event was previously set.
        # ------------------------------------------------------------------
        if len(self.sensor_x_event) == 0:
            self.sensor_x_event = options.pop(
                'sensor_x_event',
                np.identity(self.sensor_nodes)
            )

        # ------------------------------------------------------------------
        # Keep the mapping object synchronized with the current sensor-event map.
        # This is a structural consistency improvement only.
        # ------------------------------------------------------------------
        self._sync_mapping()

        # ------------------------------------------------------------------
        # Ensure maps exist.
        # This preserves the original behavior of allocating and generating maps
        # only when they are not yet available.
        # ------------------------------------------------------------------
        if not hasattr(self.mapping, "maps") or self.mapping.maps is None:
            self.mapping.allocate_maps(events.shape[1])
            self.mapping.tx_maps()

        maps = self.mapping.maps

        # ------------------------------------------------------------------
        # ASSERTIONS (unchanged)
        # ------------------------------------------------------------------
        assert self.sensor_x_event.shape[1] == events.shape[1], 'error in parameter shape: events'
        assert maps.shape[0] == events.shape[1], 'error in parameter shape: events'

        N = events.shape[0]

        received_signal = np.complex_(np.zeros((N + self.n_sub_symbol - 1, self.resource)))

        # ------------------------------------------------------------------
        # TRANSMISSION LOOP (EXACT)
        # ------------------------------------------------------------------
        for e in range(0, N):
            inds = np.argwhere(events[e] == 1)

            for ind in inds:
                inds_sensor = np.argwhere(self.sensor_x_event[:, ind])[:, 0]

                for ind_sensor in inds_sensor:
                    received_signal[e:e + self.n_sub_symbol, :] = (
                        received_signal[e:e + self.n_sub_symbol, :]
                        + maps[ind]
                        * self.path_loss[ind_sensor]
                        * np.sqrt(self.tx_amplitude[ind_sensor])
                    )

        # ------------------------------------------------------------------
        # ADD NOISE (EXACT)
        # ------------------------------------------------------------------
        received_signal = received_signal + np.random.normal(
            0, np.sqrt(0.5 * self.N0), size=received_signal.shape
        )

        received_signal = received_signal + 1j * np.random.normal(
            0, np.sqrt(0.5 * self.N0), size=received_signal.shape
        )

        received_signal = np.abs(received_signal)

        # ------------------------------------------------------------------
        # THRESHOLDING (EXACT)
        # ------------------------------------------------------------------
        rx_map = np.zeros(received_signal.shape)

        inds = np.argwhere(received_signal > self.dc_threshold)
        rx_map[inds[:, 0], inds[:, 1]] = 1

        # ------------------------------------------------------------------
        # DETECTION (EXACT)
        # ------------------------------------------------------------------
        rx_events = np.zeros(events.shape)

        for e in range(0, N):
            for i in range(0, events.shape[1]):
                rx_events[e, i] = 1 if np.sum(
                    rx_map[e:e + self.n_sub_symbol, :] * maps[i, :, :]
                ) == self.n_sub_symbol else 0

        # ------------------------------------------------------------------
        # UPDATE DYNAMICS (optional)
        # ------------------------------------------------------------------
        if options.pop('update', False):
            self.update(n_steps=options.pop('n_steps', 1))

        return rx_events, rx_map, received_signal

    # ======================================================================
    # STATE UPDATE
    # ======================================================================

    def update(self, n_steps=1, **options):
        """
        Update sensor positions and recompute propagation coefficients.

        Parameters
        ----------
        n_steps : int or float, optional
            Number of movement steps.

        **options : dict
            Additional arguments accepted for compatibility. They are not used
            here because the trusted implementation also did not use them.
        """

        self.move_nodes(n_steps=n_steps)
        self.path_loss = self.propagation.get_impulse(self.sn_positions)


__all__ = ["SFCChannel"]
