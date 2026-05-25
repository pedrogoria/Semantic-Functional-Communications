"""
sfc/core/positioning.py

Trusted core module for sensor-node positioning dynamics.

This file contains ONLY the logic related to:
- initial placement of sensor nodes;
- initial velocity assignment;
- position updates according to the selected movement model.

IMPORTANT
---------
This module is a structural extraction of the original trusted code.
The behavior must remain scientifically equivalent to the original implementation.

Allowed changes:
- file split
- code organization
- comments
- helper methods

Forbidden changes:
- changing formulas
- changing random-generation logic
- changing branching behavior
- changing update rules
"""

import numpy as np


class PositionSensorNodes:
    """
    Manage sensor-node positions and movement dynamics.

    This class preserves the original behavior for:
    - node positioning at initialization;
    - velocity definition;
    - movement update across time steps.

    Notes
    -----
    The original implementation mixes several responsibilities inside one class.
    Here, the same behavior is preserved, but the code is organized into small
    helper methods to make the logic easier to audit and maintain.

    The public behavior should remain the same.
    """

    def __init__(
            self,
            sensor_nodes=64,
            n_moves=0,
            random_method='circumference',
            random_state=None,
            uniform_low=-300,
            uniform_high=300,
            standard_deviation=70,
            energy_slot_time=500 * 10 ** -6,
            **ops
    ):
        """
        Initialize the positioning model for sensor nodes.

        Parameters
        ----------
        sensor_nodes : int, optional
            Total number of sensor nodes in the scenario.

        n_moves : int, optional
            Number of nodes that are allowed to move.
            If n_moves > sensor_nodes, it is clipped to sensor_nodes.

        random_method : str, optional
            Initial placement method for the nodes.
            Supported values:
            - 'uniform'
            - 'gaussian'
            - anything else defaults to the circumference-based placement

        random_state : int or None, optional
            Seed passed to NumPy's random generator through np.random.seed().
            The original code always calls np.random.seed(random_state),
            including when random_state is None. This exact behavior is preserved.

        uniform_low : float, optional
            Lower bound for uniform initial placement.

        uniform_high : float, optional
            Upper bound for uniform initial placement.

        standard_deviation : float, optional
            Standard deviation used in Gaussian initial placement.

        energy_slot_time : float, optional
            Time scaling used in the position update rule.

        **ops : dict
            Extra configuration parameters.
            Supported keys preserved from original code:
            - 'sn_movement_type'
            - 'radius'
            - 'sc_velocity'
            - 'sc_velocity_scale'
        """

        # ------------------------------------------------------------------
        # Preserve the exact seeding behavior from the original implementation.
        # The original code calls np.random.seed(random_state) directly,
        # even when random_state is None.
        # ------------------------------------------------------------------
        np.random.seed(random_state)

        # ------------------------------------------------------------------
        # Store time scaling used when updating positions.
        # ------------------------------------------------------------------
        self.energy_slot_time = energy_slot_time

        # ------------------------------------------------------------------
        # Store the total number of sensor nodes.
        # ------------------------------------------------------------------
        self.sensor_nodes = sensor_nodes

        # ------------------------------------------------------------------
        # Store the default movement type.
        # If not supplied, keep the original default: 'not random'.
        # ------------------------------------------------------------------
        self.sn_movement_type = ops.pop('sn_movement_type', 'not random')

        # ------------------------------------------------------------------
        # Clip the number of moving nodes to sensor_nodes, exactly as before.
        # ------------------------------------------------------------------
        self.n_moves = n_moves if n_moves <= sensor_nodes else sensor_nodes

        # ------------------------------------------------------------------
        # Initialize positions using the same rules as the original code.
        # ------------------------------------------------------------------
        self.sn_positions = self._initialize_positions(
            random_method=random_method,
            uniform_low=uniform_low,
            uniform_high=uniform_high,
            standard_deviation=standard_deviation,
            ops=ops,
        )

        # ------------------------------------------------------------------
        # Initialize the default velocity using the same original rule:
        # either use 'sc_velocity' from ops, or generate a Gaussian velocity
        # with standard deviation 'sc_velocity_scale' (default = 10).
        # ------------------------------------------------------------------
        self.sn_velocity = ops.pop(
            'sc_velocity',
            np.random.normal(
                loc=0,
                scale=ops.pop('sc_velocity_scale', 10),
                size=[self.n_moves, 2]
            )
        )

    # ======================================================================
    # Initialization helpers
    # ======================================================================

    def _initialize_positions(
            self,
            random_method,
            uniform_low,
            uniform_high,
            standard_deviation,
            ops
    ):
        """
        Initialize node positions using the exact placement logic
        from the original implementation.

        Parameters
        ----------
        random_method : str
            Placement method selector.

        uniform_low : float
            Lower bound for uniform placement.

        uniform_high : float
            Upper bound for uniform placement.

        standard_deviation : float
            Standard deviation for Gaussian placement.

        ops : dict
            Additional options dictionary. The key 'radius' may be consumed.

        Returns
        -------
        np.ndarray
            Array of shape (sensor_nodes, 2) containing x/y positions.
        """

        if random_method == 'uniform':
            # --------------------------------------------------------------
            # Uniform placement in a square region.
            # Behavior preserved exactly.
            # --------------------------------------------------------------
            return np.random.uniform(
                low=uniform_low,
                high=uniform_high,
                size=[self.sensor_nodes, 2]
            )

        elif random_method == 'gaussian':
            # --------------------------------------------------------------
            # Gaussian placement centered at zero.
            # Behavior preserved exactly.
            # --------------------------------------------------------------
            return np.random.normal(
                loc=0,
                scale=standard_deviation,
                size=[self.sensor_nodes, 2]
            )

        else:
            # --------------------------------------------------------------
            # Default placement on a circumference.
            # Behavior preserved exactly:
            # - consume 'radius' from ops with default = 50
            # - angles go from -pi to pi, endpoint=False
            # - x = r*sin(angle), y = r*cos(angle)
            # --------------------------------------------------------------
            r = ops.pop('radius', 50)
            ang = np.linspace(-np.pi, np.pi, self.sensor_nodes, endpoint=False)
            return np.array([r * np.sin(ang), r * np.cos(ang)]).transpose()

    # ======================================================================
    # Movement helpers
    # ======================================================================

    def _truncate_or_pad_velocity(self, velocity):
        """
        Adjust the velocity array so that it has exactly self.n_moves rows,
        preserving the original logic.

        Original behavior:
        - if len(velocity) > self.n_moves: truncate
        - if len(velocity) < self.n_moves: pad with np.random.random(...)
        - if equal: keep as is

        Parameters
        ----------
        velocity : np.ndarray
            Input velocity array.

        Returns
        -------
        np.ndarray
            Velocity array with exactly self.n_moves rows.
        """

        if len(velocity) > self.n_moves:
            velocity = velocity[:self.n_moves]

        if len(velocity) < self.n_moves:
            velocity = np.concatenate(
                (velocity, np.random.random((self.n_moves - len(velocity), 2)))
            )

        return velocity

    def _append_static_nodes(self, velocity):
        """
        Append zero velocities for nodes that are not moving.

        Original behavior:
        velocity = concatenate((velocity, zeros((sensor_nodes - n_moves, 2))))

        Parameters
        ----------
        velocity : np.ndarray
            Velocity array for moving nodes only.

        Returns
        -------
        np.ndarray
            Full velocity array of shape (sensor_nodes, 2).
        """

        return np.concatenate(
            (velocity, np.zeros((self.sensor_nodes - self.n_moves, 2)))
        )

    def _direction_movement_velocity(self, velocity):
        """
        Build the velocity used in the 'direction' movement mode.

        This method preserves the exact original sequence:
        1. adjust length of velocity array;
        2. compute the vector norm using sqrt(diag(dot(v, v.T)));
        3. compute the angle using np.angle(vx + j*vy);
        4. multiply the norm by a random scalar in [0, 1);
        5. build the new velocity using cos(angle), sin(angle);
        6. reshape/transposed exactly as in the original code;
        7. append zero velocity for non-moving nodes.

        Parameters
        ----------
        velocity : np.ndarray
            Input velocity array.

        Returns
        -------
        np.ndarray
            Full velocity array of shape (sensor_nodes, 2).
        """

        velocity = self._truncate_or_pad_velocity(velocity)

        module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))
        angle = np.angle(velocity[:, 0] + velocity[:, 1] * 1j)

        velocity = module * np.random.random((self.n_moves,)) * np.array(
            [[np.cos(angle), np.sin(angle)]]
        )

        velocity = velocity.reshape((2, self.n_moves)).transpose()
        velocity = self._append_static_nodes(velocity)

        return velocity

    def _not_random_movement_velocity(self, velocity):
        """
        Build the velocity used in the 'not random' movement mode.

        This preserves the exact behavior:
        1. truncate/pad to self.n_moves;
        2. append zeros for the non-moving nodes.

        Parameters
        ----------
        velocity : np.ndarray
            Input velocity array.

        Returns
        -------
        np.ndarray
            Full velocity array of shape (sensor_nodes, 2).
        """

        velocity = self._truncate_or_pad_velocity(velocity)
        velocity = self._append_static_nodes(velocity)

        return velocity

    def _module_movement_velocity(self, velocity):
        """
        Build the velocity used in the 'module' movement mode.

        Original behavior preserved:
        1. draw a random angle uniformly in [0, 2*pi);
        2. compute the original module/norm;
        3. rebuild the velocity with same module and random direction;
        4. reshape/transposed exactly as before;
        5. append zeros for static nodes.

        Parameters
        ----------
        velocity : np.ndarray
            Input velocity array.

        Returns
        -------
        np.ndarray
            Full velocity array of shape (sensor_nodes, 2).
        """

        angle = 2 * np.pi * np.random.random((self.n_moves, 1))
        module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))

        velocity = module * np.concatenate([np.cos(angle), np.sin(angle)], axis=1)
        velocity = velocity.reshape((2, self.n_moves)).transpose()
        velocity = self._append_static_nodes(velocity)

        return velocity

    def _gaussian_movement_velocity(self, **options):
        """
        Build the velocity used in the 'gaussian' movement mode.

        Original behavior preserved:
        - generate Gaussian velocity with standard deviation:
          options.pop('sc_mov_std', 10)
        - append zeros for non-moving nodes

        Parameters
        ----------
        **options : dict
            Options dictionary that may contain 'sc_mov_std'.

        Returns
        -------
        np.ndarray
            Full velocity array of shape (sensor_nodes, 2).
        """

        velocity = np.random.normal(
            loc=0,
            scale=options.pop('sc_mov_std', 10),
            size=[self.n_moves, 2]
        )

        velocity = self._append_static_nodes(velocity)
        return velocity

    # ======================================================================
    # Public API
    # ======================================================================

    def move_nodes(self, n_steps=1, **options):
        """
        Update node positions according to the selected movement model.

        This method preserves the exact branching logic of the original code.

        Parameters
        ----------
        n_steps : int or float, optional
            Number of movement steps.

        **options : dict
            Optional runtime overrides:
            - 'movement_type'
            - 'velocity'
            - 'sc_mov_std'  (used only for gaussian movement)

        Returns
        -------
        None
            The original method updates self.sn_positions in place and does not
            return anything. This behavior is preserved.

        Notes
        -----
        Exact update rule preserved:
            self.sn_positions = self.sn_positions + self.energy_slot_time * n_steps * velocity
        """

        # ------------------------------------------------------------------
        # Preserve the original defaulting behavior.
        # ------------------------------------------------------------------
        movement_type = options.pop('movement_type', self.sn_movement_type)
        velocity = options.pop('velocity', self.sn_velocity)

        # ------------------------------------------------------------------
        # Preserve the original movement branching logic exactly.
        # ------------------------------------------------------------------
        if movement_type == 'direction':
            velocity = self._direction_movement_velocity(velocity)

        elif movement_type == 'not random':
            velocity = self._not_random_movement_velocity(velocity)

        elif movement_type == 'module':
            velocity = self._module_movement_velocity(velocity)

        elif movement_type == 'gaussian':
            velocity = self._gaussian_movement_velocity(**options)

        else:
            # --------------------------------------------------------------
            # Preserve the original fallback behavior exactly.
            # In the original code, velocity becomes the scalar 0.
            # --------------------------------------------------------------
            velocity = 0

        # ------------------------------------------------------------------
        # Preserve the exact position update rule.
        # ------------------------------------------------------------------
        self.sn_positions = self.sn_positions + self.energy_slot_time * n_steps * velocity
