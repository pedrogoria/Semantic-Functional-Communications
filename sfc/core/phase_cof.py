"""
sfc/core/phase_cof.py

Trusted core module for phase-coefficient handling in the cosine-phase sampling pipeline.

This file isolates the phase-related logic that, in the original trusted
`sfc/sampling.py`, was distributed mainly across:
- CPSample.__call__        -> ta/tb to events conversion
- CPSample.calc_ta_tb      -> Fourier coefficients to phase parameters
- CPSample.event_to_ta_tb  -> events back to phase parameters
- calc_ta_tb(...)          -> legacy/global phase conversion helper

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- encapsulation into a dedicated class
- comments and documentation
- helper method extraction for readability

Forbidden changes:
- changing formulas
- changing branching logic
- changing the handling of 9999 sentinels
- changing event indexing rules
- changing random conflict-resolution behavior
- changing output shapes

The goal here is to preserve the original numerical and logical behavior.
"""

import numpy as np
import random as rnd


class PhaseCoefficientCore:
    """
    Core class for phase-coefficient operations.

    This class preserves the phase-related behavior of the trusted `CPSample`
    implementation, while removing unrelated responsibilities such as Fourier
    calculation, signal reconstruction, and quantization.

    Parameters
    ----------
    T : float, optional
        Observation period.

    harmonics : int, optional
        Number of harmonics.

    n_sub_symbol : int, optional
        Number of rows in the SFC transmission matrix.

    resource : int, optional
        Number of resources (kept for compatibility with the original
        CPSample context, although it is not directly used by the phase
        operations implemented here).

    sensor_nodes : int, optional
        Number of sensors.

    bandwidth : float, optional
        Channel bandwidth used to define sub-symbol time.

    detect_errors : bool, optional
        Whether event decoding should also generate the signal_error output.

    periods : int or str, optional
        Number of periods. If set to 'empty', the number of periods is inferred
        from the event matrix shape during decoding.

    threshold_harmonics : float, optional
        Threshold below which Fourier coefficients are treated as zero in the
        trusted complex-logarithm phase computation.

    Notes
    -----
    This class deliberately stores several parameters that were originally
    stored inside `CPSample`, because the phase logic depends on them.
    """

    def __init__(
            self,
            T=1,
            harmonics=3,
            n_sub_symbol=6,
            resource=7,
            sensor_nodes=5,
            bandwidth=100,
            **options
    ):
        """
        Initialize the phase-coefficient core.
        """

        # ------------------------------------------------------------------
        # Preserve the same public naming defaults used in CPSample.
        # ------------------------------------------------------------------
        self.name = options.pop('name', 'CPM')

        # ------------------------------------------------------------------
        # Store fundamental configuration exactly as in the trusted class.
        # ------------------------------------------------------------------
        self.T = T
        self.harmonics = harmonics
        self.sensors = sensor_nodes
        self.n_sub_symbol = n_sub_symbol
        self.resources = resource
        self.bandwidth = bandwidth

        # ------------------------------------------------------------------
        # Preserve the trusted sub-symbol timing definition:
        #     sub_symbol_time = 1 / (bandwidth / resources)
        # ------------------------------------------------------------------
        self.sub_symbol_time = 1 / (self.bandwidth / self.resources)

        # ------------------------------------------------------------------
        # Preserve the fundamental angular frequency:
        #     w0 = 2*pi/T
        # ------------------------------------------------------------------
        self.w0 = 2 * np.pi / T

        # ------------------------------------------------------------------
        # Harmonic index vector:
        #     [1, 2, ..., harmonics]
        # ------------------------------------------------------------------
        self.n = np.arange(harmonics) + 1

        # ------------------------------------------------------------------
        # Preserve trusted error-detection and period bookkeeping settings.
        # ------------------------------------------------------------------
        self.detect_errors = options.pop('detect_errors', False)
        self.rs_per_period = np.floor(T / self.sub_symbol_time)
        self.n_periods = options.pop('periods', 'empty')

        # ------------------------------------------------------------------
        # Preserve the trusted harmonic-threshold default and meaning.
        # ------------------------------------------------------------------
        self.threshold_harmonics = options.pop('threshold_harmonics', 0.001)

        # ------------------------------------------------------------------
        # Same assertion present in the trusted CPSample constructor.
        # ------------------------------------------------------------------
        assert self.rs_per_period > 1, 'error in parameter rs_per_period '

    # ======================================================================
    # ta/tb -> events
    # ======================================================================

    def ta_tb_to_events(self, ta, tb):
        """
        Convert phase parameters ta/tb into the event matrix.

        This method is a direct structural extraction of the event-generation
        block from the trusted `CPSample.__call__` implementation.

        Parameters
        ----------
        ta : np.ndarray
            Phase parameter ta with shape (n_periods, harmonics, sensors).

        tb : np.ndarray
            Phase parameter tb with shape (n_periods, harmonics, sensors).

        Returns
        -------
        np.ndarray
            Event matrix with shape:
                (rs_per_period * n_periods, 2 * harmonics * sensors)

        Notes
        -----
        This method preserves:
        - the 9999 sentinel logic;
        - the exact ceiling-based time-bin mapping;
        - clipping of positions to [1, rs_per_period];
        - exact event indexing scheme.
        """

        events = np.zeros(
            (int(self.rs_per_period * ta.shape[0]), int(self.harmonics * 2 * self.sensors))
        )

        self.n_periods = int(events.shape[0] / self.rs_per_period)

        ta_z = np.zeros(ta.shape)
        tb_z = np.zeros(tb.shape)

        for iii in range(self.sensors):
            for ii in range(ta.shape[0]):
                ta_z[ii, :, iii] = ta[ii, :, iii] * self.n * self.w0
                tb_z[ii, :, iii] = tb[ii, :, iii] * self.n * self.w0

                for i in self.n:
                    if ta[ii, i - 1, iii] != 9999:
                        pos_a = np.ceil(
                            (self.T * ta_z[ii, i - 1, iii] / (2 * np.pi) + self.T / 2)
                            / self.sub_symbol_time
                        )

                        if pos_a < 1:
                            pos_a = 1
                        elif pos_a > self.rs_per_period:
                            pos_a = self.rs_per_period

                        events[
                            int(pos_a + self.rs_per_period * ii - 1),
                            int(i - 1 + 2 * iii * self.harmonics)
                        ] = 1

                    if tb[ii, i - 1, iii] != 9999:
                        pos_b = np.ceil(
                            (self.T * tb_z[ii, i - 1, iii] / (2 * np.pi) + self.T / 2)
                            / self.sub_symbol_time
                        )

                        if pos_b < 1:
                            pos_b = 1
                        elif pos_b > self.rs_per_period:
                            pos_b = self.rs_per_period

                        events[
                            int(pos_b + self.rs_per_period * ii - 1),
                            int(i - 1 + self.harmonics + 2 * iii * self.harmonics)
                        ] = 1

        return events

    # ======================================================================
    # Fourier coefficients -> ta/tb
    # ======================================================================

    def calc_ta_tb(self, an, bn, **options):
        """
        Compute ta and tb from Fourier coefficients an and bn.

        This method preserves exactly the trusted implementation based on the
        complex logarithm.

        Parameters
        ----------
        an : np.ndarray
            Cosine Fourier coefficients with shape:
                (n_periods, harmonics, sensors)

        bn : np.ndarray
            Sine Fourier coefficients with shape:
                (n_periods, harmonics, sensors)

        Returns
        -------
        ta : np.ndarray
            Phase parameter ta, complex array during computation but usually
            interpreted via its real part downstream.

        tb : np.ndarray
            Phase parameter tb, complex array during computation but usually
            interpreted via its real part downstream.

        Notes
        -----
        Preserved exactly:
        - shape assertions;
        - thresholding via threshold_harmonics;
        - zero-harmonic detection;
        - complex-logarithm formula;
        - division by (n * w0);
        - setting 9999 in zero-indices.
        """

        assert an.shape[1] == self.harmonics, 'error in parameter shape: an '
        assert bn.shape[1] == self.harmonics, 'error in parameter shape: an '
        assert an.shape[2] == self.sensors, 'error in parameter shape: an '
        assert bn.shape[2] == self.sensors, 'error in parameter shape: an '

        an1 = np.copy(an)
        bn1 = np.copy(bn)

        ta = 1j * np.zeros(an.shape)
        tb = 1j * np.zeros(an.shape)

        an1[np.where(np.abs(an) < self.threshold_harmonics)] = 0
        bn1[np.where(np.abs(bn) < self.threshold_harmonics)] = 0

        zero_ind = (an1 == 0) & (bn1 == 0)

        for iii in range(self.sensors):
            for i in range(an.shape[0]):
                ta[i, :, iii] = - 1j * np.log(
                    (1j * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2) - np.sign(bn[i, :, iii]) *
                     np.sqrt(0j + (4 - an[i, :, iii] ** 2 - bn[i, :, iii] ** 2) * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2)))
                    / (2 * (1j * an[i, :, iii] + bn[i, :, iii])))

                tb[i, :, iii] = - 1j * np.log((1j * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2) + np.sign(bn[i, :, iii]) *
                                               np.sqrt(0j + (4 - an[i, :, iii] ** 2 - bn[i, :, iii] ** 2)
                                                       * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2)))
                                              / (2 * (1j * an[i, :, iii] + bn[i, :, iii])))

                ta[i, :, iii] = ta[i, :, iii] / (self.n * self.w0)
                tb[i, :, iii] = tb[i, :, iii] / (self.n * self.w0)

        ta[zero_ind] = 9999
        tb[zero_ind] = 9999

        return ta, tb

    # ======================================================================
    # events -> ta/tb
    # ======================================================================

    def event_to_ta_tb(self, events, **options):
        """
        Recover ta and tb from an event matrix.

        This method is a structural extraction of the trusted
        `CPSample.event_to_ta_tb` implementation.

        Parameters
        ----------
        events : np.ndarray
            Event matrix with shape:
                (rs_per_period * n_periods, 2 * harmonics * sensors)

        Returns
        -------
        tuple
            If detect_errors is False:
                (r_ta, r_tb)

            If detect_errors is True:
                (r_ta, r_tb, signal_error)

        Notes
        -----
        This method preserves exactly:
        - period-inference logic;
        - duplicate-event correction logic;
        - random conflict resolution using python.random;
        - c_empty handling;
        - phase reconstruction formula from event positions;
        - 9999 handling for empty positions;
        - optional signal_error output.
        """

        e = events.copy()

        if self.n_periods == 'empty':
            self.n_periods = int(events.shape[0] / self.rs_per_period)
        elif self.n_periods != int(events.shape[0] / self.rs_per_period):
            print('the number of periods has been updated')
            self.n_periods = int(events.shape[0] / self.rs_per_period)

        r_ta = np.zeros((self.n_periods, self.harmonics, self.sensors))
        r_tb = np.zeros((self.n_periods, self.harmonics, self.sensors))
        signal_error = np.zeros((self.n_periods, self.sensors))

        for ind0 in range(self.n_periods):
            events = e[
                     int(ind0 * self.rs_per_period):int((ind0 + 1) * self.rs_per_period), :
                     ]

            if self.detect_errors:
                for sensor in range(self.sensors):
                    if any(np.sum(events[:, 2 * self.harmonics * sensor:2 * self.harmonics * sensor + self.harmonics], 0) > 1):
                        signal_error[ind0, sensor] = 1
                    elif any(np.sum(events[:, 2 * self.harmonics * sensor + self.harmonics:2 * self.harmonics * sensor + 2 * self.harmonics], 0) > 1):
                        signal_error[ind0, sensor] = 1
                    elif any(
                            np.sum(events[:, 2 * self.harmonics * sensor:2 * self.harmonics * sensor + self.harmonics], 0)
                            + np.sum(events[:, 2 * self.harmonics * sensor + self.harmonics:2 * self.harmonics * sensor + 2 * self.harmonics], 0)
                            == 1
                    ):
                        signal_error[ind0, sensor] = 1

            c_empty = np.where(np.sum(events, 0) == 0)

            if not all(np.sum(events, 0) == 1):
                c = np.where(np.sum(events, 0) > 1)[0]
                k = np.where(events)

                for c_ind in range(len(c)):
                    events[:, c[c_ind]] = np.zeros((events.shape[0]))
                    events[rnd.choice(k[0][k[1] == c[c_ind]]), c[c_ind]] = 1

                c = np.where(np.sum(events, 0) == 0)

                for c_ind in range(len(c)):
                    events[rnd.randrange(events.shape[0]), c[c_ind]] = 1

            k = np.where(events)
            k_t = k[0][np.argsort(k[1])]

            n_t = np.tile(self.n, (1, self.sensors * 2))
            r_ta_tb = np.pi * ((self.sub_symbol_time * (2 * k_t - 1)) / self.T - 1) / (self.w0 * n_t)
            r_ta_tb[0, c_empty] = 9999

            for ind in range(self.sensors):
                r_ta[ind0, :, ind] = r_ta_tb[
                                     :, 2 * self.harmonics * ind:2 * self.harmonics * ind + self.harmonics
                                     ]
                r_tb[ind0, :, ind] = r_ta_tb[
                                     :, 2 * self.harmonics * ind + self.harmonics:2 * self.harmonics * ind + 2 * self.harmonics
                                     ]

        if self.detect_errors:
            return r_ta, r_tb, signal_error
        else:
            return r_ta, r_tb


def calc_ta_tb(an, bn, n, w0):
    """
    Legacy global helper for ta/tb computation.

    This function preserves exactly the standalone trusted implementation from
    the original `sfc/sampling.py`.

    Parameters
    ----------
    an : np.ndarray
        Cosine Fourier coefficients.

    bn : np.ndarray
        Sine Fourier coefficients.

    n : np.ndarray or scalar
        Harmonic indices.

    w0 : float
        Fundamental angular frequency.

    Returns
    -------
    tuple
        (ta, tb)

    Notes
    -----
    Unlike the class method `PhaseCoefficientCore.calc_ta_tb`, this global
    helper uses the arctangent-based trusted formula that was also present
    in the original file and must therefore be preserved.
    """

    ta = -2 * np.arctan(
        (2 * bn - (-(an ** 2 + bn ** 2) * (an ** 2 + bn ** 2 - 4)) ** (1 / 2))
        / (an ** 2 + 2 * an + bn ** 2)
        - (4 * bn) / (an ** 2 + 2 * an + bn ** 2)
    )
    ta = ta / (n * w0)

    tb = 2 * np.arctan(
        (2 * bn - (-(an ** 2 + bn ** 2) * (an ** 2 + bn ** 2 - 4)) ** (1 / 2))
        / (an ** 2 + 2 * an + bn ** 2)
    )
    tb = tb / (n * w0)

    return ta, tb
