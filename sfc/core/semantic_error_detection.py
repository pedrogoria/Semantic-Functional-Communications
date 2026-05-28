"""
sfc/core/semantic_error_detection.py

Semantic-based error detection (SED) for RbCP_SFC.

Manuscript-aligned logic
-----------------------
This module implements the two semantic checks described for the SFC receiver:

1. Nonunique \hat{t}_z^(s,n):
   the receiver recovers more than one value for the same parameter
   \hat{t}_a^(s,n) or \hat{t}_b^(s,n) within the same transmission cycle.

2. Failure in the pair:
   if one of the pair (\hat{t}_a^(s,n), \hat{t}_b^(s,n)) is received,
   then the other must also be received in the same cycle.

If any semantic error is detected in a cycle (period), the whole period can be
discarded, which is the action proposed in the manuscript.

Current event-ID convention
---------------------------
This file assumes the event ordering used elsewhere in the project:

for each sensor s:
    [ta events for N harmonics][tb events for N harmonics]

Hence:
    num_event_ids = 2 * N * S

and, for sensor s and harmonic n_idx:
    ta_id = 2 * s * N + n_idx
    tb_id = 2 * s * N + N + n_idx

Input convention
----------------
events_est:
    shape = (event_slots_total, num_event_ids)

The matrix is split into periods/cycles using:
    period_slots

so that:
    num_periods = event_slots_total // period_slots

Output convention
-----------------
This module returns:
- a per-period validity mask
- per-period duplicate and pair-failure flags
- a corrected event matrix (optional), where invalid periods are discarded
  by zeroing their corresponding rows
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# DATA CONTAINER
# =============================================================================

@dataclass(frozen=True)
class SemanticDetectionResult:
    """
    Immutable container with semantic-error-detection outputs.
    """

    period_valid_mask: np.ndarray
    duplicate_detected_mask: np.ndarray
    pair_failure_detected_mask: np.ndarray
    corrected_events_est: np.ndarray
    invalid_details: List[Dict]
    num_periods: int
    period_slots: int


# =============================================================================
# CORE CLASS
# =============================================================================

class SemanticErrorDetector:
    """
    Semantic-based error detector for RbCP_SFC.

    Parameters
    ----------
    cfg : dict or None, optional
        Configuration dictionary. If provided, the detector tries to read:
        - cfg["system"]["S"]
        - cfg["signal"]["N_override"] or cfg["signal"]["W"], cfg["signal"]["tau"]
        - cfg["channel"]["sensor_x_event"]

    N : int or None, optional
        Number of harmonics.

    sensor_x_event : np.ndarray or None, optional
        Sensor-event association matrix with shape:
            (S, num_event_ids)

    discard_invalid_periods : bool, optional
        If True, invalid periods are zeroed in the output event matrix.
    """

    def __init__(
        self,
        cfg: Optional[dict] = None,
        N: Optional[int] = None,
        sensor_x_event: Optional[np.ndarray] = None,
        discard_invalid_periods: bool = True,
    ):
        self.cfg = cfg
        self.discard_invalid_periods = discard_invalid_periods

        self.N = N
        self.sensor_x_event = sensor_x_event
        self.S = None

        if cfg is not None:
            if self.sensor_x_event is None:
                self.sensor_x_event = cfg.get("channel", {}).get("sensor_x_event", None)

            if self.sensor_x_event is not None:
                self.S = self.sensor_x_event.shape[0]
            else:
                self.S = cfg.get("system", {}).get("S", None)

            if self.N is None:
                # Prefer explicit N_override if present
                self.N = cfg.get("signal", {}).get("N_override", None)

        if self.sensor_x_event is not None and self.S is None:
            self.S = self.sensor_x_event.shape[0]

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def detect(
        self,
        events_est: np.ndarray,
        period_slots: int,
        N: Optional[int] = None,
        sensor_x_event: Optional[np.ndarray] = None,
        discard_invalid_periods: Optional[bool] = None,
    ) -> SemanticDetectionResult:
        """
        Detect semantic errors period by period.

        Parameters
        ----------
        events_est : np.ndarray
            Estimated event matrix with shape:
                (event_slots_total, num_event_ids)

        period_slots : int
            Number of possible event-start slots in one transmission cycle tau.

        N : int or None, optional
            Number of harmonics. Overrides the value provided at construction.

        sensor_x_event : np.ndarray or None, optional
            Sensor-event association matrix. Overrides the value provided at construction.

        discard_invalid_periods : bool or None, optional
            Override for the class-level discard_invalid_periods flag.

        Returns
        -------
        SemanticDetectionResult
            Result object containing masks and corrected events.
        """

        assert len(events_est.shape) == 2, \
            "events_est must have shape (event_slots_total, num_event_ids)"

        if discard_invalid_periods is None:
            discard_invalid_periods = self.discard_invalid_periods

        N_eff = self.N if N is None else N
        sensor_x_event_eff = self.sensor_x_event if sensor_x_event is None else sensor_x_event

        if N_eff is None:
            raise ValueError("N must be provided either at construction or in detect().")

        if sensor_x_event_eff is None:
            raise ValueError(
                "sensor_x_event must be provided either at construction or in detect()."
            )

        S_eff = sensor_x_event_eff.shape[0]
        num_event_ids = events_est.shape[1]

        expected_num_event_ids = 2 * N_eff * S_eff
        if num_event_ids != expected_num_event_ids:
            raise ValueError(
                f"Expected num_event_ids = 2 * N * S = {expected_num_event_ids}, "
                f"but got {num_event_ids}."
            )

        event_slots_total = events_est.shape[0]
        if event_slots_total % period_slots != 0:
            raise ValueError(
                f"event_slots_total={event_slots_total} is not divisible by "
                f"period_slots={period_slots}. Period segmentation is ambiguous."
            )

        num_periods = event_slots_total // period_slots

        period_valid_mask = np.ones(num_periods, dtype=bool)
        duplicate_detected_mask = np.zeros(num_periods, dtype=bool)
        pair_failure_detected_mask = np.zeros(num_periods, dtype=bool)

        corrected_events_est = np.copy(events_est)
        invalid_details: List[Dict] = []

        for p in range(num_periods):
            row_start = p * period_slots
            row_stop = (p + 1) * period_slots

            events_period = events_est[row_start:row_stop, :]

            duplicate_detected, pair_failure_detected, details = self._analyze_one_period(
                events_period=events_period,
                N=N_eff,
                S=S_eff
            )

            duplicate_detected_mask[p] = duplicate_detected
            pair_failure_detected_mask[p] = pair_failure_detected

            is_valid = not (duplicate_detected or pair_failure_detected)
            period_valid_mask[p] = is_valid

            if not is_valid:
                invalid_details.append({
                    "period_index": p,
                    "duplicate_detected": duplicate_detected,
                    "pair_failure_detected": pair_failure_detected,
                    "details": details,
                })

                if discard_invalid_periods:
                    corrected_events_est[row_start:row_stop, :] = 0.0

        return SemanticDetectionResult(
            period_valid_mask=period_valid_mask,
            duplicate_detected_mask=duplicate_detected_mask,
            pair_failure_detected_mask=pair_failure_detected_mask,
            corrected_events_est=corrected_events_est,
            invalid_details=invalid_details,
            num_periods=num_periods,
            period_slots=period_slots,
        )

    def apply(
        self,
        events_est: np.ndarray,
        period_slots: int,
        N: Optional[int] = None,
        sensor_x_event: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply semantic error detection and return only the corrected event matrix.

        This is a convenience wrapper around detect(...).

        Returns
        -------
        np.ndarray
            Corrected event matrix.
        """

        result = self.detect(
            events_est=events_est,
            period_slots=period_slots,
            N=N,
            sensor_x_event=sensor_x_event,
            discard_invalid_periods=True,
        )
        return result.corrected_events_est

    # ======================================================================
    # INTERNAL PERIOD ANALYSIS
    # ======================================================================

    def _analyze_one_period(self, events_period: np.ndarray, N: int, S: int):
        """
        Analyze one transmission period/cycle.

        Parameters
        ----------
        events_period : np.ndarray
            Shape:
                (period_slots, 2 * N * S)

        Returns
        -------
        tuple
            (
                duplicate_detected: bool,
                pair_failure_detected: bool,
                details: list
            )
        """

        counts = np.sum(events_period, axis=0)

        duplicate_detected = False
        pair_failure_detected = False
        details = []

        for s in range(S):
            base = 2 * s * N

            for n_idx in range(N):
                ta_id = base + n_idx
                tb_id = base + N + n_idx

                count_a = counts[ta_id]
                count_b = counts[tb_id]

                # ----------------------------------------------------------
                # Condition 1: Nonunique t_z^(s,n)
                # ----------------------------------------------------------
                has_duplicate = (count_a > 1) or (count_b > 1)

                # ----------------------------------------------------------
                # Condition 2: Failure in the pair
                #
                # If one of the pair is received, the other must also be received.
                # Both zero is acceptable (absent harmonic pair).
                # ----------------------------------------------------------
                has_pair_failure = ((count_a > 0) and (count_b == 0)) or (
                    (count_b > 0) and (count_a == 0)
                )

                if has_duplicate:
                    duplicate_detected = True

                if has_pair_failure:
                    pair_failure_detected = True

                if has_duplicate or has_pair_failure:
                    details.append({
                        "sensor": s,
                        "harmonic_index": n_idx,
                        "ta_id": ta_id,
                        "tb_id": tb_id,
                        "count_ta": int(count_a),
                        "count_tb": int(count_b),
                        "duplicate": bool(has_duplicate),
                        "pair_failure": bool(has_pair_failure),
                    })

        return duplicate_detected, pair_failure_detected, details


# =============================================================================
# FUNCTIONAL WRAPPERS
# =============================================================================

def detect_semantic_errors(
    events_est: np.ndarray,
    period_slots: int,
    N: int,
    sensor_x_event: np.ndarray,
    discard_invalid_periods: bool = True,
) -> SemanticDetectionResult:
    """
    Functional wrapper for semantic error detection.

    Parameters
    ----------
    events_est : np.ndarray
        Estimated event matrix with shape:
            (event_slots_total, num_event_ids)

    period_slots : int
        Number of event-start slots per period.

    N : int
        Number of harmonics.

    sensor_x_event : np.ndarray
        Sensor-event association matrix.

    discard_invalid_periods : bool, optional
        If True, invalid periods are zeroed in the corrected event matrix.

    Returns
    -------
    SemanticDetectionResult
        Full semantic-detection result.
    """

    detector = SemanticErrorDetector(
        cfg=None,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=discard_invalid_periods,
    )

    return detector.detect(
        events_est=events_est,
        period_slots=period_slots,
    )


def apply_semantic_error_detection(
    events_est: np.ndarray,
    period_slots: int,
    N: int,
    sensor_x_event: np.ndarray,
) -> np.ndarray:
    """
    Apply semantic error detection and return only the corrected event matrix.

    Returns
    -------
    np.ndarray
        Corrected event matrix.
    """

    result = detect_semantic_errors(
        events_est=events_est,
        period_slots=period_slots,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=True,
    )
    return result.corrected_events_est
