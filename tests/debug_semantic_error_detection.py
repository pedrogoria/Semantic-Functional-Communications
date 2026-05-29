"""
tests/debug_semantic_error_detection.py

Debug script for semantic-based error detection (SED).

It tests the two semantic conditions from the manuscript:
1. Nonunique t_z^(s,n)
2. Failure in the pair (t_a, t_b)
"""

import numpy as np

from sfc.core.semantic_error_detection import detect_semantic_errors


def build_sensor_x_event(S, N):
    num_event_ids = 2 * N * S
    sensor_x_event = np.zeros((S, num_event_ids))

    for s in range(S):
        start = 2 * s * N
        stop = 2 * (s + 1) * N
        sensor_x_event[s, start:stop] = 1.0

    return sensor_x_event


def print_result(name, result):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("period_valid_mask =", result.period_valid_mask)
    print("duplicate_detected_mask =", result.duplicate_detected_mask)
    print("pair_failure_detected_mask =", result.pair_failure_detected_mask)
    print("num_periods =", result.num_periods)
    print("period_slots =", result.period_slots)
    print("invalid_details =", result.invalid_details)
    print("corrected_events_est:")
    print(result.corrected_events_est)


def main():
    # -------------------------------------------------------------
    # Small controlled setup
    # -------------------------------------------------------------
    S = 2
    N = 3
    period_slots = 5
    num_event_ids = 2 * N * S

    sensor_x_event = build_sensor_x_event(S, N)

    # Event ID convention:
    # For sensor s:
    #   ta_id = 2*s*N + n_idx
    #   tb_id = 2*s*N + N + n_idx

    # Example:
    # sensor 0, harmonic 0:
    ta00 = 0
    tb00 = N

    # sensor 1, harmonic 1:
    ta11 = 2 * 1 * N + 1
    tb11 = 2 * 1 * N + N + 1

    # -------------------------------------------------------------
    # Case 1: Valid period
    # -------------------------------------------------------------
    events_valid = np.zeros((period_slots, num_event_ids))
    events_valid[1, ta00] = 1
    events_valid[2, tb00] = 1
    events_valid[3, ta11] = 1
    events_valid[4, tb11] = 1

    result_valid = detect_semantic_errors(
        events_est=events_valid,
        period_slots=period_slots,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=True
    )

    print_result("CASE 1: VALID PERIOD", result_valid)

    # -------------------------------------------------------------
    # Case 2: Duplicate event
    # Same ta00 received twice in the same period
    # -------------------------------------------------------------
    events_dup = np.zeros((period_slots, num_event_ids))
    events_dup[0, ta00] = 1
    events_dup[1, ta00] = 1   # duplicate
    events_dup[2, tb00] = 1

    result_dup = detect_semantic_errors(
        events_est=events_dup,
        period_slots=period_slots,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=True
    )

    print_result("CASE 2: DUPLICATE", result_dup)

    # -------------------------------------------------------------
    # Case 3: Pair failure
    # ta11 appears but tb11 never appears
    # -------------------------------------------------------------
    events_pair_fail = np.zeros((period_slots, num_event_ids))
    events_pair_fail[2, ta11] = 1

    result_pair_fail = detect_semantic_errors(
        events_est=events_pair_fail,
        period_slots=period_slots,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=True
    )

    print_result("CASE 3: PAIR FAILURE", result_pair_fail)

    # -------------------------------------------------------------
    # Case 3b: Crossed pair failure
    # ta0 and tb1 are received, but tb0 and ta1 are missing
    # This must be detected as pair failure for both harmonics.
    # -------------------------------------------------------------
    events_crossed = np.zeros((period_slots, num_event_ids))

    # sensor 0, harmonic 0
    ta00 = 0
    tb00 = N

    # sensor 0, harmonic 1
    ta01 = 1
    tb01 = N + 1

    events_crossed[1, ta00] = 1
    events_crossed[2, tb01] = 1

    result_crossed = detect_semantic_errors(
        events_est=events_crossed,
        period_slots=period_slots,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=True
    )

    print_result("CASE 3b: CROSSED PAIR FAILURE", result_crossed)

    # -------------------------------------------------------------
    # Case 4: Multi-period test
    # period 0 = valid
    # period 1 = duplicate
    # period 2 = pair failure
    # -------------------------------------------------------------
    events_all = np.zeros((3 * period_slots, num_event_ids))

    # period 0
    events_all[1, ta00] = 1
    events_all[2, tb00] = 1

    # period 1
    offset = period_slots
    events_all[offset + 0, ta00] = 1
    events_all[offset + 1, ta00] = 1
    events_all[offset + 2, tb00] = 1

    # period 2
    offset = 2 * period_slots
    events_all[offset + 3, ta11] = 1

    result_all = detect_semantic_errors(
        events_est=events_all,
        period_slots=period_slots,
        N=N,
        sensor_x_event=sensor_x_event,
        discard_invalid_periods=True
    )

    print_result("CASE 4: MULTI-PERIOD", result_all)


if __name__ == "__main__":
    main()
