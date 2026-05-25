"""
sfc/core/detection.py

Event detection in SFC channel.

Stateless → functions only.
"""

import numpy as np


def detect_events(Y, mapping):
    """
    Detect events via correlation (Hadamard product).

    Y : received matrix
    mapping : SFCMapping instance
    """

    detected = []

    for event_id, C in mapping.maps.items():

        overlap = np.sum(Y * C)

        if overlap == C.shape[0]:
            detected.append(event_id)

    return detected
