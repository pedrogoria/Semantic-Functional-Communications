"""
sfc/core/mac/tdma.py

Ideal / budget-aware TDMA MAC core.

Purpose
-------
This module implements a first TDMA MAC layer consistent with the new
architecture:

    sfc/core/mac/

and with future fair-comparison stacks such as:

    - Benchmark + TDMA
    - PPM + TDMA

Design philosophy
-----------------
This first TDMA implementation is intentionally ideal at the MAC/resource level:

1. It explicitly splits the total frame duration tau into per-sensor time slices
   tau_s according to a time-allocation vector.

2. It optionally rescales each sensor waveform so that its average power equals
   the configured budget P_per_sensor.

3. It does NOT yet enforce a fully explicit burst-packing waveform synthesis in
   a shared single-channel timeline.

Instead, this class currently acts as:
- a fairness-enforcing time allocator,
- a clean integration point for future Benchmark + TDMA / PPM + TDMA stacks,
- a provider of time-slot metadata for future explicit time-multiplexed PHY logic.

Why this is still useful
------------------------
At the current comparison-design stage, the most important requirement is to
make explicit that:

- the total time budget is tau,
- each sensor gets a time slice tau_s,
- each sensor uses the same total communication band B_total,
- each sensor keeps the same average power budget P_per_sensor,

while leaving the comparison stacks clean and consistent.

Future evolution
----------------
A later version may extend this class to:
- explicitly gate each sensor waveform into its TDMA slot,
- serialize all sensor waveforms into a shared composite waveform,
- recover waveform bursts by explicit temporal demultiplexing.

Current tensor convention
-------------------------
Sensor-wise waveforms are expected to use:

    (time, periods, sensors)

Main capabilities
-----------------
- resolve TDMA time slices
- describe slot start/stop times
- optionally normalize waveform power per sensor
- return per-sensor allocated waveforms
- provide a reversible identity-style demultiplexing contract for ideal TDMA

IMPORTANT
---------
This class is modulation-agnostic:
- it can receive waveforms produced by Benchmark-like sample communication
- it can receive waveforms produced by PPM
- it does not inspect the internal modulation structure
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sfc.core.mac.base import (
    MACCoreBase,
    MACInput,
    MACMultiplexResult,
    MACDemultiplexResult,
    ensure_2d_aggregated_waveform,
    ensure_3d_waveform_tensor,
    validate_sensor_dimension,
    validate_time_dimension,
)


class TDMACore(MACCoreBase):
    """
    Ideal / budget-aware TDMA MAC core.

    Parameters
    ----------
    S : int
        Number of sensors.

    B_total : float
        Total communication bandwidth available in the shared system.

    P_per_sensor : float
        Average available power per sensor.

    tau : float
        Global frame / period duration.

    time_allocation : array-like or None, optional
        Allocation fractions across sensors.
        If None, equal time split is assumed.

    normalize_sensor_power : bool, optional
        If True, rescale each sensor waveform so that its average power equals
        P_per_sensor. Default is False.

    return_nonexclusive_sum_preview : bool, optional
        If True, also return a simple sum across sensors as a preview waveform.
        IMPORTANT:
        this preview is NOT a physically correct explicit TDMA serialization,
        because no hard gating into exclusive time windows is applied yet.
        It is provided only as a diagnostic convenience.
        Default is False.
    """

    def __init__(
        self,
        S: int,
        B_total: float,
        P_per_sensor: float,
        tau: float,
        time_allocation: Optional[np.ndarray] = None,
        normalize_sensor_power: bool = False,
        return_nonexclusive_sum_preview: bool = False,
        **kwargs
    ):
        super().__init__(
            S=S,
            B_total=B_total,
            P_per_sensor=P_per_sensor,
            tau=tau,
            bandwidth_allocation=None,   # TDMA does not use bandwidth fractions here
            normalize_sensor_power=normalize_sensor_power,
            return_nonexclusive_sum_preview=return_nonexclusive_sum_preview,
            **kwargs,
        )

        self.time_allocation = _resolve_time_allocation(
            S=self.S,
            time_allocation=time_allocation
        )

        self.tau_per_sensor = self.tau * self.time_allocation
        self.normalize_sensor_power = bool(normalize_sensor_power)
        self.return_nonexclusive_sum_preview = bool(return_nonexclusive_sum_preview)

    # =========================================================================
    # RESOURCE DESCRIPTION
    # =========================================================================

    def get_time_slots(self) -> np.ndarray:
        """
        Return the TDMA slot start/stop times inside one frame.

        Returns
        -------
        np.ndarray
            Shape:
                (S, 2)

            where each row is:
                [t_start, t_stop]
        """

        slots = np.zeros((self.S, 2), dtype=float)

        cursor = 0.0
        for s in range(self.S):
            t_start = cursor
            t_stop = cursor + self.tau_per_sensor[s]
            slots[s, 0] = t_start
            slots[s, 1] = t_stop
            cursor = t_stop

        return slots

    def get_time_centers(self) -> np.ndarray:
        """
        Return the center time of each TDMA slot.

        Returns
        -------
        np.ndarray
            Shape:
                (S,)
        """

        slots = self.get_time_slots()
        return np.mean(slots, axis=1)

    def describe_budget(self) -> Dict[str, Any]:
        """
        Return the TDMA resource budget summary.

        Extends the base class summary with TDMA-specific time descriptors.
        """

        summary = super().describe_budget()
        summary.update({
            "mac_type": "tdma",
            "time_allocation": np.array(self.time_allocation, dtype=float),
            "tau_per_sensor": np.array(self.tau_per_sensor, dtype=float),
            "time_slots": self.get_time_slots(),
            "time_centers": self.get_time_centers(),
            "normalize_sensor_power": self.normalize_sensor_power,
            "return_nonexclusive_sum_preview": self.return_nonexclusive_sum_preview,
            # In this TDMA interpretation, the full bandwidth is available
            # during the sensor's active slot.
            "B_per_sensor_during_active_slot": np.full(self.S, self.B_total, dtype=float),
        })
        return summary

    # =========================================================================
    # MULTIPLEX
    # =========================================================================

    def multiplex(
        self,
        mac_input: MACInput,
        **kwargs
    ) -> MACMultiplexResult:
        """
        Apply ideal/budget-aware TDMA allocation.

        Parameters
        ----------
        mac_input : MACInput
            Expected main field:
                tx_waveform_per_sensor

        Keyword arguments
        -----------------
        normalize_sensor_power : bool, optional
            Overrides the instance default for this call.

        return_nonexclusive_sum_preview : bool, optional
            Overrides the instance default for this call.

        Returns
        -------
        MACMultiplexResult
            In this ideal TDMA version:
            - per_sensor_allocated_waveform is the main output
            - multiplexed_waveform is returned only if the preview sum is enabled
        """

        if mac_input.tx_waveform_per_sensor is None:
            raise ValueError(
                "TDMACore.multiplex(...) expects mac_input.tx_waveform_per_sensor "
                "to be provided."
            )

        x = ensure_3d_waveform_tensor(mac_input.tx_waveform_per_sensor)
        validate_sensor_dimension(x, self.S)

        if mac_input.t is not None:
            validate_time_dimension(x, mac_input.t)

        normalize_sensor_power = kwargs.get(
            "normalize_sensor_power",
            self.normalize_sensor_power
        )
        return_nonexclusive_sum_preview = kwargs.get(
            "return_nonexclusive_sum_preview",
            self.return_nonexclusive_sum_preview
        )

        x_alloc = np.array(x, dtype=float, copy=True)

        if normalize_sensor_power:
            x_alloc = self._scale_waveforms_to_target_power(
                x_alloc,
                target_power=self.P_per_sensor
            )

        allocation_metadata = {
            "mac_type": "tdma",
            "S": self.S,
            "B_total": self.B_total,
            "P_per_sensor": self.P_per_sensor,
            "tau": self.tau,
            "time_allocation": np.array(self.time_allocation, dtype=float),
            "tau_per_sensor": np.array(self.tau_per_sensor, dtype=float),
            "time_slots": self.get_time_slots(),
            "time_centers": self.get_time_centers(),
            "B_per_sensor_during_active_slot": np.full(self.S, self.B_total, dtype=float),
            "normalize_sensor_power": bool(normalize_sensor_power),
            "ideal_tdma": True,
            "explicit_temporal_serialization": False,
        }

        if return_nonexclusive_sum_preview:
            # IMPORTANT:
            # This is NOT a physically correct explicit TDMA serialization.
            # It is only a convenience preview waveform for diagnostics.
            multiplexed = np.sum(x_alloc, axis=2)
        else:
            multiplexed = None

        return MACMultiplexResult(
            multiplexed_waveform=multiplexed,
            per_sensor_allocated_waveform=x_alloc,
            allocation_metadata=allocation_metadata,
            aux={
                "warning": (
                    "This TDMA implementation is ideal/budget-aware. "
                    "No explicit hard gating / burst serialization is applied yet."
                )
            }
        )

    # =========================================================================
    # DEMULTIPLEX
    # =========================================================================

    def demultiplex(
        self,
        received_signal: Any,
        multiplex_result: Optional[MACMultiplexResult] = None,
        **kwargs
    ) -> MACDemultiplexResult:
        """
        Ideal TDMA inverse allocation.

        Parameters
        ----------
        received_signal : Any
            Preferred accepted forms in this first implementation:
            1. np.ndarray with shape (time, periods, sensors)
               -> interpreted as already separated sensor-wise waveforms
            2. dict containing:
                   {"recovered_waveform_per_sensor": ...}
            3. None, together with multiplex_result containing
               per_sensor_allocated_waveform
               -> useful for fully ideal/self-contained tests

        multiplex_result : MACMultiplexResult or None
            Optional transmit-side TDMA metadata.

        Keyword arguments
        -----------------
        do_power_diagnostics : bool, optional
            If True, include recovered average power per sensor in aux.

        Returns
        -------
        MACDemultiplexResult
            Recovered waveform per sensor.
        """

        do_power_diagnostics = kwargs.get("do_power_diagnostics", True)

        x_rec = None

        if received_signal is None:
            if multiplex_result is None or multiplex_result.per_sensor_allocated_waveform is None:
                raise ValueError(
                    "TDMACore.demultiplex(...) received_signal is None and no "
                    "per_sensor_allocated_waveform is available in multiplex_result."
                )
            x_rec = ensure_3d_waveform_tensor(
                multiplex_result.per_sensor_allocated_waveform
            )

        elif isinstance(received_signal, dict):
            if "recovered_waveform_per_sensor" not in received_signal:
                raise KeyError(
                    "When received_signal is a dict, it must contain "
                    "'recovered_waveform_per_sensor'."
                )
            x_rec = ensure_3d_waveform_tensor(
                received_signal["recovered_waveform_per_sensor"]
            )

        else:
            x_rec = ensure_3d_waveform_tensor(received_signal)

        validate_sensor_dimension(x_rec, self.S)

        aux = {}
        if do_power_diagnostics:
            aux["recovered_average_power_per_sensor"] = self._average_power_per_sensor(x_rec)

        return MACDemultiplexResult(
            recovered_waveform_per_sensor=x_rec,
            recovered_symbols_per_sensor=None,
            recovered_event_matrix=None,
            aux=aux
        )

    # =========================================================================
    # POWER HELPERS
    # =========================================================================

    def _scale_waveforms_to_target_power(
        self,
        x: np.ndarray,
        target_power: float
    ) -> np.ndarray:
        """
        Rescale each sensor waveform so that its average power equals target_power.

        Parameters
        ----------
        x : np.ndarray
            Canonical waveform tensor:
                (time, periods, sensors)

        target_power : float
            Desired average power per sensor.

        Returns
        -------
        np.ndarray
            Power-normalized waveform tensor.
        """

        x = np.asarray(x, dtype=float)
        x_out = np.array(x, copy=True)

        _, _, S = x.shape

        for s in range(S):
            ps = np.mean(x[:, :, s] ** 2)

            if np.isclose(ps, 0.0):
                continue

            scale = np.sqrt(target_power / ps)
            x_out[:, :, s] *= scale

        return x_out

    def _average_power_per_sensor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute average waveform power per sensor.

        Parameters
        ----------
        x : np.ndarray
            Canonical waveform tensor:
                (time, periods, sensors)

        Returns
        -------
        np.ndarray
            Shape:
                (S,)
        """

        x = np.asarray(x, dtype=float)
        _, _, S = x.shape

        p = np.zeros(S, dtype=float)
        for s in range(S):
            p[s] = np.mean(x[:, :, s] ** 2)

        return p


# =============================================================================
# TIME ALLOCATION HELPERS
# =============================================================================

def _resolve_time_allocation(
    S: int,
    time_allocation: Optional[np.ndarray]
) -> np.ndarray:
    """
    Resolve a time-allocation vector.

    Rules
    -----
    - If time_allocation is None:
        equal split among S sensors
    - Otherwise:
        * length must be S
        * entries must be >= 0
        * sum must be 1

    Parameters
    ----------
    S : int
        Number of sensors.

    time_allocation : array-like or None

    Returns
    -------
    np.ndarray
        Shape:
            (S,)
    """

    if time_allocation is None:
        return np.ones(S, dtype=float) / S

    alloc = np.asarray(time_allocation, dtype=float).reshape(-1)

    if len(alloc) != S:
        raise ValueError(
            f"time_allocation length mismatch: len={len(alloc)} but S={S}"
        )

    if np.any(alloc < 0):
        raise ValueError("time_allocation must be nonnegative")

    total = np.sum(alloc)
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"time_allocation must sum to 1. Current sum={total}"
        )

    return alloc


__all__ = [
    "TDMACore",
]
