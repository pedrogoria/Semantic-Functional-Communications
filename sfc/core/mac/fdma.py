"""
sfc/core/mac/fdma.py

Ideal / budget-aware FDMA MAC core.

Purpose
-------
This module implements a first FDMA MAC layer consistent with the new
architecture:

    sfc/core/mac/

and with the intended fair-comparison stacks:

    - Benchmark + FDMA
    - PPM + FDMA
    - SFC

Design philosophy
-----------------
This first FDMA implementation is intentionally "ideal" at the MAC/resource
level:

1. It explicitly splits the total communication bandwidth B_total into per-sensor
   slices B_s according to a bandwidth-allocation vector.

2. It optionally rescales each sensor waveform so that the average power per
   sensor matches the configured budget P_per_sensor.

3. It does NOT yet enforce a full spectral subband synthesis/demultiplexing
   chain in the waveform domain.

Instead, this class currently acts as:
- a fairness-enforcing resource allocator,
- a clean integration point for Benchmark + FDMA and PPM + FDMA,
- a carrier/subband metadata provider for future explicit spectral FDMA.

Why this is still useful
------------------------
At the current comparison-design stage, the most important requirement is to
make explicit that:

- the total bandwidth is B_total,
- each sensor gets a slice B_s,
- each sensor keeps the same average power budget P_per_sensor,

while leaving the comparison stacks clear and consistent.

Future evolution
----------------
A later version may extend this class to:
- explicitly frequency-shift each sensor waveform to its subband,
- aggregate all sensor waveforms in a shared composite waveform,
- demultiplex by ideal or practical bandpass filtering.

Current tensor convention
-------------------------
Sensor-wise waveforms are expected to use:

    (time, periods, sensors)

Main capabilities
-----------------
- resolve FDMA bandwidth slices
- describe subband edges and centers
- optionally normalize waveform power per sensor
- return per-sensor allocated waveforms
- provide a reversible identity-style demultiplexing contract for ideal FDMA

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


class FDMACore(MACCoreBase):
    """
    Ideal / budget-aware FDMA MAC core.

    Parameters
    ----------
    S : int
        Number of sensors.

    B_total : float
        Total communication bandwidth.

    P_per_sensor : float
        Average available power per sensor.

    tau : float
        Period / frame duration.

    bandwidth_allocation : array-like or None, optional
        Allocation fractions across sensors.
        If None, equal split is assumed.

    normalize_sensor_power : bool, optional
        If True, rescale each sensor waveform so that its average power equals
        P_per_sensor. Default is False.

    return_nonorthogonal_sum_preview : bool, optional
        If True, also return a simple sum across sensors as a preview waveform.
        IMPORTANT:
        this preview is NOT a physically correct explicit FDMA superposition,
        because no spectral translation is applied.
        It is provided only as a diagnostic convenience.
        Default is False.

    frequency_axis_centered_at_zero : bool, optional
        If True, subband edges are reported over a total band:
            [-B_total/2, +B_total/2]
        Otherwise:
            [0, B_total]
        Default is True.
    """

    def __init__(
        self,
        S: int,
        B_total: float,
        P_per_sensor: float,
        tau: float,
        bandwidth_allocation: Optional[np.ndarray] = None,
        normalize_sensor_power: bool = False,
        return_nonorthogonal_sum_preview: bool = False,
        frequency_axis_centered_at_zero: bool = True,
        **kwargs
    ):
        super().__init__(
            S=S,
            B_total=B_total,
            P_per_sensor=P_per_sensor,
            tau=tau,
            bandwidth_allocation=bandwidth_allocation,
            normalize_sensor_power=normalize_sensor_power,
            return_nonorthogonal_sum_preview=return_nonorthogonal_sum_preview,
            frequency_axis_centered_at_zero=frequency_axis_centered_at_zero,
            **kwargs,
        )

        self.normalize_sensor_power = bool(normalize_sensor_power)
        self.return_nonorthogonal_sum_preview = bool(return_nonorthogonal_sum_preview)
        self.frequency_axis_centered_at_zero = bool(frequency_axis_centered_at_zero)

    # =========================================================================
    # RESOURCE DESCRIPTION
    # =========================================================================

    def get_subband_edges(self) -> np.ndarray:
        """
        Return the FDMA subband edges.

        Returns
        -------
        np.ndarray
            Shape:
                (S, 2)

            where each row is:
                [f_low, f_high]
        """

        widths = self.get_bandwidth_per_sensor()

        if self.frequency_axis_centered_at_zero:
            f_start = -self.B_total / 2.0
        else:
            f_start = 0.0

        edges = np.zeros((self.S, 2), dtype=float)

        cursor = f_start
        for s in range(self.S):
            f_low = cursor
            f_high = cursor + widths[s]
            edges[s, 0] = f_low
            edges[s, 1] = f_high
            cursor = f_high

        return edges

    def get_subband_centers(self) -> np.ndarray:
        """
        Return the center frequency of each FDMA subband.

        Returns
        -------
        np.ndarray
            Shape:
                (S,)
        """

        edges = self.get_subband_edges()
        return np.mean(edges, axis=1)

    def describe_budget(self) -> Dict[str, Any]:
        """
        Return the FDMA resource budget summary.

        Extends the base class summary with FDMA-specific subband descriptors.
        """

        summary = super().describe_budget()
        summary.update({
            "subband_edges": self.get_subband_edges(),
            "subband_centers": self.get_subband_centers(),
            "normalize_sensor_power": self.normalize_sensor_power,
            "return_nonorthogonal_sum_preview": self.return_nonorthogonal_sum_preview,
            "frequency_axis_centered_at_zero": self.frequency_axis_centered_at_zero,
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
        Apply ideal/budget-aware FDMA allocation.

        Parameters
        ----------
        mac_input : MACInput
            Expected main field:
                tx_waveform_per_sensor

        Keyword arguments
        -----------------
        normalize_sensor_power : bool, optional
            Overrides the instance default for this call.

        return_nonorthogonal_sum_preview : bool, optional
            Overrides the instance default for this call.

        Returns
        -------
        MACMultiplexResult
            In this ideal FDMA version:
            - per_sensor_allocated_waveform is the main output
            - multiplexed_waveform is returned only if the preview sum is enabled
        """

        if mac_input.tx_waveform_per_sensor is None:
            raise ValueError(
                "FDMACore.multiplex(...) expects mac_input.tx_waveform_per_sensor "
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
        return_nonorthogonal_sum_preview = kwargs.get(
            "return_nonorthogonal_sum_preview",
            self.return_nonorthogonal_sum_preview
        )

        x_alloc = np.array(x, dtype=float, copy=True)

        if normalize_sensor_power:
            x_alloc = self._scale_waveforms_to_target_power(
                x_alloc,
                target_power=self.P_per_sensor
            )

        allocation_metadata = {
            "mac_type": "fdma",
            "S": self.S,
            "B_total": self.B_total,
            "P_per_sensor": self.P_per_sensor,
            "tau": self.tau,
            "bandwidth_allocation": np.array(self.bandwidth_allocation, dtype=float),
            "B_per_sensor": self.get_bandwidth_per_sensor(),
            "subband_edges": self.get_subband_edges(),
            "subband_centers": self.get_subband_centers(),
            "normalize_sensor_power": bool(normalize_sensor_power),
            "ideal_fdma": True,
            "explicit_spectral_synthesis": False,
        }

        if return_nonorthogonal_sum_preview:
            # IMPORTANT:
            # This is NOT a physically correct explicit FDMA superposition.
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
                    "This FDMA implementation is ideal/budget-aware. "
                    "No explicit spectral frequency translation is applied yet."
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
        Ideal FDMA inverse allocation.

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
            Optional transmit-side FDMA metadata.

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
                    "FDMACore.demultiplex(...) received_signal is None and no "
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
                # Leave null signals unchanged to avoid division by zero.
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


__all__ = [
    "FDMACore",
]
