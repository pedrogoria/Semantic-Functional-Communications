"""
tests/test_sod_acquisition_core.py

Unit / smoke tests for:

    sfc.core.acquisition.sod.SoDAcquisitionCore

The tests cover:
1. event detection;
2. payload accounting with event-time transmission;
3. payload accounting without event-time transmission;
4. zero-order-hold reconstruction;
5. linear reconstruction;
6. budget-aware constructor using Benchmark M.

These tests are intentionally lightweight and deterministic.
They can be executed with:

    pytest tests/test_sod_acquisition_core.py

or:

    python -m unittest tests.test_sod_acquisition_core
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from sfc.core.acquisition.sod import SoDAcquisitionCore
from sfc.core.system_parameters import compute_benchmark_M_per_sensor


class TestSoDAcquisitionCore(unittest.TestCase):
    """
    Test suite for Send-on-Delta acquisition core.
    """

    def setUp(self):
        """
        Build deterministic test signals used across multiple tests.
        """
        # ---------------------------------------------------------------------
        # Dense sinusoid used for shape/reconstruction smoke tests
        # ---------------------------------------------------------------------
        self.t_dense = np.linspace(0.0, 1.0, 101, endpoint=True)
        self.x_dense_1d = np.sin(2.0 * np.pi * 2.0 * self.t_dense)
        self.x_dense = self.x_dense_1d[:, None, None]  # (time, periods, sensors)

        # ---------------------------------------------------------------------
        # Small deterministic signal with known SoD event structure
        #
        # threshold = 0.15
        #
        # t : [0, 1, 2, 3]
        # x : [0.0, 0.2, 0.2, 0.4]
        #
        # Event logic:
        #   t=0 -> initial event at value 0.0
        #   t=1 -> |0.2 - 0.0| = 0.2 >= 0.15 -> event
        #   t=2 -> |0.2 - 0.2| = 0.0         -> no event
        #   t=3 -> |0.4 - 0.2| = 0.2 >= 0.15 -> event
        #
        # Therefore expected events:
        #   times  = [0.0, 1.0, 3.0]
        #   values = [0.0, 0.2, 0.4]
        #   count  = 3
        # ---------------------------------------------------------------------
        self.t_small = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        self.x_small_1d = np.array([0.0, 0.2, 0.2, 0.4], dtype=float)
        self.x_small = self.x_small_1d[:, None, None]  # (time, periods, sensors)

    # =========================================================================
    # EVENT DETECTION
    # =========================================================================

    def test_event_detection_count_and_locations(self):
        """
        SoD should detect the expected number of events on a simple signal.
        """
        sod = SoDAcquisitionCore(
            threshold=0.15,
            transmit_event_times=True,
            quantize_amplitudes=False,
            quantize_times=False,
            reconstruction_mode="zero_order_hold",
        )

        acq = sod.acquire(self.x_small, self.t_small)

        key = "period_0_sensor_0"

        self.assertIn(key, acq.event_count)
        self.assertEqual(acq.event_count[key], 3)

        np.testing.assert_allclose(
            acq.event_times[key],
            np.array([0.0, 1.0, 3.0], dtype=float),
        )

        np.testing.assert_allclose(
            acq.event_values[key],
            np.array([0.0, 0.2, 0.4], dtype=float),
        )

    # =========================================================================
    # RECONSTRUCTION: ZERO-ORDER HOLD
    # =========================================================================

    def test_zero_order_hold_reconstruction(self):
        """
        Zero-order-hold reconstruction should reproduce the expected held values.
        """
        sod = SoDAcquisitionCore(
            threshold=0.15,
            transmit_event_times=True,
            quantize_amplitudes=False,
            quantize_times=False,
            reconstruction_mode="zero_order_hold",
        )

        acq, rec = sod.acquire_and_reconstruct(self.x_small, self.t_small)

        x_hat = rec.reconstructed_signal[:, 0, 0]

        # Events are at t=[0, 1, 3] with values [0.0, 0.2, 0.4]
        # On evaluation grid [0,1,2,3], ZOH gives:
        #   t=0 -> 0.0
        #   t=1 -> 0.2
        #   t=2 -> 0.2
        #   t=3 -> 0.4
        expected = np.array([0.0, 0.2, 0.2, 0.4], dtype=float)

        np.testing.assert_allclose(x_hat, expected, atol=1e-12)

        self.assertEqual(rec.reconstruction_mode, "zero_order_hold")
        self.assertEqual(rec.event_count["period_0_sensor_0"], 3)

    # =========================================================================
    # RECONSTRUCTION: LINEAR
    # =========================================================================

    def test_linear_reconstruction(self):
        """
        Linear reconstruction should interpolate linearly between consecutive events.
        """
        sod = SoDAcquisitionCore(
            threshold=0.15,
            transmit_event_times=True,
            quantize_amplitudes=False,
            quantize_times=False,
            reconstruction_mode="linear",
        )

        acq, rec = sod.acquire_and_reconstruct(self.x_small, self.t_small)

        x_hat = rec.reconstructed_signal[:, 0, 0]

        # Events are at t=[0, 1, 3] with values [0.0, 0.2, 0.4]
        # Linear interpolation on [0,1,2,3]:
        #   t=0 -> 0.0
        #   t=1 -> 0.2
        #   t=2 -> 0.3
        #   t=3 -> 0.4
        expected = np.array([0.0, 0.2, 0.3, 0.4], dtype=float)

        np.testing.assert_allclose(x_hat, expected, atol=1e-12)

        self.assertEqual(rec.reconstruction_mode, "linear")
        self.assertEqual(rec.event_count["period_0_sensor_0"], 3)

    # =========================================================================
    # PAYLOAD: TIME + AMPLITUDE
    # =========================================================================

    def test_payload_with_time_and_amplitude(self):
        """
        Payload bits should include both amplitude bits and time bits when
        transmit_event_times=True.
        """
        sod = SoDAcquisitionCore(
            threshold=0.15,
            transmit_event_times=True,
            quantize_amplitudes=True,
            amplitude_bins=32,   # 5 bits
            quantize_times=True,
            time_bins=128,       # 7 bits
            reconstruction_mode="zero_order_hold",
        )

        acq = sod.acquire(self.x_small, self.t_small)

        key = "period_0_sensor_0"

        # 3 events * (log2(32) + log2(128)) = 3 * (5 + 7) = 36 bits
        self.assertEqual(acq.event_count[key], 3)
        self.assertAlmostEqual(acq.payload_bits[key], 36.0, places=10)

        self.assertTrue(acq.transmit_event_times)
        self.assertIn("bits_per_event", acq.payload_metadata)
        self.assertIn(key, acq.payload_metadata["bits_per_event"])

        bits_per_event = acq.payload_metadata["bits_per_event"][key]
        self.assertAlmostEqual(bits_per_event["amplitude_bits"], 5.0, places=10)
        self.assertAlmostEqual(bits_per_event["time_bits"], 7.0, places=10)
        self.assertAlmostEqual(bits_per_event["total_bits"], 12.0, places=10)

    # =========================================================================
    # PAYLOAD: AMPLITUDE ONLY
    # =========================================================================

    def test_payload_amplitude_only(self):
        """
        Payload bits should include amplitude only when transmit_event_times=False.
        """
        sod = SoDAcquisitionCore(
            threshold=0.15,
            transmit_event_times=False,
            quantize_amplitudes=True,
            amplitude_bins=32,   # 5 bits
            quantize_times=False,
            reconstruction_mode="zero_order_hold",
        )

        acq = sod.acquire(self.x_small, self.t_small)

        key = "period_0_sensor_0"

        # 3 events * log2(32) = 3 * 5 = 15 bits
        self.assertEqual(acq.event_count[key], 3)
        self.assertAlmostEqual(acq.payload_bits[key], 15.0, places=10)

        self.assertFalse(acq.transmit_event_times)
        self.assertIn("bits_per_event", acq.payload_metadata)
        self.assertIn(key, acq.payload_metadata["bits_per_event"])

        bits_per_event = acq.payload_metadata["bits_per_event"][key]
        self.assertAlmostEqual(bits_per_event["amplitude_bits"], 5.0, places=10)
        self.assertAlmostEqual(bits_per_event["time_bits"], 0.0, places=10)
        self.assertAlmostEqual(bits_per_event["total_bits"], 5.0, places=10)

    # =========================================================================
    # SHAPE / SMOKE TEST
    # =========================================================================

    def test_dense_signal_shape_smoke(self):
        """
        A dense sinusoid should produce a reconstructed tensor with the same shape.
        """
        sod = SoDAcquisitionCore(
            threshold=0.1,
            transmit_event_times=True,
            quantize_amplitudes=True,
            amplitude_bins=32,
            quantize_times=True,
            time_bins=128,
            reconstruction_mode="zero_order_hold",
        )

        acq, rec = sod.acquire_and_reconstruct(self.x_dense, self.t_dense)

        self.assertEqual(rec.reconstructed_signal.shape, self.x_dense.shape)

        key = "period_0_sensor_0"
        self.assertIn(key, acq.event_count)
        self.assertGreater(acq.event_count[key], 0)
        self.assertIn(key, acq.payload_bits)
        self.assertGreater(acq.payload_bits[key], 0.0)

    # =========================================================================
    # BUDGET-AWARE CONSTRUCTOR
    # =========================================================================

    def test_from_benchmark_budget_uses_benchmark_M(self):
        """
        from_benchmark_budget(...) should set amplitude bins equal to the
        Benchmark Approach bins computed by compute_benchmark_M_per_sensor(...).
        """
        cfg = {
            "quantization": {
                "force_power_of_two": False,
                "rounding_mode": "floor",
            }
        }

        params = SimpleNamespace(
            S=2,
            tau=1.0,
            B=10.0,
            P=1.0,
            N0=0.1,
            bandwidth_allocation=np.array([0.5, 0.5], dtype=float),
            quantization_force_power_of_two=False,
            quantization_rounding_mode="floor",
            M_time=64,
        )

        sampling_rate = 5.0

        expected_M = compute_benchmark_M_per_sensor(
            S=params.S,
            tau=params.tau,
            B=params.B,
            P=params.P,
            N0=params.N0,
            sampling_rate=sampling_rate,
            bandwidth_allocation=params.bandwidth_allocation,
            force_power_of_two=False,
            rounding_mode="floor",
        )

        sod = SoDAcquisitionCore.from_benchmark_budget(
            cfg=cfg,
            params=params,
            threshold=0.1,
            sampling_rate=sampling_rate,
            transmit_event_times=True,
        )

        # amplitude_bins and time_bins are constructor-level config
        np.testing.assert_array_equal(
            np.asarray(sod.amplitude_bins, dtype=int),
            np.asarray(expected_M, dtype=int),
        )

        # by default, time_bins should fall back to params.M_time
        self.assertEqual(int(sod.time_bins), int(params.M_time))
        self.assertTrue(sod.transmit_event_times)
        self.assertTrue(sod.quantize_amplitudes)
        self.assertTrue(sod.quantize_times)

    # =========================================================================
    # MULTI-SENSOR PAYLOAD
    # =========================================================================

    def test_multi_sensor_bins_vector(self):
        """
        Scalar or per-sensor bins should be resolved correctly.
        """
        t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

        x_sensor0 = np.array([0.0, 0.2, 0.2, 0.4], dtype=float)
        x_sensor1 = np.array([0.0, 0.3, 0.3, 0.6], dtype=float)

        x = np.stack([x_sensor0, x_sensor1], axis=1)  # (time, sensors)
        x = x[:, None, :]  # (time, periods=1, sensors=2)

        sod = SoDAcquisitionCore(
            threshold=0.15,
            transmit_event_times=True,
            quantize_amplitudes=True,
            amplitude_bins=[32, 16],   # sensor0: 5 bits, sensor1: 4 bits
            quantize_times=True,
            time_bins=[128, 64],       # sensor0: 7 bits, sensor1: 6 bits
            reconstruction_mode="zero_order_hold",
        )

        acq = sod.acquire(x, t)

        key0 = "period_0_sensor_0"
        key1 = "period_0_sensor_1"

        # Sensor 0 -> 3 events, bits/event = 5 + 7 = 12, total = 36
        self.assertEqual(acq.event_count[key0], 3)
        self.assertAlmostEqual(acq.payload_bits[key0], 36.0, places=10)

        # Sensor 1 -> values [0.0,0.3,0.3,0.6], threshold 0.15
        # events at 0,1,3 => 3 events, bits/event = 4 + 6 = 10, total = 30
        self.assertEqual(acq.event_count[key1], 3)
        self.assertAlmostEqual(acq.payload_bits[key1], 30.0, places=10)


if __name__ == "__main__":
    unittest.main()
