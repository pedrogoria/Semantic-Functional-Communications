import argparse
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# Project imports
from sfc import channel as sfc_c
from sfc.sampling import *
from sfc.config import load_config
from sfc.params import build_error_only_params
from sfc.paths import RESULTS_DIR


def parse_args():
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="experiments/configs/error_only_default.yaml",
        help="Path to configuration file"
    )
    return p.parse_args()


def save_results(results, prefix="error_only", config=None):
    """
    Save results and metadata using structured directory.
    """

    experiment_dir = RESULTS_DIR / prefix
    experiment_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_file = experiment_dir / f"{prefix}_{timestamp}.dat"
    meta_file = experiment_dir / f"{prefix}_{timestamp}_meta.txt"

    np.savetxt(data_file, results, delimiter="\t")

    if config is not None:
        with open(meta_file, "w") as f:
            f.write(str(config))

    print(f"[INFO] Results saved to: {data_file}")
    print(f"[INFO] Metadata saved to: {meta_file}")


def main(config_path=None):
    """
    Error-only simulation entry point.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file
    """

    # Default config if none provided
    if config_path is None:
        config_path = "experiments/configs/error_only_default.yaml"

    cfg = load_config(config_path)
    params = build_error_only_params(cfg)

    # --------------------------------------------------
    # Extract parameters
    # --------------------------------------------------
    bw_channel_lim = params["bw_channel_lim"]
    n_periods = params["n_periods"]
    n_sensors = params["n_sensors"]

    n_resource = params["n_resource"]
    n_sub_symbol = params["n_sub_symbol"]

    snr_dB_lim = params["snr_dB_lim"]
    average_power = params["average_power"]

    bw_signal = params["bw_signal"]
    W = params["W"]
    sampling_rate = params["sampling_rate"]
    T = params["T"]
    NN = params["NN"]
    Tt = params["Tt"]
    w0 = params["w0"]
    xi = params["xi"]
    p2p = params["p2p"]

    # --------------------------------------------------
    # Initialize results
    # --------------------------------------------------
    results = []

    # --------------------------------------------------
    # Simulation loop
    # --------------------------------------------------
    for bw_channel in bw_channel_lim:
        for snr_dB in snr_dB_lim:

            # ------------------------------------------
            # Noise power spectral density
            # ------------------------------------------
            N0 = average_power / (
                (10 ** (snr_dB / 10)) * bw_channel / n_sensors
            )

            # ------------------------------------------
            # Transmission amplitude
            # ------------------------------------------
            sfc_tx_amplitude = np.sqrt(
                T * average_power * bw_channel /
                (4 * n_sub_symbol * n_resource * NN)
            ) * np.ones((n_sensors, 1))

            # ------------------------------------------
            # Time vector
            # ------------------------------------------
            t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
            t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

            # ------------------------------------------
            # Generate random signals
            # ------------------------------------------
            x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5

            for ind2 in range(x.shape[2]):
                for ind1 in range(x.shape[1]):
                    x[:, ind1, ind2] = x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])
                    x[:, ind1, ind2] = p2p * (
                        filter_periodic(x[:, ind1, ind2], W, Tt, T)
                    ) / (
                        x[:, ind1, ind2].max() - x[:, ind1, ind2].min()
                    )

            # ------------------------------------------
            # Create system objects
            # ------------------------------------------
            s = CPSample(
                T=T,
                harmonics=NN,
                n_sub_symbol=n_sub_symbol,
                resource=n_resource,
                sensor_nodes=n_sensors,
                bandwidth=bw_channel,
                detect_errors=True
            )

            ch = sfc_c.SFCChannel(
                sensor_nodes=n_sensors,
                resource=n_resource,
                n_sub_symbol=n_sub_symbol,
                sensor_x_event=s.sensors_x_event,
                N0=N0,
                tx_amplitude=sfc_tx_amplitude,
                power_at_receiver=True,
                bandwidth=bw_channel
            )

            # ------------------------------------------
            # Sampling (semantic-functional)
            # ------------------------------------------
            ta, tb, x = s.sample(x, Tt, t=t[t_1p])

            # ------------------------------------------
            # Event generation
            # ------------------------------------------
            events = s(x, Tt, t=t[t_1p])

            # ------------------------------------------
            # Channel transmission
            # ------------------------------------------
            rx_events, rx_map, received_signal = ch(events)

            # ------------------------------------------
            # Error computation
            # ------------------------------------------
            event_error_rate = np.sum(events != rx_events) / events.size

            # Store result
            results.append([bw_channel, snr_dB, event_error_rate])

            print(f"[INFO] bw={bw_channel}, snr={snr_dB}, error={event_error_rate:.4f}")

    # Convert to numpy array
    results = np.array(results)

    # --------------------------------------------------
    # Save results + metadata
    # --------------------------------------------------
    save_results(results, prefix="error_only", config=cfg)


# Entry point
if __name__ == "__main__":
    main()