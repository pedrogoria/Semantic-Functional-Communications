import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Project imports
from sfc import channel as sfc_c
from sfc.sampling import *
from sfc.config import load_config
from sfc.params import build_main_params
from sfc.paths import RESULTS_DIR


def parse_args():
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="experiments/configs/default.yaml",
        help="Path to configuration file"
    )
    return p.parse_args()


def save_results(data, prefix="run_main", config=None):
    """
    Save signal comparison results and experiment metadata.
    """

    experiment_dir = RESULTS_DIR / prefix
    experiment_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_file = experiment_dir / f"{prefix}_{timestamp}.dat"
    meta_file = experiment_dir / f"{prefix}_{timestamp}_meta.txt"

    # Save numerical results
    np.savetxt(
        data_file,
        data,
        delimiter="\t",
        header="time\toriginal\tCbCP\tNyquist",
        comments=""
    )

    # Save config for reproducibility
    if config is not None:
        with open(meta_file, "w") as f:
            f.write(str(config))

    print(f"[INFO] Results saved to: {data_file}")
    print(f"[INFO] Metadata saved to: {meta_file}")

    return data_file


def main(config_path=None):
    """
    Main entry point for full simulation.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file
    """

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------
    if config_path is None:
        config_path = "experiments/configs/default.yaml"

    cfg = load_config(config_path)
    params = build_main_params(cfg)
    args = parse_args()

    # --------------------------------------------------
    # Extract parameters
    # --------------------------------------------------
    bw_channel = params["bw_channel"]
    n_periods = params["n_periods"]
    n_sensors = params["n_sensors"]

    n_resource = params["n_resource"]
    n_sub_symbol = params["n_sub_symbol"]
    detect_errors = params["detect_errors"]

    snr_dB = params["snr_dB"]
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
    # Preliminary calculations
    # --------------------------------------------------
    N0 = average_power / ((10 ** (snr_dB / 10)) * bw_channel / n_sensors)

    sfc_tx_amplitude = np.sqrt(
        T * average_power * bw_channel
        / (4 * n_sub_symbol * n_resource * NN)
    ) * np.ones((n_sensors, 1))

    # --------------------------------------------------
    # Time vector
    # --------------------------------------------------
    t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
    t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

    # --------------------------------------------------
    # Generate signals
    # --------------------------------------------------
    x_s1 = np.cos(2 * np.pi * 3 * t)
    x_s1 = x_s1.reshape(-1, len(t_1p[0]))
    x_s1 = np.transpose(x_s1)

    x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5

    for ind2 in range(x.shape[2]):
        for ind1 in range(x.shape[1]):
            x[:, ind1, ind2] -= np.mean(x[:, ind1, ind2])
            x[:, ind1, ind2] = p2p * (
                filter_periodic(x[:, ind1, ind2], W, Tt, T)
            ) / (
                                       x[:, ind1, ind2].max() - x[:, ind1, ind2].min()
                               )

    x[:, :, 0] = x_s1

    # --------------------------------------------------
    # Create system objects
    # --------------------------------------------------
    s = CPSample(
        T=T,
        harmonics=NN,
        n_sub_symbol=n_sub_symbol,
        resource=n_resource,
        sensor_nodes=n_sensors,
        bandwidth=bw_channel,
        detect_errors=detect_errors
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

    N_sample = Nyquist(
        T=T,
        Tt=Tt,
        sampling_rate=sampling_rate,
        sensor_nodes=n_sensors,
        bandwidth=bw_channel,
        snr_dB=snr_dB
    )

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------
    Mc = np.floor((1 + (10 ** (snr_dB / 10))) ** ((T * bw_channel) / (2 * NN * n_sensors)))

    ta, tb, x = s.sample(x, Tt, t=t[t_1p])

    # Stability loop
    cont_tr = 0
    while (np.max(np.imag(ta)) > 0.001 or np.max(np.imag(tb)) > 0.001) and cont_tr < 100:
        cont_tr += 1
        x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5

        for ind2 in range(x.shape[2]):
            for ind1 in range(x.shape[1]):
                x[:, ind1, ind2] -= np.mean(x[:, ind1, ind2])
                x[:, ind1, ind2] = p2p * (
                    filter_periodic(x[:, ind1, ind2], W, Tt, T)
                ) / (
                                           x[:, ind1, ind2].max() - x[:, ind1, ind2].min()
                                   )

        x *= (100 - cont_tr) / 100
        ta, tb, x = s.sample(x, Tt, t=t)

    ta = np.real(ta)
    tb = np.real(tb)

    # --------------------------------------------------
    # Channel transmission
    # --------------------------------------------------
    events = s(x, Tt, t=t[t_1p])
    rx_events, rx_map, received_signal = ch(events)

    # --------------------------------------------------
    # Reconstruction
    # --------------------------------------------------
    if detect_errors:
        ta_time, tb_time, _ = s.event_to_ta_tb(events)
        ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
        ta_sfc, tb_sfc, error_sfc = s.event_to_ta_tb(rx_events)

        x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
        x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
        x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])

        indx_error = np.argwhere(error_sfc == 1)
        for i in range(indx_error.shape[0]):
            x_sfc[:, indx_error[i, 0], indx_error[i, 1]] = \
                x_time[:, indx_error[i, 0], indx_error[i, 1]]
    else:
        ta_time, tb_time = s.event_to_ta_tb(events)
        ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
        ta_sfc, tb_sfc = s.event_to_ta_tb(rx_events)

        x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
        x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
        x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])

    # Nyquist reconstruction
    y = N_sample(x, t[t_1p], quantize=True)
    yr = N_sample.recover_signal(y)

    # --------------------------------------------------
    # Plot results
    # --------------------------------------------------
    signal_plt = 1

    fig0, ax0 = plt.subplots()
    ax0.set_title("SFC vs CbCP")
    ax0.plot(t[t_1p], x[:, 0, signal_plt], label="Original")
    ax0.plot(t[t_1p], x_cbcp[:, 0, signal_plt], label="CbCP")
    ax0.legend()
    ax0.set_xlabel("time (s)")
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.set_title("Nyquist sampling")
    ax1.plot(t[t_1p], x[:, 0, signal_plt], label="Original")
    ax1.plot(t[t_1p], yr[:, 0, signal_plt], label="Nyquist")
    ax1.legend()
    ax1.set_xlabel("time (s)")
    plt.show()

    # --------------------------------------------------
    # Save signals
    # --------------------------------------------------
    data = np.concatenate(
        (
            t.reshape(-1, 1),
            x[:, 0, signal_plt].reshape(-1, 1),
            x_cbcp[:, 0, signal_plt].reshape(-1, 1),
            yr[:, 0, signal_plt].reshape(-1, 1),
        ),
        axis=1
    )

    save_results(data, prefix="run_main", config=cfg)


# Entry point
if __name__ == "__main__":
    main()
