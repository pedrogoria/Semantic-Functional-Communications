# sfc/pipelines/main_pipeline.py

"""
Main pipeline: runs the full SFC + CbCP + Nyquist demonstration
and saves the signal comparison results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sfc import channel as sfc_c
from sfc.sampling import CPSample, Nyquist, filter_periodic, quantize_ta_tb
from sfc.config import load_config
from sfc.params import build_main_params
from sfc.io import save_dat_with_metadata


def run(config_path: str, *, save: bool = True, plot: bool = True):
    """
    Run the main experiment pipeline.

    Parameters
    ----------
    config_path : str
        Path to YAML config.
    save : bool
        If True, saves outputs to data/results/main/.
    plot : bool
        If True, shows plots.

    Returns
    -------
    dict
        Dictionary with key results and arrays.
    """
    cfg = load_config(config_path)
    params = build_main_params(cfg)

    # -----------------------------
    # Unpack parameters
    # -----------------------------
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
    p2p = params["p2p"]

    # -----------------------------
    # Preliminary calculations
    # -----------------------------
    N0 = average_power / ((10 ** (snr_dB / 10)) * bw_channel / n_sensors)

    sfc_tx_amplitude = np.sqrt(
        T * average_power * bw_channel / (4 * n_sub_symbol * n_resource * NN)
    ) * np.ones((n_sensors, 1))

    # -----------------------------
    # Time vector
    # -----------------------------
    t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
    t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

    # -----------------------------
    # Generate signals
    # -----------------------------
    x_s1 = np.cos(2 * np.pi * 3 * t)
    x_s1 = x_s1.reshape(-1, len(t_1p[0])).T

    x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5

    for ind2 in range(x.shape[2]):
        for ind1 in range(x.shape[1]):
            x[:, ind1, ind2] -= np.mean(x[:, ind1, ind2])
            x[:, ind1, ind2] = p2p * (
                filter_periodic(x[:, ind1, ind2], W, Tt, T)
            ) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())

    x[:, :, 0] = x_s1

    # -----------------------------
    # Objects: sampler, channel, nyquist
    # -----------------------------
    s = CPSample(
        T=T,
        harmonics=NN,
        n_sub_symbol=n_sub_symbol,
        resource=n_resource,
        sensor_nodes=n_sensors,
        bandwidth=bw_channel,
        detect_errors=detect_errors,
    )

    ch = sfc_c.SFCChannel(
        sensor_nodes=n_sensors,
        resource=n_resource,
        n_sub_symbol=n_sub_symbol,
        sensor_x_event=s.sensors_x_event,
        N0=N0,
        tx_amplitude=sfc_tx_amplitude,
        power_at_receiver=True,
        bandwidth=bw_channel,
    )

    N_sample = Nyquist(
        T=T,
        Tt=Tt,
        sampling_rate=sampling_rate,
        sensor_nodes=n_sensors,
        bandwidth=bw_channel,
        snr_dB=snr_dB,
    )

    # Quantization resolution
    Mc = np.floor((1 + (10 ** (snr_dB / 10))) ** ((T * bw_channel) / (2 * NN * n_sensors)))

    # -----------------------------
    # Sampling + stability loop
    # -----------------------------
    ta, tb, x = s.sample(x, Tt, t=t[t_1p])

    cont_tr = 0
    while (np.max(np.imag(ta)) > 0.001 or np.max(np.imag(tb)) > 0.001) and cont_tr < 100:
        cont_tr += 1
        x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5

        for ind2 in range(x.shape[2]):
            for ind1 in range(x.shape[1]):
                x[:, ind1, ind2] -= np.mean(x[:, ind1, ind2])
                x[:, ind1, ind2] = p2p * (
                    filter_periodic(x[:, ind1, ind2], W, Tt, T)
                ) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())

        x *= (100 - cont_tr) / 100
        ta, tb, x = s.sample(x, Tt, t=t)

    ta = np.real(ta)
    tb = np.real(tb)

    # -----------------------------
    # Channel transmission
    # -----------------------------
    events = s(x, Tt, t=t[t_1p])
    rx_events, rx_map, received_signal = ch(events)

    # -----------------------------
    # Reconstruction
    # -----------------------------
    if detect_errors:
        ta_time, tb_time, _ = s.event_to_ta_tb(events)
        ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
        ta_sfc, tb_sfc, error_sfc = s.event_to_ta_tb(rx_events)

        x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
        x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
        x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])

        indx_error = np.argwhere(error_sfc == 1)
        for i in range(indx_error.shape[0]):
            x_sfc[:, indx_error[i, 0], indx_error[i, 1]] = x_time[:, indx_error[i, 0], indx_error[i, 1]]
    else:
        ta_time, tb_time = s.event_to_ta_tb(events)
        ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
        ta_sfc, tb_sfc = s.event_to_ta_tb(rx_events)

        x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
        x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
        x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])

    # Nyquist baseline
    y = N_sample(x, t[t_1p], quantize=True)
    yr = N_sample.recover_signal(y)

    # -----------------------------
    # Plot
    # -----------------------------
    signal_plt = 1

    if plot:
        fig0, ax0 = plt.subplots()
        ax0.set_title("SFC with cosine phase sampling")
        ax0.plot(t[t_1p], x[:, 0, signal_plt], label="Original")
        ax0.plot(t[t_1p], x_cbcp[:, 0, signal_plt], label="CbCP")
        ax0.legend(loc="upper right")
        ax0.set_xlabel("time (s)")
        plt.show()

        fig1, ax1 = plt.subplots()
        ax1.set_title("Traditional communication with Nyquist sampling")
        ax1.plot(t[t_1p], x[:, 0, signal_plt], label="Original")
        ax1.plot(t[t_1p], yr[:, 0, signal_plt], label="Nyquist")
        ax1.legend(loc="upper right")
        ax1.set_xlabel("time (s)")
        plt.show()

    # -----------------------------
    # Save
    # -----------------------------
    signals_comp = np.concatenate(
        (
            t.reshape(-1, 1),
            x[:, 0, signal_plt].reshape(-1, 1),
            x_cbcp[:, 0, signal_plt].reshape(-1, 1),
            yr[:, 0, signal_plt].reshape(-1, 1),
        ),
        axis=1
    )

    if save:
        save_dat_with_metadata(
            signals_comp,
            experiment="main",
            config=cfg,
            header="time\toriginal\tCbCP\tNyquist",
            fmt="%.8f"
        )

    return {
        "t": t,
        "t_1p": t_1p,
        "x": x,
        "x_cbcp": x_cbcp,
        "yr": yr,
        "signals_comp": signals_comp,
        "config": cfg,
    }