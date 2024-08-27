import numpy as np

import sfc_channel as sfc_c
from sampling_methods import *
import math

# simulation parameters
# bw_channel = 24000  # total bandwidth for communicate
bw_channel_lim = np.arange(10000, 110000, 10000)
n_periods = int(1)  # number of signal periods considered in the simulation
n_sensors = 8  # number of signals, we assume that each sensor only monitors a single signal.

# semantic-functional channel parameters
n_resource = 12  # sub-band number assigned to the SF Channel.
n_sub_symbol = 4  # length (number of rows) of the SF matrix for the SF channel

# nyquist sampling and traditional communication
snr_dB_lim = (-10, -20)
average_power = 1  # average power transmission per sensor in watts

# other parameters
bw_signal = 3
# bw_signal_lim = (5, 5.2, 5.4, 5.6, 5.8)
W = 2 * bw_signal
sampling_rate = 2 * W
T = 1  # window for sampling. It is the period assigned to signals
NN = 3  # int(bw_signal * T)  # number of harmonics, i.e., number of DFT terms
Tt = 0.01  # time increment
w0 = 2 * np.pi / T
xi = np.pi * W / w0 - np.floor(np.pi * W / w0)
p2p = 4

file_name = 'sfc_error_tz_uniform'
cont_file = 0
# snr_dB_lim = (-16, -14)
interactions = int(1000)

for snr_dB in snr_dB_lim:
    results = np.zeros((len(bw_channel_lim), 19))
    indx_res = 0
    for bw_channel in bw_channel_lim:
        MSE_sfc = np.zeros(interactions)
        MSE_time = np.zeros(interactions)
        MSE_scp = np.zeros(interactions)
        MSE = np.zeros(interactions)
        event_error_rate = np.zeros(interactions)
        overlapping_tx = np.zeros(interactions)
        mean_p2p = np.zeros(interactions)

        # preliminary calculations
        W = 2 * bw_signal
        NN = int(np.floor(np.pi * W / w0))
        sampling_rate = 2 * W
        N0 = average_power / ((10 ** (snr_dB / 10)) * bw_channel / n_sensors)
        # N0c = (2 * n_sub_symbol * (np.mean(sf_tx_power) ** 2) * NN) / (bw_channel * (10 ** (snr_dB / 10)) * T)
        xi = np.pi * W / w0 - np.floor(np.pi * W / w0)
        sfc_tx_amplitude = np.sqrt(T * average_power * bw_channel / (4 * n_sub_symbol * n_resource * NN)) * np.ones((n_sensors, 1))
        # sfc_tx_power = 10 * np.ones((n_sensors, 1))

        # time vectors
        t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
        t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

        # generates random signal and filters them
        # x = p2p * np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5 * p2p
        # # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
        # for ind2 in range(x.shape[2]):
        #     for ind1 in range(x.shape[1]):
        #         x[:, ind1, ind2] = 2 * (x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])) / x.max(0)[ind1, ind2]
        #         x[:, ind1, ind2] = filter_periodic(x[:, ind1, ind2], W, Tt, T)
        # x[:, :, 0] = x_s1

        # Temperature = pd.read_csv("Data/Temperature_oct_22_lapp.csv", dayfirst=True, sep=",",
        #                           header=0, decimal=b".", index_col=0,
        #                           parse_dates=[[0, 1, 2, 3]], usecols=[0, 1, 2, 3, 5])
        for interac in range(interactions):
            s = CPSample(T=T, harmonics=NN, n_sub_symbol=n_sub_symbol, resource=n_resource, sensor_nodes=n_sensors, bandwidth=bw_channel)
            ch = sfc_c.SFCChannel(sensor_nodes=n_sensors, resource=n_resource, n_sub_symbol=n_sub_symbol, sensor_x_event=s.sensors_x_event,
                                  N0=N0, tx_amplitude=sfc_tx_amplitude, power_at_receiver=True, bandwidth=bw_channel)
            # N_sample = Nyquist(T=T, Tt=Tt, sampling_rate=sampling_rate, sensor_nodes=n_sensors, bandwidth=bw_channel, snr_dB=snr_dB)

            # events = np.array([rnd.randint(0, 1) for _ in range(0, N * n_sensors)]).reshape((-1, n_sensors))
            # events = np.array([1 if rnd.random() < event_rate else 0 for _ in range(0, N * n_sensors * NN)]).reshape((-1, n_sensors * NN))

            # Mc = max([np.floor(N_sample.bins ** ((np.pi * W) / (np.pi * W - xi * w0))), 1])
            # Mc = max([np.floor((1 + 10 ** (snr_dB / 10)) ** (T * bw_channel / (2 * NN * n_sensors))), 1])

            # uniform distribution for ta and tb
            ta = 2 * np.pi * np.random.rand(n_periods, NN, n_sensors) - np.pi
            tb = 2 * np.pi * np.random.rand(n_periods, NN, n_sensors) - np.pi
            for iii in range(n_sensors):
                for ii in range(n_periods):
                    ta[ii, :, iii] = ta[ii, :, iii] / (s.n * s.w0)
                    tb[ii, :, iii] = tb[ii, :, iii] / (s.n * s.w0)
            # standard
            # ta, tb, x = s.sample(x, Tt, t=t)

            cont_tr = 0

            # while (np.max(np.imag(ta)) > 0.001 or np.max(np.imag(tb)) > 0.001) and cont_tr < 100:
            #     cont_tr = cont_tr + 1
            #     x = p2p * np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5 * p2p
            #     # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
            #     x = x * (100 - cont_tr) / 100
            #     for ind2 in range(x.shape[2]):
            #         for ind1 in range(x.shape[1]):
            #             x[:, ind1, ind2] = 2 * (x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])) / x.max(0)[ind1, ind2]
            #             x[:, ind1, ind2] = filter_periodic(x[:, ind1, ind2], W, Tt, T)
            #
            #     ta, tb, x = s.sample(x, Tt, t=t)

            mean_p2p[interac] = p2p * (100 - cont_tr) / 100

            ta = np.real(ta)
            tb = np.real(tb)

            # standard
            # events = s(x, Tt, t=t)

            # uniform ta tb
            events = s(0, Tt, t=t, ta_tb=True, ta=ta, tb=tb)

            rx_events, rx_map, received_signal = ch(events)

            # ta_q, tb_q = quantize_ta_tb(ta, tb, w0, Mc)
            # xs = s.recover_signal(ta_q, tb_q, t[t_1p])
            # ta_time, tb_time = s.event_to_ta_tb(events)
            # x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
            # r_ta, r_tb = s.event_to_ta_tb(rx_events)
            # xr = s.recover_signal(r_ta, r_tb, t[t_1p])

            # y = N_sample(x, t, quantize=True)
            # yr = N_sample.recover_signal(y)

            event_error_rate[interac] = min([np.sum(events != rx_events), 1])
            overlapping_tx[interac] = np.sum(np.sum(events, 1) > 1)
            MSE_sfc[interac] = 0
            MSE_time[interac] = 0
            MSE_scp[interac] = 0
            MSE[interac] = 0

            # x = p2p * np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5 * p2p
            # # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
            # for ind2 in range(x.shape[2]):
            #     for ind1 in range(x.shape[1]):
            #         x[:, ind1, ind2] = 2 * (x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])) / x.max(0)[ind1, ind2]
            #         x[:, ind1, ind2] = filter_periodic(x[:, ind1, ind2], W, Tt, T)

        results[indx_res, :] = np.array([[np.mean(event_error_rate), np.mean(MSE_sfc), np.mean(MSE_time), np.mean(MSE_scp), np.mean(MSE),
                                          min([100000, 0]), min([100000, 0]), bw_channel, n_sensors, n_resource, n_sub_symbol,
                                          snr_dB, W, sampling_rate, NN, T, xi, np.mean(mean_p2p), np.mean(overlapping_tx)]])
        np.savetxt(file_name + str(cont_file) + '.dat', results, fmt='%.8f', delimiter='\t', header='error, MSE_sfc, MSE_time, MSE_cbcp, MSE, M, Mc, '
                                                                                                    'bw, sensors, R, L, snr dB, W, '
                                                                                                    'sampling_rate, N, tau, xi, p2p, overlapping_tx')
        indx_res = indx_res + 1
        print("round: {}  file: {}".format(indx_res, cont_file))

    cont_file = cont_file + 1

# np.savetxt(file_name + str(cont_file) + '.dat', results, fmt='%.8f', delimiter='\t', header='error, MSE_sfc, MSE_scp, MSE bw, sensors, R, L, '
#                                                                                             'snr dB, bw_signal, sampling_rate, N, tau, xi, p2p')
