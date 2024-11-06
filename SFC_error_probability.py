# import numpy as np
import numpy as np

import sfc_channel as sfc_c
from sampling_methods import *

# import math

# simulation parameters
# bw_channel = 24000  # total bandwidth for communicate
bw_channel_lim = (1000, 3000, 5000, 7000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000)
n_periods = int(1)  # number of signal periods considered in the simulation
n_sensors = 8  # number of signals, we assume that each sensor only monitors a single signal.

# semantic-functional channel parameters
n_resource = 12  # sub-band number assigned to the SF Channel.
n_sub_symbol = 4  # length (number of rows) of the SF matrix for the SF channel
detect_errors = True

# nyquist sampling and traditional communication
snr_dB = -16
# snr_dB_lim = (-16)
average_power = 1  # average power transmission per sensor in watts

# other parameters
bw_signal = 5.3
bw_signal_lim = (5.2, 5.4)
W = 2 * bw_signal
sampling_rate = W
T = 1  # window for sampling. It is the period assigned to signals
NN = 5  # int(bw_signal * T)  # number of harmonics, i.e., number of DFT terms
Tt = 0.01  # time increment
w0 = 2 * np.pi / T
xi = np.pi * W / w0 - np.floor(np.pi * W / w0)
p2p = 4

# N0 = average_power / ((10 ** (-12 / 10)) * 10000 / n_sensors)

file_name = 'sfc_error_detect_'
cont_file = 0
# snr_dB_lim = (-16, -14)
interactions = int(500)

for bw_signal in bw_signal_lim:
    results = np.zeros((len(bw_channel_lim), 21))

    # for taking previous results
    # prv_results = np.loadtxt(file_name + str(cont_file) + '.dat', delimiter='\t')
    # results = np.concatenate((results, prv_results), axis=0)

    indx_res = 0
    for bw_channel in bw_channel_lim:
        MSE_sfc = np.zeros(interactions)
        MSE_time = np.zeros(interactions)
        MSE_cbcp = np.zeros(interactions)
        MSE = np.zeros(interactions)
        event_error_rate = np.zeros(interactions)
        overlapping_tx = np.zeros(interactions)
        mean_p2p = np.zeros(interactions)
        throughput = np.zeros(interactions)

        # preliminary calculations
        W = 2 * bw_signal
        NN = int(np.floor(np.pi * W / w0))
        sampling_rate = W
        N0 = average_power / ((10 ** (snr_dB / 10)) * bw_channel / n_sensors)
        # N0c = (2 * n_sub_symbol * (np.mean(sf_tx_power) ** 2) * NN) / (bw_channel * (10 ** (snr_dB / 10)) * T)
        # snr_dB = 10 * np.log10(average_power * n_sensors / (bw_channel * N0))
        xi = np.pi * W / w0 - np.floor(np.pi * W / w0)
        sfc_tx_amplitude = np.sqrt(T * average_power * bw_channel / (4 * n_sub_symbol * n_resource * NN)) * np.ones((n_sensors, 1))
        # sfc_tx_power = 10 * np.ones((n_sensors, 1))

        # time vectors
        t = np.arange(T / Tt) * Tt - T / 2
        t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

        for interac in range(interactions):
            s = CPSample(T=T, harmonics=NN, n_sub_symbol=n_sub_symbol, resource=n_resource, sensor_nodes=n_sensors, bandwidth=bw_channel,
                         detect_errors=detect_errors)
            ch = sfc_c.SFCChannel(sensor_nodes=n_sensors, resource=n_resource, n_sub_symbol=n_sub_symbol, sensor_x_event=s.sensors_x_event,
                                  N0=N0, tx_amplitude=sfc_tx_amplitude, power_at_receiver=True, bandwidth=bw_channel)
            N_sample = Nyquist(T=T, Tt=Tt, sampling_rate=sampling_rate, sensor_nodes=n_sensors, bandwidth=bw_channel, snr_dB=snr_dB)

            # events = np.array([rnd.randint(0, 1) for _ in range(0, N * n_sensors)]).reshape((-1, n_sensors))
            # events = np.array([1 if rnd.random() < event_rate else 0 for _ in range(0, N * n_sensors * NN)]).reshape((-1, n_sensors * NN))

            Mc = max([np.floor(N_sample.bins ** ((np.pi * W) / (np.pi * W - xi * w0))), 1])
            # Mc = max([np.floor((1 + 10 ** (snr_dB / 10)) ** (T * bw_channel / (2 * NN * n_sensors))), 1])

            x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5
            # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
            for ind2 in range(x.shape[2]):
                for ind1 in range(x.shape[1]):
                    x[:, ind1, ind2] = x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])
                    x[:, ind1, ind2] = p2p * (filter_periodic(x[:, ind1, ind2], W, Tt, T)) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())

            cont_tr = 0
            while (np.max(np.imag(ta)) > 0.001 or np.max(np.imag(tb)) > 0.001) and cont_tr < 100:
                cont_tr = cont_tr + 1
                x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5
                # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
                for ind2 in range(x.shape[2]):
                    for ind1 in range(x.shape[1]):
                        x[:, ind1, ind2] = x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])
                        x[:, ind1, ind2] = p2p * (filter_periodic(x[:, ind1, ind2], W, Tt, T)) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())

                x = x * (100 - cont_tr) / 100
                ta, tb, x = s.sample(x, Tt, t=t)

            mean_p2p[interac] = p2p * (100 - cont_tr) / 100

            ta = np.real(ta)
            tb = np.real(tb)
            events = s(x, Tt, t=t)
            rx_events, rx_map, received_signal = ch(events)

            if detect_errors:
                ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
                x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
                ta_time, tb_time, error_time = s.event_to_ta_tb(events)
                x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
                ta_sfc, tb_sfc, error_sfc = s.event_to_ta_tb(rx_events)
                x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])
                indx_error = np.argwhere(error_sfc == 1)
                for i in range(indx_error.shape[0]):
                    x_sfc[:, indx_error[i, 0], indx_error[i, 1]] = x_time[:, indx_error[i, 0], indx_error[i, 1]]

                throughput[interac] = n_sensors - indx_error.shape[0]
            else:
                ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
                x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
                ta_time, tb_time = s.event_to_ta_tb(events)
                x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
                ta_sfc, tb_sfc = s.event_to_ta_tb(rx_events)
                x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])

            y = N_sample(x, t, quantize=True)
            yr = N_sample.recover_signal(y)

            event_error_rate[interac] = min([np.sum(events != rx_events), 1])
            overlapping_tx[interac] = np.sum(np.sum(events, 1) > 1)
            MSE_time[interac] = np.mean((x - x_time) ** 2)
            MSE_cbcp[interac] = np.mean((x - x_cbcp) ** 2)
            MSE[interac] = np.mean((x - yr) ** 2)
            if detect_errors:
                indx_no_error = np.argwhere(error_sfc == 0)
                MSE_sfc_aux = np.zeros(indx_no_error.shape[0])
                if indx_no_error.shape[0] == 0:
                    MSE_sfc_aux = MSE_time[interac]
                for i in range(indx_no_error.shape[0]):
                    MSE_sfc_aux[i] = np.mean((x[:, indx_no_error[i, 0], indx_no_error[i, 1]] -
                                              x_sfc[:, indx_no_error[i, 0], indx_no_error[i, 1]]) ** 2)
                MSE_sfc[interac] = np.mean(MSE_sfc_aux)
            else:
                MSE_sfc[interac] = np.mean((x - x_sfc) ** 2)

            x = p2p * np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5 * p2p
            # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
            for ind2 in range(x.shape[2]):
                for ind1 in range(x.shape[1]):
                    x[:, ind1, ind2] = 2 * (x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])) / x.max(0)[ind1, ind2]
                    x[:, ind1, ind2] = filter_periodic(x[:, ind1, ind2], W, Tt, T)
        theoretical_MSE = (p2p ** 2) / (12 * N_sample.bins)
        results[indx_res, :] = np.array([[np.mean(event_error_rate), np.mean(MSE_sfc), np.mean(MSE_time), np.mean(MSE_cbcp), np.mean(MSE),
                                          theoretical_MSE, min([100000, N_sample.bins]), min([100000, Mc]), bw_channel, n_sensors, n_resource,
                                          n_sub_symbol, snr_dB, W, sampling_rate, NN, T, xi, np.mean(mean_p2p), np.mean(overlapping_tx),
                                          np.mean(throughput)]])

        np.savetxt(file_name + str(cont_file) + '.dat', results, fmt='%.8f', delimiter='\t', header='error, MSE_sfc, MSE_time, MSE_cbcp, MSE, '
                                                                                                    'Theoretical MSE, M, Mc, bw, sensors, R, L, '
                                                                                                    'snr dB, W, sampling_rate, N, tau, xi, p2p, '
                                                                                                    'overlapping_tx, throughput')
        indx_res = indx_res + 1
        print("round: {}  file: {}".format(indx_res, cont_file))

    cont_file = cont_file + 1

# np.savetxt(file_name + str(cont_file) + '.dat', results, fmt='%.8f', delimiter='\t', header='error, MSE_sfc, MSE_scp, MSE bw, sensors, R, L, '
#                                                                                             'snr dB, bw_signal, sampling_rate, N, tau, xi, p2p')
