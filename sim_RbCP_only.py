import numpy as np

import sfc_channel as sfc_c
from sampling_methods import *

# simulation parameters
bw_channel = 15000  # total bandwidth for communicate
# bw_channel_lim = np.arange(5000, 26000, 1000)
n_periods = int(1)  # number of signal periods considered in the simulation
n_sensors = 25  # number of signals, we assume that each sensor only monitors a single signal.

# nyquist sampling and traditional communication
# snr_dB_lim = (-10, -13)
snr_dB = -10
average_power = 1  # average power transmission per sensor in watts

# other parameters
# bw_signal = 7
bw_signal_lim = (7, 7.2, 7.7)
# W = 2 * bw_signal
# sampling_rate = 1.5 * W
T = 1  # window for sampling. It is the period assigned to signals
NN = 7  # int(bw_signal * T)  # number of harmonics, i.e., number of DFT terms
Tt = 0.01  # time increment
w0 = 2 * np.pi / T
# xi = np.pi * W / w0 - np.floor(np.pi * W / w0)
p2p_lim = np.arange(0.5, 15.5, 0.5)
# p2p = 10

file_name = 'sim_RbCP_p2p'
# file_name = 'sim_RbCP'
cont_file = 0
interactions = int(500)

for bw_signal in bw_signal_lim:
    results = np.zeros((len(p2p_lim), 17))
    # results = np.zeros((len(bw_channel_lim), 17))
    indx_res = 0
    for p2p in p2p_lim:
    # for bw_channel in bw_channel_lim:
        MSE_rbcp = np.zeros(interactions)
        MSE = np.zeros(interactions)
        mean_p2p = np.zeros(interactions)

        # preliminary calculations
        W = 2 * bw_signal
        NN = int(np.floor(np.pi * W / w0))
        sampling_rate = 1.2 * W
        N_sample = Nyquist(T=T, Tt=Tt, sampling_rate=sampling_rate, sensor_nodes=n_sensors, bandwidth=bw_channel, snr_dB=snr_dB)
        # M = np.floor(Mc ** (2 * NN / T / sampling_rate))
        # Mc = np.floor((1 + (10 ** (snr_dB / 10))) ** ((T * bw_channel) / (2 * NN * n_sensors)))
        # M = np.floor((1 + (10 ** (snr_dB / 10))) ** (bw_channel / (sampling_rate * n_sensors)))
        M = N_sample.bins
        Mc = np.floor(M ** (T * sampling_rate / (2 * NN)))

        # time vectors
        t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
        t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

        for interac in range(interactions):
            s = CPSample(T=T, harmonics=NN, sensor_nodes=n_sensors, bandwidth=bw_channel)

            # uniform distribution for ta and tb
            # ta = 2 * np.pi * np.random.rand(n_periods, NN, n_sensors) - np.pi
            # tb = 2 * np.pi * np.random.rand(n_periods, NN, n_sensors) - np.pi
            # for iii in range(n_sensors):
            #     for ii in range(n_periods):
            #         ta[ii, :, iii] = ta[ii, :, iii] / (s.n * s.w0)
            #         tb[ii, :, iii] = tb[ii, :, iii] / (s.n * s.w0)

            # standard
            x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5
            # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
            for ind2 in range(x.shape[2]):
                for ind1 in range(x.shape[1]):
                    x[:, ind1, ind2] = x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])
                    x[:, ind1, ind2] = p2p * (filter_periodic(x[:, ind1, ind2], W, Tt, T)) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())
            ta, tb, x = s.sample(x, Tt, t=t)

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

            ta = np.real(ta)
            tb = np.real(tb)

            ta_q, tb_q = quantize_ta_tb(ta, tb, w0, Mc)
            xs = s.recover_signal(ta_q, tb_q, t[t_1p])
            # ta_time, tb_time = s.event_to_ta_tb(events)
            # x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
            # r_ta, r_tb = s.event_to_ta_tb(rx_events)
            # xr = s.recover_signal(r_ta, r_tb, t[t_1p])

            y = N_sample(x, t, quantize=True, peak2peak=p2p)
            yr = N_sample.recover_signal(y)

            MSE_rbcp[interac] = np.mean((x - xs) ** 2)
            MSE[interac] = np.mean((x - yr) ** 2)
            mean_p2p[interac] = p2p * (100 - cont_tr) / 100

        Q = (Mc / 2 / np.pi) * np.sin(np.pi / Mc)
        upBound_MSE = 4 * NN * (0.5 - Q) * (3 / 2 - Q)
        lowBound_MSE = 1 - 4 * (Q ** 2)
        results[indx_res, :] = np.array([[upBound_MSE, lowBound_MSE, np.mean(MSE_rbcp), np.mean(MSE),
                                          ((np.mean(mean_p2p)) ** 2) / (12 * M ** 2), Mc, N_sample.bins, M, bw_channel, n_sensors, snr_dB, W,
                                          sampling_rate, NN, T, p2p, np.mean(mean_p2p)]])
        np.savetxt(file_name + str(cont_file) + '.dat', results, fmt='%.8f', delimiter='\t', header='upBound_MSE, lowBound_MSE, MSE_rbcp, MSE, '
                                                                                                    'MSExy, Mc, Bins, M, bw, sensors, snr dB, W, '
                                                                                                    'sampling_rate, N, tau, p2p, mean_p2p')
        indx_res = indx_res + 1
        print("round: {}  file: {}".format(indx_res, cont_file))

    cont_file = cont_file + 1

# np.savetxt(file_name + str(cont_file) + '.dat', results, fmt='%.8f', delimiter='\t', header='error, MSE_sfc, MSE_scp, MSE bw, sensors, R, L, '
#                                                                                             'snr dB, bw_signal, sampling_rate, N, tau, xi, p2p')
