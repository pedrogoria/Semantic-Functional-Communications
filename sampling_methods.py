# Import libraries
import pandas as pd  # https://pandas.pydata.org/
import matplotlib.pyplot as plt  # https://matplotlib.org/
# import matplotlib.dates as mdates  # https://matplotlib.org/
import numpy as np  # http://www.numpy.org/
import glob as glob
# from datetime import date  # https://docs.python.org/3/library/datetime.html
# from collections import defaultdict
from matplotlib import style
from pandas.plotting import register_matplotlib_converters
import random as rnd
import scipy.fftpack
import scipy

# from scipy import signal

register_matplotlib_converters()


# Threshold function
def print_new_samples(dataset, percentage, ts):
    df = dataset

    n_lines = round(600 / ts)

    # 1. Inserting extra rows between existing values

    line_ins = n_lines  # Number of lines to insert
    res_dict = {col: [y for val in df[col] for y in [val] + [np.nan] * line_ins][:-line_ins] for col in df.columns}
    df_new = pd.DataFrame(res_dict)

    # Determining values of new rows (interpolation)

    df_new['Air temperature (degC)'] = df_new['Air temperature (degC)'].interpolate()

    mimi, mama, momo = 0, 0, 0

    dfs = df_new

    max_v1 = dfs.iloc[:].max()
    mama = dfs.iloc[:].mean()
    min_v1 = dfs.iloc[:].min()

    # Percentage UP AND DOWN THE MEAN

    mimi = mama + ((percentage / 100) * abs(max_v1 - mama))
    momo = mama - ((percentage / 100) * abs(min_v1 - mama))

    # Retrieve the indices of values within the specified interval
    indices_within_interval = dfs[(dfs['Air temperature (degC)'] >= float(momo)) & (dfs['Air temperature (degC)'] <= float(mimi))].index.tolist()

    # Create a new column and fill it with the original column
    dfs['Threshold'] = dfs['Air temperature (degC)']

    # Replace the values within the interval with 'x' in the 'New_Column'
    dfs.loc[indices_within_interval, 'Threshold'] = float(mama)

    # dfs[fault].to_csv('NEW_'+str(p)+'_d0'+str(fault)+'_te.dat',sep=" ",index=False, header=False)
    return mimi, mama, momo, dfs


# Plotting Function

def plot_threshold(Temperature_threshold, plot_v, mama, mimi, momo):
    t1 = mama
    t2 = mimi
    t3 = momo

    style.use('bmh')
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 18})
    plt.xlabel('Lappeenranta October 2022')
    plt.ylabel('Temperature signal')

    if plot_v == 1:
        plt.plot(Temperature_threshold['Air temperature (degC)'], 'k')
    else:
        plt.plot(Temperature_threshold['Threshold'], 'k')

    plt.xticks([])

    plt.axhline(t2[0], linestyle='--', color='r')
    plt.axhline(t1[0], linestyle='--', color='g')
    plt.axhline(t3[0], linestyle='--', color='r')

    plt.autoscale()
    plt.show()


def plot_delta(Temperature_threshold, plot_v=1):
    style.use('bmh')
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 18})
    plt.xlabel('Lappeenranta October 2022')
    plt.ylabel('Temperature signal')

    if plot_v == 1:
        plt.plot(Temperature_threshold['Air temperature (degC)'], 'k')
    else:
        plt.plot(Temperature_threshold['Delta'], 'k')

    plt.xticks([])

    plt.autoscale()
    plt.show()


# Delta technique
def delta(dfs, p):
    m = len(dfs)
    val_delta = np.zeros((m, 1))  # Initiate delta sample per sample, m total samples
    m_delta = 0

    dfs['Delta'] = [0] * len(dfs)  # Create delta column
    dfs.iloc[0, 2] = dfs.iloc[0, 0]

    compress = 0  # Initiate compression rate

    for d in range(0, m - 1):  # "For" all samples
        val_delta[d] = abs(dfs.iloc[d, 0] - dfs.iloc[d + 1, 0])  # Calculate all deltas sample per sample
    m_delta = max(val_delta[:])  # Calculate the biggest delta in normal operation and store

    for d in range(0, m - 1):  # "For" all samples

        if abs(dfs.iloc[d, 0] - dfs.iloc[d + 1, 0]) < ((p / 100) * m_delta[0]):  # Calculate per sample fault
            dfs.iloc[d + 1, 2] = dfs.iloc[d, 2]
            # count += 1
        else:
            dfs.iloc[d + 1, 2] = dfs.iloc[d + 1, 0]

    return dfs, m_delta, val_delta


def print_new_samples_delta(fault):
    # percentages = [5,10,20,30,40,50,60,70,80,90,95]
    # percentages = [5,10,20,40,60,80,90,95]
    percentages = [50]
    paths = glob.glob('d*_te.dat')
    dfs = {}
    columns_name = ['Variable_' + str(x) for x in range(1, 53)]
    for mov in range(0, len(paths)):
        data = np.genfromtxt(paths[mov])
        dfs[mov] = pd.DataFrame(data=data, columns=columns_name)

    for p in percentages:

        delta = np.zeros((960, 52))  # Initiate delta sample per sample, 960 total samples
        m_delta = np.zeros((1, 52))  # Initiate max delta for normal operation, 52 total variables
        compress = np.zeros((1, 52))  # Initiate compression rate

        for v in range(0, 52):  # "For" all 52 variables of normal operation
            for d in range(0, dfs[0].shape[0] - 1):  # "For" all samples

                delta[d, v] = abs(dfs[0].iloc[d, v] - dfs[0].iloc[d + 1, v])  # Calculate all deltas sample per sample
            m_delta[0, v] = max(delta[:, v])  # Calculate the biggest delta in normal operation and store

        for v in range(0, 52):  # "For" all 52 variables of each fault
            count = 0
            for d in range(0, dfs[0].shape[0] - 1):  # "For" all samples

                if abs(dfs[fault].iloc[d, v] - dfs[fault].iloc[d + 1, v]) < ((p / 100) * m_delta[0, v]):  # Calculate per sample fault
                    dfs[fault].iloc[d + 1, v] = dfs[fault].iloc[d, v]
                    count += 1
            compress[0, v] = count / 960  # Compression rate for 'v' variable

        # dfs[fault].to_csv('DEL_'+str(p)+'_d0'+str(fault)+'_te.dat',sep=" ",index=False, header=False)
    # return(compress)        #Uncomment this is you want to vizualize compression rate


# cosine phase sampling
class CPSample:
    def __init__(self, T=1, harmonics=3, n_sub_symbol=6, resource=7, sensor_nodes=5, bandwidth=100, **options):
        self.name = options.pop('name', 'CPM')
        self.T = T
        self.harmonics = harmonics
        self.sensors = sensor_nodes
        self.n_sub_symbol = n_sub_symbol
        self.resources = resource
        self.bandwidth = bandwidth
        self.sub_symbol_time = 1 / (self.bandwidth / self.resources)
        self.sensors_x_event = np.zeros((self.sensors, 2 * self.sensors * self.harmonics))
        self.w0 = 2 * np.pi / T
        self.n = np.arange(harmonics) + 1
        self.detect_errors = options.pop('detect_errors', False)
        self.rs_per_period = np.floor(T / self.sub_symbol_time)
        self.n_periods = options.pop('periods', 'empty')
        self.threshold_harmonics = options.pop('threshold_harmonics', 0.001)
        self.dft_signal_periods = int(options.pop('dft_signal_periods', 1))
        assert self.rs_per_period > 1, 'error in parameter rs_per_period '
        for i in range(self.sensors):
            self.sensors_x_event[i, (2 * i * self.harmonics): (2 * (i + 1) * self.harmonics)] = np.ones((1, 2 * self.harmonics))

    def __call__(self, x, Tt, **options):
        if options.pop('ta_tb', False):
            ta = options.pop('ta')
            tb = options.pop('tb')
        else:
            # each column of x must represent a period.
            assert len(x.shape) < 4, 'error in parameter shape: x '

            if len(x.shape) == 1:
                x = x.reshape(-1, 1)

            if len(x.shape) == 2:
                xx = np.zeros(np.append(x.shape, self.sensors))
                xx[:, :, 0] = x
                x = xx

            ta, tb, x = self.sample(x, Tt, **options)
            ta = np.real(ta)
            tb = np.real(tb)

        events = np.zeros((int(self.rs_per_period * ta.shape[0]), int(self.harmonics * 2 * self.sensors)))

        self.n_periods = int(events.shape[0] / self.rs_per_period)

        ta_z = np.zeros(ta.shape)
        tb_z = np.zeros(ta.shape)
        for iii in range(self.sensors):
            for ii in range(ta.shape[0]):
                ta_z[ii, :, iii] = ta[ii, :, iii] * self.n * self.w0
                tb_z[ii, :, iii] = tb[ii, :, iii] * self.n * self.w0

                for i in self.n:
                    if ta[ii, i - 1, iii] != 9999:
                        pos_a = np.ceil((self.T * ta_z[ii, i - 1, iii] / (2 * np.pi) + self.T / 2) / self.sub_symbol_time)
                        if pos_a < 1:
                            pos_a = 1
                        elif pos_a > self.rs_per_period:
                            pos_a = self.rs_per_period

                        events[int(pos_a + self.rs_per_period * ii - 1), int(i - 1 + 2 * iii * self.harmonics)] = 1

                    if tb[ii, i - 1, iii] != 9999:
                        pos_b = np.ceil((self.T * tb_z[ii, i - 1, iii] / (2 * np.pi) + self.T / 2) / self.sub_symbol_time)
                        if pos_b < 1:
                            pos_b = 1
                        elif pos_b > self.rs_per_period:
                            pos_b = self.rs_per_period

                        events[int(pos_b + self.rs_per_period * ii - 1), int(i - 1 + self.harmonics + 2 * iii * self.harmonics)] = 1

        return events

    def calc_ta_tb(self, an, bn, **options):
        assert an.shape[1] == self.harmonics, 'error in parameter shape: an '
        assert bn.shape[1] == self.harmonics, 'error in parameter shape: an '
        assert an.shape[2] == self.sensors, 'error in parameter shape: an '
        assert bn.shape[2] == self.sensors, 'error in parameter shape: an '
        an1 = np.copy(an)
        bn1 = np.copy(bn)
        ta = 1j * np.zeros(an.shape)
        tb = 1j * np.zeros(an.shape)
        an1[np.where(np.abs(an) < self.threshold_harmonics)] = 0
        bn1[np.where(np.abs(bn) < self.threshold_harmonics)] = 0
        zero_ind = (an1 == 0) & (bn1 == 0)
        for iii in range(self.sensors):
            for i in range(an.shape[0]):
                ta[i, :, iii] = - 1j * np.log((1j * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2) - np.sign(bn[i, :, iii]) * np.sqrt(
                    0j + (4 - an[i, :, iii] ** 2 - bn[i, :, iii] ** 2) * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2))) / (
                                                      2 * (1j * an[i, :, iii] + bn[i, :, iii])))
                tb[i, :, iii] = - 1j * np.log((1j * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2) + np.sign(bn[i, :, iii]) * np.sqrt(
                    0j + (4 - an[i, :, iii] ** 2 - bn[i, :, iii] ** 2) * (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2))) / (
                                                      2 * (1j * an[i, :, iii] + bn[i, :, iii])))

                # ta[i, :, iii] = -2 * np.arctan((2 * bn[i, :, iii] - (-(an[i, :, iii] ** 2 + bn[i, :, iii] ** 2) *
                #                                                      (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2 - 4)) ** (1 / 2)) /
                #                                (an[i, :, iii] ** 2 + 2 * an[i, :, iii] + bn[i, :, iii] ** 2) - (4 * bn[i, :, iii]) /
                #                                (an[i, :, iii] ** 2 + 2 * an[i, :, iii] + bn[i, :, iii] ** 2))
                ta[i, :, iii] = ta[i, :, iii] / (self.n * self.w0)
                #
                # tb[i, :, iii] = 2 * np.arctan((2 * bn[i, :, iii] - (-(an[i, :, iii] ** 2 + bn[i, :, iii] ** 2) *
                #                                                     (an[i, :, iii] ** 2 + bn[i, :, iii] ** 2 - 4)) ** (1 / 2)) /
                #                               (an[i, :, iii] ** 2 + 2 * an[i, :, iii] + bn[i, :, iii] ** 2))
                tb[i, :, iii] = tb[i, :, iii] / (self.n * self.w0)

        ta[zero_ind] = 9999
        tb[zero_ind] = 9999
        return ta, tb

    def calc_an_bn_dft(self, x, Tt, **opt):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(x.shape) == 2:
            xx = np.zeros(np.append(x.shape, self.sensors))
            xx[:, :, 0] = x
            x = xx

        t = opt.pop('t', np.arange(x.shape[0] * self.dft_signal_periods) * Tt)
        FX = np.zeros((x.shape[1], self.harmonics + 1, self.sensors)) * 1j
        for iii in range(self.sensors):
            for ii in range(x.shape[1]):
                for i in range(0, self.harmonics + 1):
                    FX[ii, i, iii] = Tt * np.sum(np.tile(x[:, ii, iii], self.dft_signal_periods) * np.exp(-1j * i * self.w0 * t)) / self.T
                    FX[ii, i, iii] = FX[ii, i, iii] / self.dft_signal_periods

        an = 2 * np.real(FX[:, 1:, :])
        bn = -2 * np.imag(FX[:, 1:, :])

        if opt.pop('normalize', False):
            nor_x = opt.pop('norm', 3.9)
            for iii in range(self.sensors):
                for ii in range(x.shape[1]):
                    aux = max(an[ii, :, iii] ** 2 + bn[ii, :, iii] ** 2)
                    while aux >= 4:
                        print('normalizing')
                        x[:, ii, iii] = x[:, ii, iii] * np.sqrt(nor_x / aux)
                        for i in range(0, self.harmonics + 1):
                            FX[ii, i, iii] = Tt * np.sum(np.tile(x[:, ii, iii], self.dft_signal_periods) * np.exp(-1j * i * self.w0 * t)) / self.T
                            FX[ii, i, iii] = FX[ii, i, iii] / self.dft_signal_periods
                        an[ii, :, iii] = 2 * np.real(FX[ii, 1:, iii])
                        bn[ii, :, iii] = -2 * np.imag(FX[ii, 1:, iii])
                        aux = max(an[ii, :, iii] ** 2 + bn[ii, :, iii] ** 2)

        return an, bn, x

    def sample(self, x, Tt, **options):
        an, bn, x = self.calc_an_bn_dft(x, Tt, **options)
        ta, tb = self.calc_ta_tb(an, bn, **options)
        return ta, tb, x

    def event_to_ta_tb(self, events, **options):
        e = events.copy()
        if self.n_periods == 'empty':
            self.n_periods = int(events.shape[0] / self.rs_per_period)
        elif self.n_periods != int(events.shape[0] / self.rs_per_period):
            print('the number of periods has been updated')
            self.n_periods = int(events.shape[0] / self.rs_per_period)

        r_ta = np.zeros((self.n_periods, self.harmonics, self.sensors))
        r_tb = np.zeros((self.n_periods, self.harmonics, self.sensors))
        signal_error = np.zeros((self.n_periods, self.sensors))
        for ind0 in range(self.n_periods):
            events = e[int(ind0 * self.rs_per_period):int((ind0 + 1) * self.rs_per_period), :]
            if self.detect_errors:
                for sensor in range(self.sensors):
                    if any(np.sum(events[:, 2 * self.harmonics * sensor:2 * self.harmonics * sensor + self.harmonics], 0) > 1):
                        signal_error[ind0, sensor] = 1
                    elif any(np.sum(events[:, 2 * self.harmonics * sensor + self.harmonics:2 * self.harmonics * sensor + 2 * self.harmonics], 0) > 1):
                        signal_error[ind0, sensor] = 1
                    elif any(np.sum(events[:, 2 * self.harmonics * sensor:2 * self.harmonics * sensor + self.harmonics], 0) +
                             np.sum(events[:, 2 * self.harmonics * sensor + self.harmonics:2 * self.harmonics * sensor + 2 * self.harmonics], 0)
                             == 1):
                        signal_error[ind0, sensor] = 1
            c_empty = np.where(np.sum(events, 0) == 0)
            if not all(np.sum(events, 0) == 1):
                c = np.where(np.sum(events, 0) > 1)[0]
                k = np.where(events)
                for c_ind in range(len(c)):
                    events[:, c[c_ind]] = np.zeros((events.shape[0]))
                    events[rnd.choice(k[0][k[1] == c[c_ind]]), c[c_ind]] = 1

                c = np.where(np.sum(events, 0) == 0)
                for c_ind in range(len(c)):
                    events[rnd.randrange(events.shape[0]), c[c_ind]] = 1

            k = np.where(events)
            k_t = k[0][np.argsort(k[1])]
            # r_ta_tb = np.pi * ((self.sub_symbol_time * (2 * k_t - 1)) / self.T - 1)
            n_t = np.tile(self.n, (1, self.sensors * 2))
            r_ta_tb = np.pi * ((self.sub_symbol_time * (2 * k_t - 1)) / self.T - 1) / (self.w0 * n_t)
            r_ta_tb[0, c_empty] = 9999
            for ind in range(self.sensors):
                r_ta[ind0, :, ind] = r_ta_tb[:, 2 * self.harmonics * ind:2 * self.harmonics * ind + self.harmonics]
                r_tb[ind0, :, ind] = r_ta_tb[:, 2 * self.harmonics * ind + self.harmonics:2 * self.harmonics * ind + 2 * self.harmonics]

        if self.detect_errors:
            return r_ta, r_tb, signal_error
        else:
            return r_ta, r_tb

    def recover_signal(self, ta, tb, t, **options):
        xr = np.zeros((len(t), ta.shape[0], int(self.sensors)))
        for iii in range(self.sensors):
            for ii in range(ta.shape[0]):
                x = np.zeros((self.harmonics, len(t)))
                for n in range(self.harmonics):
                    if ta[ii, n, iii] != 9999:
                        x[n, :] = np.cos((n + 1) * self.w0 * (t - ta[ii, n, iii]))
                    if tb[ii, n, iii] != 9999:
                        x[n, :] = x[n, :] + np.cos((n + 1) * self.w0 * (t - tb[ii, n, iii]))
                x = np.sum(x, 0)
                xr[:, ii, iii] = x

        return xr


def calc_ta_tb(an, bn, n, w0):
    ta = -2 * np.arctan((2 * bn - (-(an ** 2 + bn ** 2) * (an ** 2 + bn ** 2 - 4)) ** (1 / 2)) / (an ** 2 + 2 * an + bn ** 2) - (4 * bn) / (
            an ** 2 + 2 * an + bn ** 2))
    ta = ta / (n * w0)

    tb = 2 * np.arctan((2 * bn - (-(an ** 2 + bn ** 2) * (an ** 2 + bn ** 2 - 4)) ** (1 / 2)) / (an ** 2 + 2 * an + bn ** 2))
    tb = tb / (n * w0)

    return ta, tb


def calc_an_bn_dft(x, Tt, T, NN, **opt):
    t = np.arange(len(x)) * Tt
    w0 = 2 * np.pi / T
    FX = np.zeros(NN + 1) * 1j
    for i in range(0, NN + 1):
        FX[i] = Tt * np.sum(x * np.exp(-1j * i * w0 * t)) / T

    an = 2 * np.real(FX[1:])
    bn = -2 * np.imag(FX[1:])

    if opt.pop('normalize', True):
        aux = max(an ** 2 + bn ** 2)
        nor_x = opt.pop('norm', 3.9)
        while aux >= 4:
            x = x * np.sqrt(nor_x / aux)
            FX = np.zeros(NN + 1) * 1j
            for i in range(0, NN + 1):
                FX[i] = Tt * np.sum(x * np.exp(-1j * i * w0 * t)) / T
            an = 2 * np.real(FX[1:])
            bn = -2 * np.imag(FX[1:])
            aux = max(an ** 2 + bn ** 2)

    return an, bn, FX


def cpm_sample(x, Tt, T, NN):
    an, bn, f = calc_an_bn_dft(x, Tt, T, NN)
    return calc_ta_tb(an, bn, np.arange(1, NN + 1), 2 * np.pi / T)


def recover_signal(ta, tb, t, w0):
    NN = len(ta)
    x = np.zeros((NN, len(t)))
    for n in range(NN):
        x[n, :] = np.cos((n + 1) * w0 * (t - ta[n])) + np.cos((n + 1) * w0 * (t - tb[n]))

    return np.sum(x, 0)


def sinc_filter(x, BW=10, Tt=0.001, Tth=10, x_clones=20):
    Ts = 1 / (2 * BW)
    th = np.arange(-Tth, Tth, Tt)
    h = np.sinc(th / Ts)
    N = x.shape[0]
    H = int(len(h) / 2)

    xf = np.zeros(x.shape)
    if len(x.shape) == 3:
        for ind1 in range(x.shape[1]):
            for ind2 in range(x.shape[2]):
                xf[:, ind1, ind2] = np.convolve(h, np.tile(x[:, ind1, ind2], x_clones))[H + int(x_clones / 2) * N:H + int(1 + x_clones / 2) * N]
        xf = xf / np.sum(h)
    elif len(x.shape) == 2:
        for ind1 in range(x.shape[1]):
            xf[:, ind1] = np.convolve(h, np.tile(x[:, ind1], x_clones))[H + int(x_clones / 2) * N:H + int(1 + x_clones / 2) * N]
        xf = xf / np.sum(h)
    elif len(x.shape) == 1:
        xf = np.convolve(h, np.tile(x, x_clones))[H + int(x_clones / 2) * N:H + int(1 + x_clones / 2) * N]
        xf = xf / np.sum(h)
    else:
        print('x.shape is no valid')

    return xf


def plot_fft(x, Tt=0.001, p_log=True):
    N = len(x)
    yf = scipy.fftpack.fft(x)
    yf = (np.abs(yf)) / N
    xf = np.linspace(0.0, 1.0 / (2.0 * Tt), N // 2)
    fig, ax = plt.subplots()
    if p_log:
        # yf = np.log10(yf)
        plt.semilogy(xf, yf[:N // 2])
    else:
        plt.plot(xf, yf[:N // 2])
    plt.show()


def filter_periodic(x, W, Tt, tau):
    Ft = 1 / Tt
    t = np.arange(-tau / 2, tau / 2, Tt)
    f = Ft / len(t) * np.arange(-len(t) / 2, len(t) / 2, 1)
    # f_w = np.array([indx for indx in range(len(f)) if abs(f[indx]) <= W / 2])

    FX = 1j * np.zeros(int(2 * np.floor(tau * W / 2) + 1))
    w0 = 2 * np.pi / tau
    L = int((len(FX) - 1) / 2)
    for w in range(len(FX)):
        FX[w] = Tt * np.sum(x * np.exp(- 1j * w0 * (w - L) * t))

    x_f = np.zeros(len(t))
    for w in range(len(FX)):
        x_f = x_f + FX[w] * np.exp(1j * w0 * (w - L) * t)

    return np.real(x_f)


def quantize(x, x_min, x_max, bins):
    bins = np.floor(bins)
    delta_bin = (x_max - x_min) / bins

    x_s = np.ceil((x - x_min) / delta_bin)
    x_s[np.where(x_s > bins)] = bins
    x_s[np.where(x_s <= 0)] = 1
    x_s = x_s * delta_bin - delta_bin / 2 + x_min

    return x_s


def quantize_ta_tb(ta, tb, w0, bins):
    n = np.arange(ta.shape[1]) + 1
    x_min = - np.pi / (n * w0)
    x_max = np.pi / (n * w0)
    ta_q = np.zeros(ta.shape)
    tb_q = np.zeros(tb.shape)

    if len(ta.shape) == 3:
        for indx0 in range(ta.shape[0]):
            for indx2 in range(ta.shape[2]):
                ta_q[indx0, :, indx2] = quantize(ta[indx0, :, indx2], x_min, x_max, bins)
                tb_q[indx0, :, indx2] = quantize(tb[indx0, :, indx2], x_min, x_max, bins)
    elif len(ta.shape) == 2:
        for indx0 in range(ta.shape[0]):
            ta_q[indx0, :] = quantize(ta[indx0, :], x_min, x_max, bins)
            tb_q[indx0, :] = quantize(tb[indx0, :], x_min, x_max, bins)
    else:
        ta_q = quantize(ta, x_min, x_max, bins)
        tb_q = quantize(tb, x_min, x_max, bins)

    ta_q[np.where(ta == 9999)] = 9999
    tb_q[np.where(tb == 9999)] = 9999

    return ta_q, tb_q


# Sampling, Quantization and Encoding
class Nyquist:
    def __init__(self, T=1, Tt=0.001, sampling_rate=10, sensor_nodes=5, bandwidth=100, **options):
        self.name = options.pop('name', 'Nyquist')
        self.sampling_rate = sampling_rate
        self.sensor_nodes = sensor_nodes
        self.bandwidth = bandwidth
        self.T = T
        self.T_sinc = options.pop('T_sinc', 10)
        self.sensor_bw = bandwidth / sensor_nodes
        self.snr_dB = options.pop('snr_dB', 3)
        P_N0bw = (10 ** (self.snr_dB / 10))
        self.sensor_channel_capacity = self.sensor_bw * np.log2(1 + P_N0bw)
        self.bits_codeword = options.pop('bits_codeword', int(np.floor(self.sensor_channel_capacity / self.sampling_rate)))
        if self.bits_codeword > 1000:
            self.bits_codeword = 1000

        assert self.bits_codeword * self.sampling_rate <= self.sensor_channel_capacity, 'bits per message exceeds the channel capacity'

        self.header = options.pop('header', False)
        if self.header:
            self.bits_msg = int(self.bits_codeword - np.ceil(np.log2(sensor_nodes)))
        else:
            self.bits_msg = self.bits_codeword
        self.bins = max([2 ** self.bits_msg, 1])
        self.Tt = Tt
        self.x_clones = options.pop('x_clones', 20)

    def __call__(self, x, t, **options):
        quantize = options.pop('quantize', True)
        # peak2peak = options.pop('peak2peak', 'sample')

        t_s = np.floor(np.arange(self.T * self.sampling_rate) * (1 / self.Tt) / self.sampling_rate)
        assert len(x.shape) < 4, 'input shape'
        if len(x.shape) == 3:
            x_s = np.zeros((len(t_s), x.shape[1], x.shape[2]))
            for ind1 in range(x.shape[1]):
                for ind2 in range(x.shape[2]):
                    x_s[:, ind1, ind2] = x[t_s.astype(int), ind1, ind2]
        elif len(x.shape) == 2:
            x_s = np.zeros((len(t_s), x.shape[1]))
            for ind1 in range(x.shape[1]):
                x_s[:, ind1] = x[t_s.astype(int), ind1]
        else:
            x_s = x[t_s.astype(int)]

        if quantize:
            peak2peak = options.pop('peak2peak', 'sample')

            if peak2peak == 'sample':
                x_max = np.max(x, 0)
                x_min = np.min(x, 0)
                delta_bin = (x_max - x_min) / self.bins
            else:
                x_max = peak2peak / 2
                x_min = -peak2peak / 2
                delta_bin = (x_max - x_min) / self.bins

            x_s = np.ceil((x_s - x_min) / delta_bin)
            x_s[np.where(x_s > self.bins)] = self.bins
            x_s[np.where(x_s <= 0)] = 1
            x_s = x_s * delta_bin - delta_bin / 2 + x_min

        return x_s

    def quantize(self, x, **options):
        peak2peak = options.pop('peak2peak', 'sample')

        if peak2peak == 'sample':
            x_max = np.max(x, 0)
            x_min = np.min(x, 0)
            delta_bin = (x_max - x_min) / self.bins
        else:
            x_max = peak2peak / 2
            x_min = -peak2peak / 2
            delta_bin = (x_max - x_min) / self.bins

        x_s = np.ceil((x - x_min) / delta_bin)
        x_s[np.where(x_s > self.bins)] = self.bins
        x_s[np.where(x_s <= 0)] = 1
        return x_s * delta_bin - delta_bin / 2 + x_min

    def recover_signal(self, x_s):
        th = np.arange(-int(self.T_sinc), int(self.T_sinc), self.Tt)
        h = np.sinc(th * self.sampling_rate)
        H = int(len(h) / 2)
        t_s = np.floor(np.arange(self.T * self.sampling_rate) * (1 / self.Tt) / self.sampling_rate)

        N = int(self.T / self.Tt)

        assert len(x_s.shape) < 4, 'input shape'

        if len(x_s.shape) == 3:
            xr = np.zeros((N, x_s.shape[1], x_s.shape[2]))
            x_s_t = np.zeros(N)
            for ind1 in range(x_s.shape[1]):
                for ind2 in range(x_s.shape[2]):
                    x_s_t[t_s.astype(int)] = x_s[:, ind1, ind2]
                    x_r = np.convolve(h, np.tile(x_s_t, self.x_clones))
                    xr[:, ind1, ind2] = x_r[H + int(self.x_clones / 2) * N:H + int(1 + self.x_clones / 2) * N]
        elif len(x_s.shape) == 2:
            xr = np.zeros((N, x_s.shape[1]))
            x_s_t = np.zeros(N)
            for ind1 in range(x_s.shape[1]):
                x_s_t[t_s.astype(int)] = x_s[:, ind1]
                x_r = np.convolve(h, np.tile(x_s_t, self.x_clones))
                xr[:, ind1] = x_r[H + int(self.x_clones / 2) * N:H + int(1 + self.x_clones / 2) * N]
        else:
            x_s_t = np.zeros(N)
            x_s_t[t_s.astype(int)] = x_s
            x_r = np.convolve(h, np.tile(x_s_t, self.x_clones))
            xr = x_r[H + int(self.x_clones / 2) * N:H + int(1 + self.x_clones / 2) * N]

        # xr = np.zeros((int(2 * self.n_samples_sinc - 1), x_s.shape[1], x_s.shape[2]))
        # for k in range(int(-self.n_samples_sinc), int(self.n_samples_sinc)):
        #     for p in range(x_s.shape[0]):
        #         xr[k] = xr[k] + x_s[p, 0, 0] * np.sinc((k * Tt - p / self.sampling_rate) * self.sampling_rate)

        return xr
