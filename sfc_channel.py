import numpy as np
import matplotlib.pyplot as plt
from numpy.random import standard_normal
import random as rnd


# from scipy.fftpack import fft, fftfreq


class PositionSensorNodes:
    def __init__(self, sensor_nodes=64, n_moves=0, random_method='circumference', random_state=None,
                 uniform_low=-300, uniform_high=300, standard_deviation=70, energy_slot_time=500 * 10 ** -6, **ops):
        np.random.seed(random_state)
        self.energy_slot_time = energy_slot_time
        self.sensor_nodes = sensor_nodes
        self.sn_movement_type = ops.pop('sn_movement_type', 'not random')
        self.n_moves = n_moves if n_moves <= sensor_nodes else sensor_nodes
        if random_method == 'uniform':
            self.sn_positions = np.random.uniform(low=uniform_low, high=uniform_high, size=[sensor_nodes, 2])
        elif random_method == 'gaussian':
            self.sn_positions = np.random.normal(loc=0, scale=standard_deviation, size=[sensor_nodes, 2])
        else:
            r = ops.pop('radius', 50)
            ang = np.linspace(-np.pi, np.pi, sensor_nodes, endpoint=False)
            self.sn_positions = np.array([r * np.sin(ang), r * np.cos(ang)]).transpose()
        self.sn_velocity = ops.pop('sc_velocity', np.random.normal(loc=0, scale=ops.pop('sc_velocity_scale', 10), size=[self.n_moves, 2]))

    def move_nodes(self, n_steps=1, **options):
        movement_type = options.pop('movement_type', self.sn_movement_type)
        velocity = options.pop('velocity', self.sn_velocity)

        if movement_type == 'direction':
            if len(velocity) > self.n_moves:
                velocity = velocity[:self.n_moves]
            if len(velocity) < self.n_moves:
                velocity = np.concatenate((velocity, np.random.random((self.n_moves - len(velocity), 2))))
            module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))
            angle = np.angle(velocity[:, 0] + velocity[:, 1] * 1j)
            velocity = module * np.random.random((self.n_moves,)) * np.array([[np.cos(angle), np.sin(angle)]])
            velocity = velocity.reshape((2, self.n_moves)).transpose()
            velocity = np.concatenate((velocity, np.zeros((self.sensor_nodes - self.n_moves, 2))))
        elif movement_type == 'not random':
            if len(velocity) > self.n_moves:
                velocity = velocity[:self.n_moves]
            if len(velocity) < self.n_moves:
                velocity = np.concatenate((velocity, np.random.random((self.n_moves - len(velocity), 2))))
            velocity = np.concatenate((velocity, np.zeros((self.sensor_nodes - self.n_moves, 2))))
        elif movement_type == 'module':
            angle = 2 * np.pi * np.random.random((self.n_moves, 1))
            module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))
            velocity = module * np.concatenate([np.cos(angle), np.sin(angle)], axis=1)
            velocity = velocity.reshape((2, self.n_moves)).transpose()
            velocity = np.concatenate((velocity, np.zeros((self.sensor_nodes - self.n_moves, 2))))
        elif movement_type == 'gaussian':
            velocity = np.random.normal(loc=0, scale=options.pop('sc_mov_std', 10), size=[self.n_moves, 2])
            velocity = np.concatenate((velocity, np.zeros((self.sensor_nodes - self.n_moves, 2))))
        else:
            velocity = 0

        self.sn_positions = self.sn_positions + self.energy_slot_time * n_steps * velocity


def ricean_fading(K_dB=1, n=1):
    k = 10 ** (K_dB / 10)  # K factor in linear scale
    mu = np.sqrt(k / (2 * (k + 1)))  # mean
    sigma = np.sqrt(1 / (2 * (k + 1)))  # sigma
    h = (sigma * standard_normal(n) + mu) + 1j * (sigma * standard_normal(n) + mu)
    return np.abs(h)


class BBChannel(PositionSensorNodes):
    def __init__(self, base_station=np.array([[0, 0]]), sensor_nodes=64, carrier_frequency=2.4e9, energy_slot_time=500e-6,
                 n_sub_symbol=6, resource=7, **options):
        self.name = options.pop('name', 'Generic')
        self.base_station = base_station
        self.carrier_frequency = options.pop('carrier_frequency', carrier_frequency)
        self.fading = options.pop('fading', None)
        self.path_loss_exp = options.pop('path_loss_exp', 0)  # 2 for for propagation in free space
        self.energy_slot_time = energy_slot_time
        self.sn_movement_type = options.pop('sn_movement_type', 'module')
        self.sn_tx_power = options.pop('sn_tx_power', 1 * np.ones((sensor_nodes, 1)))
        self.n_sub_symbol = n_sub_symbol
        self.resource = resource
        self.map_class = 'random'
        self.maps = np.zeros((sensor_nodes, n_sub_symbol, resource))
        self.dc_threshold = options.pop('dc_threshold', 1)
        # self.tx_signal = np.zeros((sensor_nodes, n_sub_symbol, resource))

        if self.sn_movement_type == 'gaussian':
            self.sn_mov_std = options.pop('sn_mov_std', 10)

        super().__init__(sensor_nodes=sensor_nodes, energy_slot_time=self.energy_slot_time, sn_movement_type=self.sn_movement_type,
                         **options)

        if self.base_station.shape != (1, 2):
            self.base_station = np.array([[0, 0]])

        if self.sn_tx_power.shape != (sensor_nodes,):
            self.sn_tx_power = 10 * np.ones((sensor_nodes, 1))

        self.path_loss = self.get_impulse()
        self.tx_maps()

    def __call__(self, events, **options):
        assert self.sensor_nodes == events.shape[1], 'error in parameter shape: events '

        N = events.shape[0]
        received_signal = np.complex_(np.zeros((N + self.n_sub_symbol - 1, self.resource)))

        for e in range(0, N):
            inds = np.argwhere(events[e] == 1)
            for ind in inds:
                received_signal[e:e + self.n_sub_symbol, :] = received_signal[e:e + self.n_sub_symbol, :] + self.maps[ind] * self.path_loss[ind] \
                                                              * self.sn_tx_power[ind]

        received_signal = received_signal + np.random.normal(0, 0.01, size=received_signal.shape)
        received_signal = received_signal + 1j * np.random.normal(0, 0.01, size=received_signal.shape)
        received_signal = np.abs(received_signal)
        rx_map = np.zeros(received_signal.shape)
        rx_map[np.argwhere(received_signal > self.dc_threshold)[:, 0], np.argwhere(received_signal > self.dc_threshold)[:, 1]] = 1

        rx_events = np.zeros(events.shape)
        for e in range(0, N):
            for i in range(0, self.sensor_nodes):
                rx_events[e, i] = 1 if np.sum(rx_map[e:e + self.n_sub_symbol, :] * self.maps[i, :, :]) == self.n_sub_symbol else 0

        if options.pop('update', False):
            self.update(n_steps=options.pop('n_steps', 1))

        return rx_events

    def update(self, n_steps=1):
        self.move_nodes(n_steps=n_steps)
        self.path_loss = self.get_impulse()

    def get_impulse(self, c0=3e8):
        a1 = (self.sn_positions - self.base_station)
        path_length = np.reshape(np.sqrt(np.diag(a1.dot(a1.T))), (len(a1), 1))

        # path_delay = path_length / c0
        path_phase = np.mod(-path_length * self.carrier_frequency / c0, 2 * np.pi)
        amplitude = (c0 / (4 * np.pi * self.carrier_frequency * path_length)) ** self.path_loss_exp
        envelope = amplitude * np.exp(1j * path_phase)

        return envelope

    def unique_maps(self):
        x = []
        for i in range(0, self.maps.shape[0]):
            if not np.all(np.sum(self.maps[i], axis=1) == 1):
                x.append((i, i))
            else:
                for j in range(i + 1, self.maps.shape[0]):
                    if np.array(self.maps[i] == self.maps[j]).all():
                        x.append((i, j))
        return x

    def tx_maps(self, max_stop=1000):
        stop = 0
        not_unique_maps = self.unique_maps()
        while stop == 0 or (bool(not_unique_maps) and stop < max_stop):
            for x in not_unique_maps:
                self.maps[x[0]] = np.zeros((self.n_sub_symbol, self.resource))
                for y in range(0, self.n_sub_symbol):
                    self.maps[x[0], y, rnd.randint(0, self.resource - 1)] = 1

                # np.array([rnd.randint(0, 1) for _ in range(0, self.n_sub_symbol * self.resource)]).reshape((-1, self.resource))

            not_unique_maps = self.unique_maps()
            stop = stop + 1
        assert stop < max_stop, 'error in maps: Not every ID map is unique or valid'

    def plot_sensor_nodes(self):
        plt.figure(figsize=(7, 7))
        plt.scatter(self.sn_positions[:, 0], self.sn_positions[:, 1], marker="*")
        plt.scatter(self.base_station[0, 0], self.base_station[0, 1], s=100, marker="1")
        plt.xlabel('$x$', fontsize=14)
        plt.ylabel('$y$', fontsize=14)
        # plt.grid()
        plt.show()
