import numpy as np
import matplotlib.pyplot as plt
from numpy.random import standard_normal
import random as rnd


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


class SFCChannel(PositionSensorNodes):
    def __init__(self, base_station=np.array([[0, 0]]), sensor_nodes=64, carrier_frequency=2.4e9, n_sub_symbol=6, resource=7, **options):
        self.name = options.pop('name', 'Generic')
        self.base_station = base_station
        self.carrier_frequency = options.pop('carrier_frequency', carrier_frequency)
        self.fading = options.pop('fading', None)
        self.path_loss_exp = options.pop('path_loss_exp', 0)  # 2 for for propagation in free space
        self.bandwidth = options.pop('bandwidth', 1000)
        self.energy_slot_time = 2 / (self.bandwidth / resource)
        self.sn_movement_type = options.pop('sn_movement_type', 'module')
        self.n_sub_symbol = n_sub_symbol
        self.resource = resource
        self.map_class = 'random'
        # self.maps = np.zeros((sensor_nodes, n_sub_symbol, resource))
        self.sensor_nodes = sensor_nodes
        self.sensor_x_event = options.pop('sensor_x_event', [])  # np.identity(sensor_nodes)
        self.N0 = options.pop('N0', 0)

        if self.sn_movement_type == 'gaussian':
            self.sn_mov_std = options.pop('sn_mov_std', 10)

        super().__init__(sensor_nodes=sensor_nodes, energy_slot_time=self.energy_slot_time, sn_movement_type=self.sn_movement_type,
                         **options)

        if self.base_station.shape != (1, 2):
            self.base_station = np.array([[0, 0]])

        self.path_loss = self.get_impulse()

        self.tx_amplitude = options.pop('tx_amplitude', 10 * np.ones((sensor_nodes, 1)))

        if self.tx_amplitude.shape != (sensor_nodes, 1):
            self.tx_amplitude = 10 * np.ones((sensor_nodes, 1))
            self.dc_threshold = options.pop('dc_threshold', 5)

        if options.pop('power_at_receiver', False):
            self.tx_amplitude = self.tx_amplitude / np.abs(self.path_loss)
        # self.AWGN_std = options.pop('AWGN_std', 0.01)

        # self.tx_maps()
        self.dc_threshold = options.pop('dc_threshold', 0.5 * np.min(np.sqrt(self.tx_amplitude * np.abs(self.path_loss))))

    def __call__(self, events, **options):

        if len(self.sensor_x_event) == 0:
            self.sensor_x_event = options.pop('sensor_x_event', np.identity(self.sensor_nodes))
            self.maps = np.zeros((events.shape[1], self.n_sub_symbol, self.resource))
            self.tx_maps()

        if not ('maps' in dir(self)):
            self.maps = np.zeros((events.shape[1], self.n_sub_symbol, self.resource))
            self.tx_maps()

        assert self.sensor_x_event.shape[1] == events.shape[1], 'error in parameter shape: events '
        assert self.maps.shape[0] == events.shape[1], 'error in parameter shape: events '

        N = events.shape[0]
        received_signal = np.complex_(np.zeros((N + self.n_sub_symbol - 1, self.resource)))

        for e in range(0, N):
            inds = np.argwhere(events[e] == 1)
            for ind in inds:
                inds_sensor = np.argwhere(self.sensor_x_event[:, ind])[:, 0]
                for ind_sensor in inds_sensor:
                    received_signal[e:e + self.n_sub_symbol, :] = received_signal[e:e + self.n_sub_symbol, :] + self.maps[ind] * self.path_loss[
                        ind_sensor] * np.sqrt(self.tx_amplitude[ind_sensor])

        received_signal = received_signal + np.random.normal(0, np.sqrt(0.5 * self.N0), size=received_signal.shape)
        received_signal = received_signal + 1j * np.random.normal(0, np.sqrt(0.5 * self.N0), size=received_signal.shape)
        received_signal = np.abs(received_signal)
        rx_map = np.zeros(received_signal.shape)
        rx_map[np.argwhere(received_signal > self.dc_threshold)[:, 0], np.argwhere(received_signal > self.dc_threshold)[:, 1]] = 1

        rx_events = np.zeros(events.shape)
        for e in range(0, N):
            for i in range(0, events.shape[1]):
                rx_events[e, i] = 1 if np.sum(rx_map[e:e + self.n_sub_symbol, :] * self.maps[i, :, :]) == self.n_sub_symbol else 0

        if options.pop('update', False):
            self.update(n_steps=options.pop('n_steps', 1))

        return rx_events, rx_map, received_signal

    def update(self, n_steps=1, **options):
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

    def tx_maps(self):
        assert len(self.sensor_x_event) != 0, 'error in sensor_x_event map: No sensor_x_event map is set'

        L_aux = np.arange(0, self.resource, np.floor(self.resource / self.n_sub_symbol))
        groupSize = np.arange(np.floor(self.resource / self.n_sub_symbol)-1, self.resource, np.floor(self.resource / self.n_sub_symbol))
        groupSize[-1] = self.resource - 1

        if self.maps.shape[0] > (np.floor(self.resource / self.n_sub_symbol)**(self.n_sub_symbol-1)) * (groupSize[-1] - groupSize[-2]):
            print('Not every ID is unique')

        for index in range(self.maps.shape[0]):
            for index1 in range(self.maps.shape[1]):
                self.maps[index, index1, int(L_aux[index1])] = 1
            aux = 0
            L_aux[aux] = L_aux[aux] + 1
            while L_aux[aux] > groupSize[aux]:
                L_aux[aux] = L_aux[aux] - np.floor(self.resource / self.n_sub_symbol)
                aux = aux + 1
                if aux < self.n_sub_symbol:
                    L_aux[aux] = L_aux[aux] + 1
                else:
                    L_aux = np.arange(0, self.resource, np.floor(self.resource / self.n_sub_symbol))
                    aux = 0

        return self.unique_maps()

        # assert stop < max_stop, 'error in maps: Not every ID map is unique or valid'

    def plot_sensor_nodes(self):
        plt.figure(figsize=(7, 7))
        plt.scatter(self.sn_positions[:, 0], self.sn_positions[:, 1], marker="*")
        plt.scatter(self.base_station[0, 0], self.base_station[0, 1], s=100, marker="1")
        plt.xlabel('$x$', fontsize=14)
        plt.ylabel('$y$', fontsize=14)
        # plt.grid()
        plt.show()
