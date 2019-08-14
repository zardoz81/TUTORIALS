import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from utils import axvlines as pltX
from memory_profiler import profile
import matplotlib.pyplot as plt
from NeuroTools import stgen


class Neuron_():
    def __init__(self, V0=-65, Ie=0, num_synapses=[], neur_type=[], i=[], k=[], N_in=[], idx_in_network=[]):
        self.neur_type = neur_type  # excitatory or inhibitory
        self.idx_in_network = idx_in_network
        if self.neur_type == 2:
            self.i = i  # index of input neuron
            self.k = k  # preceding stimulus type
            self.N_in = N_in  # number in input neurons
            self.r = 0
            self.old_k = -333333
            self.spike_train = []
            self.spike_counter = 0
            self.timer = 0
        else:
            self.V_E = 0  # equilibrium potential for the excitatory synapse
            self.V_I = -80  # equilibrium potential for the inhibitory synapse
            self.I_I = 0
            self.I_E = 0

            # initialize the axon's signalaling (what neurotransmitter this neuron can release):
            self.num_synapses = num_synapses
            self.synapses = []
            for i in range(self.num_synapses):
                self.synapses.append(self.Synapse(belongs_to=self.idx_in_network, index=i))

        # neuron's electric constants:
        self.Ie = Ie if self.neur_type != 2 else None  # external current, nA
        self.tau = 10 if neur_type == 0 or neur_type == 2 else 20  # ms
        self.Vth = -55  # threshold after which an AP is fired,   mV
        self.Vr = -70  # reset voltage (after an AP is fired), mV
        self.EL = -65  # leakage potential, mV
        self.Vspike = 0
        self.AP = 0

        self.refractory_period = 1  # ms
        self.time_since_last_spike = 1

        # Intial conditions
        self.V = V0  # intial membrane voltage

    def update_r(self, k):
        N_in = self.N_in
        R_max = 110
        sigma = 0.25  ################  0.15
        L = 0.4
        i = self.i
        self.r = (R_max / L) * np.exp(-(i / N_in - 0.1 * (k)) ** 2 / (2 * (sigma * L) ** 2))

    #         print('RUN FUNCTION UPDATE_R, i: {}'.format(self.i))

    def get_spike_train(self):
        st_gen = stgen.StGen()
        T = 100000
        rate = np.array([0.8, 0.9, 1, 1.1, 1.2]) * self.r
        rate = np.array([max(rate[i], 0.1) for i in range(len(rate))])  # we don't want frequencies lower than 0.1 Hz
        time = np.linspace(0, T, 6)
        time = time[:-1]
        a = st_gen.inh_poisson_generator(rate=rate, t=time, t_stop=T, array=True)
        self.spike_train = a.T

    #         print('RUN FUNCTION get_spike_train, i: {}'.format(self.i))
    #         print('spike_train:\n, {}\n'.format(self.spike_train))


    def check4spike(self, dt, k):
        self.timer += dt
        if self.timer > self.spike_train[self.spike_counter]:
            self.AP = 1
            self.spike_counter += 1
        try:
            tmp = self.spike_train[self.spike_counter + 2]
        except IndexError:
            print('Generating new spikes')
            self.timer = 0
            self.spike_counter = 0
            self.spike_train = []
            self.update_r(k)
            self.get_spike_train()

    def step(self, dt, k=0):
        """
        MAKE ONE TIME STEP TO UPDATE THE NEURON'S PARAMETERS
        """
        if self.neur_type != 2:
            for synapse in self.synapses:
                synapse.update(dt)
            # IAF model: dV/dt is membrane current.
            g_AMPA = np.sum([self.synapses[i].g_AMPA for i in range(self.num_synapses)])
            g_NMDA = np.sum([self.synapses[i].g_NMDA for i in range(self.num_synapses)])
            self.I_E = -g_AMPA * (self.V - self.V_E) - 0.1 * g_NMDA * (self.V - self.V_E)
            g_GABA = np.sum([self.synapses[i].g_GABA for i in range(self.num_synapses)])
            self.I_I = -g_GABA * (self.V - self.V_I)
            dV = (-(self.V - self.EL) / self.tau + self.I_E + self.I_I + self.Ie) * dt

            self.time_since_last_spike += dt

            if self.AP == 1:
                self.AP = 0
                self.V = self.Vr

            # if the threshold voltage is reached, fire and action potential:
            if self.V > self.Vth and self.AP == 0 and self.time_since_last_spike >= self.refractory_period:
                self.V = self.Vspike
                self.AP = 1
                self.time_since_last_spike = 0

            if self.AP == 0:
                self.V += dV

        if self.neur_type == 2:
            if k != self.old_k:
                self.timer = 0
                self.spike_counter = 0
                self.spike_train = []
                self.update_r(k)
                self.get_spike_train()
                self.old_k = k
            self.check4spike(dt, k)
            if self.AP == 1 and self.V != self.Vspike:
                self.V = self.Vspike
            elif self.AP == 1 and self.V == self.Vspike:
                self.V = self.EL
                self.AP = 0
            else:
                pass

        return self.V

    class Synapse:
        def __init__(self, belongs_to=[], index=[]):
            self.neur_type = []
            self.incoming_AP = 0
            # synaptic conductances:
            self.g_GABA = 0
            self.g_AMPA = 0
            self.g_NMDA = 0
            # tau-parameters
            self.tau_AMPA = 8
            self.tau_NMDA = 100
            self.tau_GABA = 8
            self.W = 0  # weight
            self.listens_to = []
            self.belongs_to = belongs_to
            self.index = index

        def update(self, dt):
            kronecker = 1 if self.incoming_AP == 1 else 0
            if self.neur_type == 0:
                self.g_AMPA += (-self.g_AMPA / self.tau_AMPA + kronecker * self.W) * dt
                self.g_NMDA += (-self.g_NMDA / self.tau_NMDA + kronecker * self.W) * dt
            elif self.neur_type == 1:
                self.g_GABA += (-self.g_GABA / self.tau_GABA + kronecker * self.W) * dt
            elif self.neur_type == 2:
                self.g_AMPA += (-self.g_AMPA / self.tau_AMPA + kronecker * self.W) * dt
            else:
                pass  # here we do nothing because the synapse receives not input and hence has type []
            # print('ERROR. Belongs to: {}\t Index:{} \t Listens_to: {} \t of type:({})'.format(self.belongs_to, self.index, self.listens_to, self.neur_type))
            self.incoming_AP = 0


class Network():
    def __init__(self, num_synapses: object = 0, num_input_neurons: object = 0, num_reservoir_neurons: object = 2, num_readout_neurons: object = 2,
                 sparsity: object = 0) -> object:
        # DEFINE SIMULATION PARAMETERS:
        self.N_in = num_input_neurons
        self.N_ro = num_readout_neurons
        self.N = num_reservoir_neurons  # number of reservoir neurons in the network
        self.M = num_synapses  # number of synapses a neuron has (not all of them have to receive connections from upstream neurons)
        self.S = sparsity  # network's sparsity (how many synapses in the network receive no input from upstream neurons)
        self.dt = 0.01  # time step in milliseconds

        # create a list of neuron objects and populate it:
        self.neurons, idx = [], 0
        neur_types = np.random.choice([0, 1], p=[0.2, 0.8], size=self.N)
        # instantiate input neurons:
        for i in range(self.N_in):
            self.neurons.append(Neuron_(N_in=self.N_in, i=i, neur_type=2, idx_in_network=idx))
            idx += 1
        # instantiate reservoir neurons:
        for i in range(self.N):
            self.neurons.append(Neuron_(Ie=0, num_synapses=self.M, neur_type=neur_types[i], idx_in_network=idx))
            idx += 1
        # instantiate readout neurons:
        for i in range(self.N_ro):
            self.neurons.append(Neuron_(Ie=0, num_synapses=self.M, neur_type=3, idx_in_network=idx))
            idx += 1

        # Define network topology:
        self.route = self.define_topology(self.N_in, self.N_ro, self.N, self.M, self.S)
        self.route = self.route * np.random.rand(*self.route.shape)  # randomize weights
        self.update_topology()  # define weights and assign them to all synapses in all neurons

    def update_topology(self, verbose=0):
        # you should call this function to update the topology after manually changing it:
        # clear synapses in all but input neurons (because they don't have synapses):
        for i in range(self.N_in, len(self.neurons)):
            for j in range(self.M):
                self.neurons[i].synapses[j].listens_to = []
        for postsynaptic in range(self.N_in, len(self.neurons)):  # INPUT NEURONS CANNNOT RECEIVE PROJECTIONS
            listens_to = np.nonzero(self.route[:, postsynaptic, :])[0].tolist()
            self.neurons[postsynaptic].listens_to = listens_to
            for presynaptic in range(len(self.neurons)):  # ANY NEURON CAN PROJECT to any other
                target_synapse = np.nonzero(self.route[presynaptic, postsynaptic, :])[0].tolist()
                for i in target_synapse:
                    if verbose == 1:
                        print('Receive {} Send {} syn_idx {} val {}'.format(postsynaptic, presynaptic, i,
                                                                            self.route[presynaptic, postsynaptic, i]))
                    self.neurons[postsynaptic].synapses[i].W = self.route[presynaptic, postsynaptic, i]
                    self.neurons[postsynaptic].synapses[i].neur_type = self.neurons[presynaptic].neur_type
                    self.neurons[postsynaptic].synapses[i].listens_to = presynaptic

    def define_topology(self, N_in, N_ro, N, nSyn, sparsity=0.0):
        N_tot = N_in + N + N_ro
        route = np.random.choice([0, 1], p=[sparsity, 1 - sparsity], size=(N_tot, N_tot, nSyn))
        route[:, :N_in, :] = 0  # input neurons don't synapse on input neurons
        route[N_in + N:, N_in + N:, :] = 0  # readout neurons don't synapse on readout neurons
        route[:N_in, N_in + N:, :] = 0  # input neurons don't synapse on readout neurons
        for i in range(N_tot):
            route[i, i, :] = 0  # prohibit autapses
        a = np.array(np.arange(0, N_tot))
        for i in range(nSyn):
            b = np.delete(a, i)
            route[b, :, i] = 0
        return route

    def summary(self):
        # Show network config and topology:
        print(
            'NET PARAMETERS:\nReservior neurons:\t{}\nInput neurons:\t{}\nReadout neurons:\t{}\nSynapses:\t{}\nSparsity:\t{}\ndt:\t\t{}\n'.format(
                self.N, self.N_in, self.N_ro, self.M, self.S, self.dt))

        types = ['exc' if n.neur_type == 0
                 else ('inp' if n.neur_type == 2
                       else ('inh' if n.neur_type == 1
                             else 'RO'))
                 for n in net.neurons]

        if self.N < 10:
            if self.N >= 2:
                plt.figure(figsize=(15, 15))
            for i in range(self.M):
                plt.subplot(round(len(self.neurons) / 1.8), 5, i + 1)
                plt.imshow(self.route[:, :, i])
                if i == 0:
                    plt.ylabel('Signaling neurons')
                plt.xlabel('Receiving neurons')
                plt.title('Receiving synapse ' + str(i))
                ax = plt.gca()
                tmp = ax.set_yticks(np.arange(0, len(net.neurons)))
                tmp = ax.set_xticks(np.arange(0, len(net.neurons)))
                tmp = ax.set_yticklabels(enumerate(types))
            plt.tight_layout()

            for i in range(self.N_in, len(self.neurons)):
                for j in range(len(self.neurons[i].synapses)):
                    cur_type = self.neurons[i].synapses[j].neur_type
                    print('{} neuron #: {}, \t Receives {}\t on synapse #: {}\t from neuron: {}'.format(
                        'EXC' if self.neurons[i].neur_type == 0 else 'INH',
                        i,
                        'exc. input' if cur_type == 0 else (
                            'inh. input' if cur_type == 1 else (
                                'Inp. input' if cur_type == 2 else(
                                    'RO feedback' if cur_type == 3 else 'Nothing'))),
                        j,
                        self.neurons[i].synapses[j].listens_to))

    def step(self, dt, k=1):
        VV = np.zeros((len(self.neurons),))
        for n in range(len(self.neurons)):
            CN = self.neurons[n]
            VV[n] = CN.step(dt, k=k)
        # tell relevant synapses in relevant downstream neurons which upstream neurons have fired:
        for source in range(len(self.neurons)):
            for target in range(len(self.neurons)):
                target_synapses = np.nonzero(self.route[source, target, :])[0].tolist()
                for target_synapse in target_synapses:
                    if self.neurons[source].AP == 1:
                        self.neurons[target].synapses[target_synapse].incoming_AP = 1
        return VV

@profile
def execute():
    net = Network(num_synapses=20,
                  num_input_neurons=3,
                  num_reservoir_neurons=16,
                  num_readout_neurons=1,
                  sparsity=0.6)
    print('Neuron 0 type:{}'.format(net.neurons[0].neur_type))
    del(net)
    net1 = Network(num_synapses=20,
                  num_input_neurons=3,
                  num_reservoir_neurons=16,
                  num_readout_neurons=1,
                  sparsity=0.6)
    print('Neuron 0 type:{}'.format(net1.neurons[0].neur_type))
    net2 = Network(num_synapses=20,
                  num_input_neurons=3,
                  num_reservoir_neurons=16,
                  num_readout_neurons=1,
                  sparsity=0.6)
    print('Neuron 0 type:{}'.format(net2.neurons[0].neur_type))

execute()



