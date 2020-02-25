#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:05:21 2019

@author: romankoshkin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#from scipy.special import softmax
from scipy.ndimage.filters import gaussian_filter1d as gauss
from sklearn.metrics import auc

def times2spiketrain(times, spike_times):
    return signal.unit_impulse(len(times), np.nonzero(np.isin(times,spike_times))[0].tolist())

def getV(dt, tau, spike_train):
    V = [0]
    for activity in spike_train:
        dV = dt/tau*(-V[-1] + activity)
        V.append(V[-1] + dV)
    return np.array(V[:-1]).flatten() * tau * srate

def norm_gauss(out, ker_len_s, norm=True):
    gg = gauss(out, ker_len_s)
    if norm==False:
        return gg
    if norm==True:
        return gg/np.max(gg)

srate = 100
T = 10
dt = 1/srate
times = np.round(np.arange(0,T, dt), 2)

threshold = 1.6
tau = 3
refractory_period = 0.2
ker_len_s = np.round(0.1/dt, 2)


spike_times_0 = [1.0, 1.2, 2.1, 8.1, 8.2, 9.0]
spike_times_1 = [1.4, 2.2, 3.2]
spike_times_2 = [0.4, 0.8, 0.9, 4.2, 5.1]

spikeTrain0 = times2spiketrain(times, spike_times_0)
spikeTrain1 = times2spiketrain(times, spike_times_1)
spikeTrain2 = times2spiketrain(times, spike_times_2)
ST = [spikeTrain0, spikeTrain1, spikeTrain2]

ww = np.array([[0.05, 0.05, 0.9],
               [0.05, 0.9, 0.05],
               [0.9, 0.05, 0.05]])

class net():
    def __init__(self, n_neurons=3, in_size=3, lrate=0.01):
        self.lrate = lrate
        self.n_neurons = n_neurons
        self.in_size = in_size
        self.W = np.zeros((n_neurons, in_size))
        self.weighted_inputs = []
        self.r = np.nan
        self.srate = 100
        self.threshold = 1.6
        self.refractory_period = 0.2
        self.r = 0
        self.fitness = []
           
    def fwd(self, list_of_activations):
        self.fitness = []
        self.AUC = []
        self.E = []
        SS = []
        for neuron in range(self.W.shape[0]):
            ss = np.dot(self.W[neuron,:], np.vstack(list_of_activations))
            self.weighted_inputs.append(np.copy(ss))
            i = 0
            while i < len(ss):
                if ss[i] > self.threshold:
                    ss[i] = 1
                    ss[(i+1):(i+int(self.refractory_period * self.srate))] = 0
                    i += int(self.refractory_period * self.srate)
                else:
                    ss[i] = 0
                    i += 1
            SS.append(ss)
            self.fitness.append(auc(times, norm_gauss(ss, ker_len_s) * norm_gauss(self.r, ker_len_s)))
            self.AUC.append([])
            self.E.append([])
            for syn in range(self.in_size):
                e = norm_gauss(ss, ker_len_s) * norm_gauss(self.r, ker_len_s) * list_of_activations[syn]
                self.E[neuron].append(e)
                self.AUC[neuron].append(auc(times, e))
        return np.stack(SS)
    
    def optimize(self):
        for neuron in range(self.n_neurons):
            max_e = np.argmax(self.AUC[neuron])
            if self.W[neuron, max_e] + self.lrate <= 1:
                self.W[neuron, max_e] += self.lrate
                losers_idx = np.nonzero(np.arange(self.in_size) != max_e)[0]
                x = (1 - self.W[neuron, max_e])/np.sum([self.W[neuron, j] for j in losers_idx])
                for i, loser_idx in enumerate(losers_idx):
                    self.W[neuron, loser_idx] *= x
        
n = net(n_neurons=3, in_size=3)
n.W = np.copy(ww)
n.r = times2spiketrain(times, np.round(np.arange(2, 4, dt),2))

out = n.fwd([getV(dt, tau, spikeTrain0), getV(dt, tau, spikeTrain1), getV(dt, tau, spikeTrain2)])
out_ = np.copy(out)


fitness = []
for i in range(140):
    out = n.fwd([getV(dt, tau, spikeTrain0), getV(dt, tau, spikeTrain1), getV(dt, tau, spikeTrain2)])
    fitness.append(n.fitness)
    n.optimize()

col = ['blue', 'red', 'black']
plt.figure(figsize=(15,3))
for i in range(len(ST)):
#     plt.plot(times, getV(dt, tau, ST[i]), c=col[i])
# plt.tight_layout()
    plt.plot(times, 0.1*out_[i]+i, alpha=0.2, c=col[i])
    plt.plot(times, 0.1*out[i]+i, c=col[i])
    
    
plt.figure(figsize=(15,5))
for neuron in range(3):
    plt.subplot(3,1,neuron+1)
    plt.plot(times, norm_gauss(out[neuron,:], ker_len_s))
    plt.plot(times, n.weighted_inputs[neuron], c=col[neuron])
    plt.plot(times, norm_gauss(n.r, ker_len_s))
    for syn in range(3):
        plt.fill_between(times, 
                     n.E[neuron][syn],
                     alpha=0.2, label='e_AUC on syn {} is {:.2f}'.format(syn, n.AUC[neuron][syn]))
    plt.legend()
    plt.plot(times, -0.2 * out[neuron,:], c=col[i])
    plt.axhline(n.threshold, linewidth=1, linestyle='dotted')
    plt.title('Neuron: {}, Fitness: {:.3f}'.format(neuron, n.fitness[neuron]))
plt.tight_layout()
plt.figure()
plt.plot(fitness)
