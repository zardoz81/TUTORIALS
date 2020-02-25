import nest
from sklearn.datasets import fetch_openml
if not 'mnist' in locals():
    mnist = fetch_openml('mnist_784', version=1, cache=True)

import matplotlib.pyplot as plt

import scipy.stats as ss
from scipy.signal import convolve2d as conv2d
from sklearn.preprocessing import minmax_scale

from skimage.transform import resize as imresize
from skimage.color import rgb2gray
from skimage.util.shape import view_as_windows

from utils.printProgressBar import printProgressBar
from utils.AddFig2Movie import AddFig2Movie
import copy
import numpy as np


def gabor(sigma=1, theta=0, Lambda=1, psi=0, gamma=1):
    """
    Gabor feature extraction.
    sigma - length of a standard deviation (pixels)
    lambda - wavelength
    psi - phase shift
    gamma - ellipticity
    
    Usage:
    gabor(sigma=5, theta=np.pi/4, Lambda=6, psi=0, gamma=1)
    will give you a round (not elliptical) gabor filter of size (5,5) and a wavelength of 6
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 1  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def get_scaled(im, imsize, pad=False):
    IM = []
    im = rgb2gray(im)
    im = imresize(im, imsize)
    org_s = im.shape[0]
    percentages = [1, 0.75, 0.5, 0.35, 0.25]
    for p in percentages:
        s = int(np.sqrt(im.flatten().shape[0]*p)) # сторона квадрата
        im_ = imresize(im, (s, s))
        if pad:
            pad_with = int(np.floor((org_s-s)/2)) + 1
            im_ = np.pad(im_, pad_width=pad_with, mode='constant', constant_values=0)
            print(im_.shape)
            im_ = im_[:imsize[0], :imsize[1]]
        IM.append(im_)
    return IM

def get_filters(THETAS):
    sigma = 2
    Lambda = 5
    gamma = 1
    G = []
    for theta in THETAS:
        G.append(gabor(sigma=sigma, theta=theta, Lambda=Lambda, psi=0, gamma=gamma))
    return G

def get_S1_maps(scaled_IM, kernels, input_size):
    S1maps = {}
    for im in range(len(scaled_IM)): # loop over the scaled versions of the image
        S1maps['scale'+str(im)] = np.zeros((4,) + scaled_IM[im].shape)
        for k in range(len(kernels)):# loop over the filters
            S1map = minmax_scale(conv2d(scaled_IM[im], kernels[k], mode='same').flatten(), (0, 255-0.01)).reshape(scaled_IM[im].shape)
            S1maps['scale'+str(im)][k,:,:] = S1map
    return S1maps

def get_C1_maps(S1maps):
    C1maps = {}
    for scale in S1maps.keys():
        C1maps[scale] = []    # init empty list for one processing scale
        for orientation in range(S1maps[scale].shape[0]):
            # break into receptive fields (windows of size 7 with stride 6)
            X = view_as_windows(S1maps[scale][orientation,:,:], window_shape = (7,7), step=(6,6))
            C1map = np.zeros((X.shape[0], X.shape[1]))
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X_r = ss.rankdata(X[i,j,:,:].flatten(), method='ordinal').reshape(7,7)
                    C1map[i,j] = X[(i,j) + np.where(X_r==X_r.max())]
            C1maps[scale].append(C1map) # append to the current processing scale another orientation map
        C1maps[scale] = np.stack(C1maps[scale], axis=0)
    return C1maps

def inv_gauss_mask(size, fwhm=3, center=None):
    """ Make a square gaussian kernel. 
    size is the length of a side of the square 
    fwhm is full-width-half-maximum, which can be thought of as an effective radius."""
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    g = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2) + 1
    g[x0, y0] = 1
    return g

def encode_C1maps(C1maps, snap=2): 
    """ covert pixel intensities to lantecies """
    C1maps_ = copy.deepcopy(C1maps)
    for k in C1maps_.keys():
        for j in range(C1maps_[k].shape[0]):
            C1maps_[k][j] = np.abs(np.round(255 - C1maps_[k][j], snap)) # snap to 0.01 ms steps
    return C1maps_

def inhibit_laterals(input_im, out, span=5, min_att=1.05, max_att=1.15, t=-1):
    X = np.copy(input_im)
    if t >= 0:
        I, J = np.where(X==t)
        if I.size == 0:
            I, J = np.where(X==np.min(X))
            return X, I[0], J[0], np.min(X[X>t])
    if t==-1:
        I, J = np.where(X==np.amin(X))
    factor = np.linspace(min_att, max_att, span).tolist() + [1.0] + np.linspace(max_att, min_att, span).tolist()
    
    g = inv_gauss_mask(11, fwhm = 6, center=None)

    for c0,i in enumerate(np.arange(I[0]-span, I[0]+span+1)):
        for c1,j in enumerate(np.arange(J[0]-span, J[0]+span+1)):
            if i>=0 and i<X.shape[0] and j>=0 and j<X.shape[1]:
                f = g[c0,c1]
                if X[i,j] >= t:
                    X[i,j] = np.round(X[i,j]*f, 2)
            else:
                continue
    next_spike_t = np.min(X[X>t])
    return X, J[0], I[0], next_spike_t


def get_fully_inhibited_C1(C1maps_encoded):
    """Takes encoded C1 map. Runs lateral inhibition on a C1 map. The spikes that get inhibited
       beyond 250 are simply removed"""
    printProgressBar(0, 20, prefix='Progress:', suffix='Complete', length=50)
    C1maps_ = copy.deepcopy(C1maps_encoded)
    c = 0
    for scale in C1maps.keys():
        for filt in range(4):
            X_ = np.copy(C1maps_[scale][filt])      
            out = []
            t = 0.0
            next_spike_t = 0.0
            while next_spike_t < 255:
                X_, i, j, next_spike_t = inhibit_laterals(X_, out, span=5, min_att=1.05, max_att=1.55, t=t)
                t = np.round(t+0.01, 2)
                out.append((X_, i, j, next_spike_t, t))
            C1maps_[scale][filt] = X_
            c += 1
            printProgressBar(c, 20, prefix='Progress:', suffix='Sc: {}, Filt: {}, max: {:.2f}'.format(scale, filt, np.max(C1maps_[scale][filt])), length=50)
    return C1maps_

def new_simulate(T, dt, neuron_type, n, n_spike_generators, weights, in_spike_train):
    """implements online STDP. I don't exactly know now it works, but it does something"""
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt})
    
    iaf_neuron_parms = {
     'C_m': 1000.0, # pF
     'E_L': -65.0, # resting potential
     'I_e': 1000.0,
     't_ref': 5.0,
     'tau_m': 10.0,
     'tau_syn_ex': 1.0,
     'tau_syn_in': 2.0,
     'V_m': -60.0, # initial potential?
     'V_reset': -65.0,
     'V_th': -55.0}
    
    neurons = nest.Create(neuron_type, n)   
    spikedetectors = nest.Create('spike_detector', n)
    spikegenerators = nest.Create('spike_generator', n_spike_generators)
    voltmeters = nest.Create('voltmeter', n, params={"interval": dt})
    multimeter = nest.Create('multimeter', 1,  params = {
        'withtime':True,
        'interval':dt,
        'record_from':['V_m', 'I_syn_ex', 'I_syn_in']})
    pre_pop = nest.Create('parrot_neuron', n_spike_generators)
    
    nest.Connect(spikegenerators, pre_pop, "one_to_one")
    nest.Connect(pre_pop, neurons,
                 conn_spec='all_to_all',
                 syn_spec={'model': 'static_synapse',
                           'weight': weights,
                           'delay': 0.01}
                )
    nest.Connect(multimeter, neurons, 'one_to_one')
    nest.Connect(voltmeters, neurons, 'one_to_one')
    nest.Connect(neurons, spikedetectors, 'one_to_one')
    
    if neuron_type=='iaf_psc_alpha':
        nest.SetStatus(neurons, params=iaf_neuron_parms)
    c = 0

    
    for i in spikegenerators:
        nest.SetStatus([i], {'spike_times': np.array(in_spike_train[c]).flatten()})
        c += 1
#     nest.SetStatus([neurons[0]], {'I_e':0.0})

    nest.Simulate(T+dt)
    weights_ = np.array(nest.GetStatus(nest.GetConnections(pre_pop, neurons), keys='weight')).reshape(1,-1)
    times = nest.GetStatus(voltmeters)[0]['events']['times']
    return times, voltmeters, spikedetectors, multimeter, spikegenerators, neurons, pre_pop, weights_

def init_weights():
    W = {}
    for scale in C1maps_inh.keys():
        W[scale] = np.random.rand((*C1maps_inh[scale].shape))*1
#         W[scale] = np.ones_like(C1maps_inh[scale]) * 450
    return W

def get_winning_scale(out_ST):
    ealiest_spikes_in_scales = [np.min(i) for i in out_ST.values()]
    winning_scale_id = np.argmin(ealiest_spikes_in_scales)
    winning_scale_name = list(out_ST.keys())[winning_scale_id] # where the earliest spike is
    winner_spike = np.round(np.min(out_ST[winning_scale_name]), 2)
    return winning_scale_name, winner_spike

def get_S2(W=[], C1maps_inh=[], T=250.0, dt=0.01, neuron_type='iaf_psc_alpha', n=1):
    voltage, out_ST = {}, {}
    for scale in C1maps_inh.keys():
        weights = W[scale].reshape(1,-1)
        in_spike_train = np.sort(C1maps_inh[scale].flatten())
        n_spike_generators = in_spike_train.shape[0]
        times, voltmeters, spikedetectors, multimeter, spikegenerators, neurons, pre_pop, weights = new_simulate(T, dt, neuron_type, n, n_spike_generators, weights, in_spike_train)
        voltage[scale] = nest.GetStatus(voltmeters)[0]['events']['V_m']
        # ST_trunc = times[np.isin(times, in_spike_train)]
        out_ST[scale] = nest.GetStatus(spikedetectors)[0]['events']['times']
    return out_ST, voltage

def STDP(W, out_ST, C1maps_inh):
    ws_t = get_winning_scale(out_ST)[1] # latency of the earliest spike
    DW = {}
    for scale in C1maps_inh.keys():
        t_j = C1maps_inh[scale] # presynaptic spike times
        # t_i = out_ST[scale]
        diff = t_j - ws_t # get who late or early the presynaptic spikes are relative to the winner postsynaptic spike
        a = np.zeros_like(diff) # make a vector of learning rates 
        a_plus = 0.01
        a_minus = -0.75 * a_plus
        a[diff<=0] = a_plus # (posive for those presyn spikes that preceed (i.e. cause) the winner)
        a[diff>0] = a_minus # (negative for those presyn spikes that trail (i.e. not cause) the winner)
        dw = a * W[scale] * (1-W[scale])
        W[scale] += dw
        DW[scale] = np.copy(dw)
    return W, DW

#################################################


input_size = (128, 128)
T = 250.0 # ms
dt = 0.01 # ms
# neuron_type = 'hh_psc_alpha'
neuron_type = 'iaf_psc_alpha'
n = 1                                       # number of SNN outputs
nframes = 500
THETAS = [np.pi/8, np.pi/4 + np.pi/8, np.pi/2 + np.pi/8, 3*np.pi/4 + np.pi/8]


# im = skimage.data.coffee()
im = mnist['data'][10].reshape(28,28)

G = get_filters(THETAS)           # get the FOUR spatial filters
IM = get_scaled(im, input_size, pad=False)   # returns a list of scaled version of the input image
S1maps = get_S1_maps(IM, G, input_size) # returns an array of size (n_ker, n_scales, H, W)
C1maps = get_C1_maps(S1maps)
C1maps_encoded = encode_C1maps(C1maps, snap=2)
C1maps_inh = get_fully_inhibited_C1(C1maps_encoded)
W = init_weights()
S2, voltage = get_S2(W=W, C1maps_inh=C1maps_inh, T=T, dt=dt, neuron_type=neuron_type, n=n)

plt.ioff()
for i in range(nframes):
    W, DW = STDP(W, S2, C1maps_inh)
    fig = plt.figure(figsize=(18,10))

    plt.subplot(2,3,1)
    plt.imshow(C1maps_inh['scale0'][0,:,:])
    plt.colorbar()
    plt.title('C1 map')

    plt.subplot(2,3,2)
    plt.imshow(DW['scale0'][0,:,:])
    plt.colorbar()
    plt.title('dw')

    plt.subplot(2,3,3)
    plt.imshow(W['scale0'][0,:,:])
    plt.colorbar()
    plt.title('Wts in orienation 0')
    
    plt.subplot(2,3,4)
    plt.imshow(W['scale0'][1,:,:])
    plt.colorbar()
    plt.title('Wts in orienation 1')
    
    plt.subplot(2,3,5)
    plt.imshow(W['scale0'][2,:,:])
    plt.colorbar()
    plt.title('Wts in orienation 2')
    
    plt.subplot(2,3,6)
    plt.imshow(W['scale0'][3,:,:])
    plt.colorbar()
    plt.title('Wts in orienation 3')
    
    AddFig2Movie(fig, i, 50, nframes)