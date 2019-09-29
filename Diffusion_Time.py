import numpy as np
import matplotlib.pyplot as plt

def ismember(a, b, mode='straight'):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    x = [bind.get(itm, None) for itm in a]
    if mode=='straight':
        x = [i for i, e in enumerate(x) if e is not None]
    if mode=='inverse':
        x = [i for i, e in enumerate(x) if e is None]
    return x

def get_mat(free, n_vesicles, n_sensors, n_Ca):
    l = n_vesicles*n_sensors
    i = np.random.choice(free, n_Ca, replace=False)
    mat = np.zeros((l,))
    mat[i] = 1
    mat = mat.reshape((n_vesicles, n_sensors))
    return mat


class synapse():
    def __init__(self, n_vesicles=0, num_ca_chan=0):
        self.n_vesicles = n_vesicles
        self.num_ca_chan = num_ca_chan
        self.vesicles = []
        for i in range(n_vesicles):
            self.vesicles.append(vesicle(num_ca_chan=num_ca_chan))
        self.blocked_idx = []
        self.free = []
        
    def count_release(self):
        c = 0
        for vesicle in self.vesicles:
            c += 1 if vesicle.bound_ch.sum() == self.num_ca_chan else 0
        return c
            
    def apply_blocker(self, amount_of_blocker):
        l = np.arange(self.n_vesicles * self.num_ca_chan)
        i = np.random.choice(l, amount_of_blocker, replace=False)
        self.blocked_idx = i
#         print('self.blocked', self.blocked_idx)
        self.free = list(set(l) - set(S.blocked_idx))
#         print('self.free', self.free)
        blocked = np.zeros((len(l),))
        blocked[i] = 0
        blocked = blocked.reshape((self.n_vesicles, self.num_ca_chan))
        for j, vesicle in enumerate(self.vesicles):
            vesicle.blocked_ch = blocked[j,:].flatten()
            
    def release_ca(self, amount_of_ca):
        # released ca will bind to free Ca2+ channels:
        l = np.arange(self.n_vesicles * self.num_ca_chan)
        i = np.random.choice(self.free, amount_of_ca, replace=False)
        bound = np.zeros((len(l),))
        bound[i] = 1
        bound = bound.reshape((self.n_vesicles, self.num_ca_chan))
        for j, vesicle in enumerate(self.vesicles):
            vesicle.bound_ch = bound[j,:].flatten()
            
            
class vesicle():
    def __init__(self, num_ca_chan=3):
        self.num_ca_chan = num_ca_chan
        self.chan_states = np.zeros((self.num_ca_chan, ))
        self.trans_rel = 0
        self.blocked_ch = np.zeros((self.num_ca_chan, ))
        self.bound_ch = np.zeros((self.num_ca_chan, ))


n_vesicles = 1000         
num_ca_chan = 1           
S = synapse(n_vesicles=n_vesicles, num_ca_chan=num_ca_chan)
amount_of_blocker = int((n_vesicles * num_ca_chan)/50)
S.apply_blocker(amount_of_blocker)
R = []
CA = []
for ca in range(0,3000,10):
    amount_of_ca = min(ca, int(n_vesicles * num_ca_chan - amount_of_blocker))
    S.release_ca(amount_of_ca)
    R.append(S.count_release())
    CA.append(amount_of_ca)
    CA_ = np.array(CA)/np.max(CA)

plt.plot(CA_, R)
plt.title('Amount of blocker: {}'.format(amount_of_blocker))
plt.xlabel('$\mathbf{Ca}^{2+}$')
plt.ylabel('Transmitter release')


n_vesicles = 1000         
num_ca_chan = 2           
S = synapse(n_vesicles=n_vesicles, num_ca_chan=num_ca_chan)
amount_of_blocker = int((n_vesicles * num_ca_chan)/50)
S.apply_blocker(amount_of_blocker)
R = []
CA = []
for ca in range(0,3000,10):
    amount_of_ca = min(ca, int(n_vesicles * num_ca_chan - amount_of_blocker))
    S.release_ca(amount_of_ca)
    R.append(S.count_release())
    CA.append(amount_of_ca)
    CA_ = np.array(CA)/np.max(CA)
plt.plot(CA_, R)
plt.title('Amount of blocker: {}'.format(amount_of_blocker))
plt.xlabel('$\mathbf{Ca}^{2+}$')
plt.ylabel('Transmitter release')


n_vesicles = 1000         
num_ca_chan = 3           
S = synapse(n_vesicles=n_vesicles, num_ca_chan=num_ca_chan)
amount_of_blocker = int((n_vesicles * num_ca_chan)/50)
S.apply_blocker(amount_of_blocker)
R = []
CA = []
for ca in range(0,3000,10):
    amount_of_ca = min(ca, int(n_vesicles * num_ca_chan - amount_of_blocker))
    S.release_ca(amount_of_ca)
    R.append(S.count_release())
    CA.append(amount_of_ca)
    CA_ = np.array(CA)/np.max(CA)
plt.plot(CA_, R)

n_vesicles = 1000         
num_ca_chan = 5         
S = synapse(n_vesicles=n_vesicles, num_ca_chan=num_ca_chan)
amount_of_blocker = int((n_vesicles * num_ca_chan)/50)
S.apply_blocker(amount_of_blocker)
R = []
CA = []
for ca in range(0,5000,10):
    amount_of_ca = min(ca, int(n_vesicles * num_ca_chan - amount_of_blocker))
    S.release_ca(amount_of_ca)
    R.append(S.count_release())
    CA.append(amount_of_ca)
    CA_ = np.array(CA)/np.max(CA)
plt.plot(CA_, R)

plt.title('Amount of blocker: {}'.format(amount_of_blocker))
plt.xlabel('$\mathbf{Ca}^{2+}$')
plt.ylabel('Transmitter release')
plt.legend(['1', '2', '3', '5'], title='Open channels per visicle \n needed transmitter release')
plt.show()