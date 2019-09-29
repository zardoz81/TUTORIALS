class Net2():
    def __init__(self, N=4, T=100, dt=0.01, seed=10):
        
        np.random.seed(seed)
        self.N = N
        self.T = T
        self.dt = dt
        self.eligibility_lr = 0.1
        self.refractory_period = 1
        self.AP = np.zeros((self.N,1))

        # equilibrium potentials:
        self.V_E = 0
        self.V_I = -80    # equilibrium potential for the inhibitory synapse
        self.EL = -70      # leakage potential, mV
        
        # critical voltages:
        self.Vth = -45    # threshold after which an AP is fired,   mV
        self.Vr = -60     # reset voltage (after an AP is fired), mV
        self.Vspike = 10
        

        # define neuron types in the network:
        self.neur_type_mask = np.random.choice([0,1], self.N, p=[0.2, 0.8]).reshape(*self.AP.shape).astype('float64')
        self.neur_type_mask[0] = 1
        self.neur_type_mask[1] = 1
        self.neur_type_mask[-1] = 0
        self.neur_type_mask[-2] = 0
        

        # taus
        self.tau = np.zeros((1, self.N))
        self.tau[0, np.where(self.neur_type_mask==0)[0]] = 10
        self.tau[0, np.where(self.neur_type_mask==1)[0]] = 20

        self.tau_ampa = 8
        self.tau_nmda = 100
        self.tau_gaba = 8
        self.postsyn_tau = 80
        self.presyn_tau = 5
        self.tau_eligibility = 40
        
        self.V = np.ones((1,self.N)) * self.EL
        self.init_4_1_trial()
        
        
        # define weights:
        self.w = np.ones((self.N,self.N)).astype('float')
        self.w += np.random.randn(*self.w.shape) * 0.01

        for i in range(self.N):
            self.w[i,i] = 0 
        self.w[:,:2] = 0
        self.w[:2,-2:] = 0
        self.w[-2:,-2:] = 0
        self.w[-2,-1] = 1
        self.w[-1,-2] = 1
    
        self.w_mask = []
        self.i = 0
    
    def init_4_1_trial(self, stim_type=0):
        self.i = 0
        self.t = np.arange(0, self.T, self.dt)
        self.V = self.V * 0 + self.EL
        self.AP = self.AP * 0
        self.ampa = np.zeros((self.N,self.N))
        self.nmda = np.zeros((self.N,self.N))
        self.gaba = np.zeros((self.N,self.N))
        self.eligibility = np.zeros((self.N,self.N))
        self.postsyn_act = np.zeros((1,self.N))
        self.presyn_act = np.zeros((self.N,self.N))
        self.in_refractory = np.zeros_like(self.postsyn_act)
        self.VV = np.zeros((self.N, len(self.t)))
        self.I_EE = np.zeros((self.N, len(self.t)))
        self.I_II = np.zeros((self.N, len(self.t)))
        self.AMPA = np.zeros((self.N,self.N,len(self.t)))
        self.NMDA = np.zeros((self.N,self.N,len(self.t)))
        self.GABA = np.zeros((self.N,self.N,len(self.t)))
        self.POSTSYN = np.zeros((self.N, len(self.t)))
        self.PRESYN = np.zeros((self.N,self.N,len(self.t)))
        self.ELIGIBILITY = np.zeros((self.N,self.N,len(self.t)))
        self.Ie = np.zeros((self.N, len(t)))
        stim_t = range(int(15/dt), int(40/dt))
        if stim_type==0:
            self.Ie[0, stim_t] = 10
            self.Ie[1, stim_t] = 5
        if stim_type==1:
            self.Ie[1, stim_t] = 10
            self.Ie[0, stim_t] = 5
        
    def step(self, stimulus=0):
        self.w_mask = (self.w != 0).astype('int')  
        preAP = self.AP.dot(np.logical_not(self.AP.T).astype(int)) # turn postsynaptic APs to presynaptic

        self.ampa += (-self.ampa/self.tau_ampa + self.neur_type_mask*preAP*self.w)*self.dt
        self.nmda += (-self.nmda/self.tau_nmda + self.neur_type_mask*preAP*self.w)*self.dt
        self.gaba += (-self.gaba/self.tau_gaba + (1.0-self.neur_type_mask)*preAP*self.w)*self.dt
        self.postsyn_act += (-self.postsyn_act/self.postsyn_tau + self.AP.T)*self.dt
        self.presyn_act += (-self.presyn_act/self.presyn_tau + preAP)*self.dt
        self.eligibility += (self.eligibility_lr * self.w_mask*(self.presyn_act*self.postsyn_act - self.eligibility/self.tau_eligibility)) * self.dt

        self.AP = self.AP*0
        where_reset = np.where(self.V.flatten() >= self.Vspike)[0]
        self.V[:,where_reset] = self.Vr
        self.in_refractory[:,where_reset] = self.refractory_period
        where_refractory = np.where(self.in_refractory.flatten() > 0)[0]

        self.I_E = np.sum(-self.ampa*(self.V - self.V_E) - 0.1*self.nmda*(self.V - self.V_E), 0)
        self.I_I = np.sum(-self.gaba*(self.V - self.V_I), 0)
        
        dV = (-(self.V - self.EL)/self.tau + self.I_E + self.I_I + self.Ie[:,self.i]) * self.dt
        try:
            dV[0, where_refractory] = 0
        except:
            print(where_refractory.shape, where_refractory, dV.shape)
        self.V += dV
        
        where_is_AP = np.where(self.V.flatten() > self.Vth)[0]
        self.V[:,where_is_AP] = self.Vspike
        self.AP[where_is_AP,:] = 1

        self.I_EE[:,self.i] = self.I_E
        self.I_II[:,self.i] = self.I_I
        self.VV[:,self.i] = self.V
        self.AMPA[:,:,self.i] = self.ampa
        self.NMDA[:,:,self.i] = self.nmda
        self.GABA[:,:,self.i] = self.gaba
        self.POSTSYN[:,self.i] = self.postsyn_act.flatten()
        self.PRESYN[:,:,self.i] = self.presyn_act
        self.ELIGIBILITY[:,:,self.i] = self.eligibility
        self.i += 1
        self.in_refractory -= self.dt
        
    def learn(self, alpha=None, U=None):
#         e = self.eligibility
#         e_mean = np.mean(e, 0, keepdims=True) + 0.0000000000000001
#         e_prime = 5 * e_mean * np.tanh(e/5*e_mean)
#         e_prime_mean = np.mean(e_prime, 0, keepdims=True) + 0.0000000000000001
#         dw = U*(e_prime - e_prime_mean)
#         self.w += alpha*dw
#         if np.any(self.w < 0):
#             print('Negative weight DETECTED')
#             self.w = self.w * np.logical_not(self.w<0).astype('int')
        
        dw = alpha * U * np.tanh(self.eligibility)
#         print('Summed update: {}'.format(np.abs(dw).sum()))
        self.w += dw + np.random.randn(*dw.shape)*0.01
        if np.any(self.w < 0):
            print('Negative weight DETECTED')
            self.w = self.w * np.logical_not(self.w<0).astype('int')