# BATCH ACTOR-CRITIC (A2C) (NOT ONLINE)

# conda install swig # needed to build Box2D in the pip install
# pip install box2d-py # a repackaged version of pybox2d


# lessons learned
#    - rewards must be discounted 
#    - rewards must be normalized
#    - entorpy loss should be added
#    - there should be a maximum number of steps, otherwise you'll get strange results

# STATE VALUES:
# if the actual rewards at time steps in the episode were [0, 0, 0, 1], then
# the state values (sometimes called, discouted rewards, yeah, terminology is
# a mess) are [0.64, 0.81, 0.9, 1 ] if gamma (discounting factor) is 0.9

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import matplotlib.pyplot as plt

############## Hyperparameters ##############
render = True
_ppo = True
from_scratch = False
# env_name = "CartPole-v1"
env_name = "LunarLander-v2"

# creating environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
solved_reward = 299         # stop training if avg_reward > solved_reward
log_interval = 20           # print avg reward in the interval
max_episodes = 50000        # max training episodes
max_timesteps = 300         # max timesteps in one episode
n_latent_var = 64           # number of variables in hidden layer
update_timestep = 3000      # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = None

#############################################


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        """ RETURNS ACTION PROBABILITIES, LOGS STATES, ACTIONS PROBS AND ACTION LOGPROBS """
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)                 # a vector of 4 values for LunarLander
        dist = Categorical(action_probs)                        
        action = dist.sample()                                  # sample an action from the distribution
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))           # ERROR HERE? No, see below
        
        return action.item()
    
    def evaluate(self, state, action):
        """ here the critic evaluates the VALUE of the current state. Value being discouted rewards. 
        How rewards are discounted, see RL_debug. """
        action_probs = self.action_layer(state) # state is a tensor of shape (steps, state_space)
        dist = Categorical(action_probs)

        # action_probs are probability distributions from which you sampled actions
        # and executed them. Now you take those executed actions and ask what their
        # log-probability was. You can check that the probabilities of the TAKEN
        # actions were exactly as predicted by the actor (check action_logprobs.exp() == [np.round(action_probs[i, action[i]].item(), 4) for i in range(len(action))])
        action_logprobs = dist.log_prob(action) # logprob of the action chosen by the actor
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    # def save_progress():
    #     plt.figure(figsize=(7,7))
    #     plt.plot(state_values.detach().cpu().numpy())
    #     plt.plot(rewards.cpu())

    #     # ratios and clamped ratios:
    #     plt.plot(torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip).cpu().detach().numpy())
    #     plt.plot(ratios.cpu().detach().numpy())

    def update(self, memory, _ppo):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            
            
            if _ppo:
                # Finding the ratio between the old and new probabilities:
                ratios = torch.exp(logprobs - old_logprobs.detach())
                
                # Advantages = actual state values - predicted state values:
                # we want the critic to predict actual state values as accurately as possible
                advantages = rewards - state_values.detach() # rewards - actual state values

                # surrogate loss.
                # we are going to maximize the ratios * adavantages. Why not logprobs?
                # because by maximizing ratios * advantages, we maximize only logprobs, because
                # old_logprobs and state_values are detached, rewards don't depend on weights
                # so no gradients will be calculated for these variables, and no update will happen for them.

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # we try to maximize log-probability of the actions associated with the biggest agvantage,
                # minimize the difference between predicted and actual state values
                # and keep the distribution entropy high (make it less pointy not to be too confident with action choices)

                # below rewards have already been DISCOUNTED
                # PPO selects the minimum value (torch.min(surr1, surr2)) to do gradient ascent on.
                # suppose advantages are 1 (for simplicity) then the new policy is good and we should make it more 
                # likely by increasing its log-likelihood (moving its negative log-likelihood towards zer).
                # But if the new policy is outside of the trust region (i.e.
                # the ratio between the old and the new policy is outside [0.8, 1.2]), the gradient of the 
                # objective will be set to zero to prevent further increasing (or decreasing) the objective
                # (i.e. the log-likelihood of the decisions associated with the positive advantage))
                loss = -torch.min(surr1, surr2) + 0.5*(state_values - rewards).pow(2).mean() - 0.01*dist_entropy 
            else:
                # this implementation is strange. See p.17 http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf
                sv = torch.zeros_like(state_values)
                sv[:-1] = state_values.detach()[1:]
                sv[-1] = self.policy.evaluate(old_states[-1], old_actions[-1])[1]
                advantages = rewards + self.gamma*state_values.detach() - sv
                loss = -logprobs * advantages + advantages.pow(2).mean() - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        


if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
if not from_scratch:
    ppo.policy.load_state_dict(torch.load("BATCH_{}.pth".format(env_name), map_location=torch.device(device)))
    ppo.policy_old.load_state_dict(torch.load("BATCH_{}.pth".format(env_name), map_location=torch.device(device)))

print(lr,betas)

# logging variables
running_reward = 0
avg_length = 0
timestep = 0

# training loop
MAX_REW = 0
for i_episode in range(1, max_episodes+1):
    state = env.reset()                     # state is a vector of eight values
    for t in range(max_timesteps):
        timestep += 1
        
        # Running policy_old:
        action = ppo.policy_old.act(state, memory)
        state, reward, done, _ = env.step(action)
        
        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        
        # update if its time
        if timestep % update_timestep == 0:
            ppo.update(memory, _ppo)
            memory.clear_memory()
            timestep = 0
        
        running_reward += reward
        if render:
            env.render()
        if done:
            break
            
    avg_length += t
    
    # stop training if avg_reward > solved_reward
    if running_reward > (log_interval*solved_reward):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), './BATCH_{}.pth'.format(env_name), _use_new_zipfile_serialization=False)
        break
        
    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length/log_interval)
        running_reward = int((running_reward/log_interval))
        
        if running_reward > MAX_REW:
            MAX_REW = running_reward
            try:
                # with newer versions of torch
                torch.save(ppo.policy.state_dict(), "BATCH_{}.pth".format(env_name), _use_new_zipfile_serialization=False)
            except:
                # with older versions of torch
                torch.save(ppo.policy.state_dict(), "BATCH_{}.pth".format(env_name))
            saving = True
        else:
            saving = False
        print('Episode {} \t avg length: {} \t reward: {}\t saving {}'.format(i_episode, avg_length, running_reward, saving))
        avg_length = 0
        running_reward = 0
        