# https://github.com/hermesdt/reinforcement-learning/blob/master/ppo/cartpole_ppo_online.ipynb
# PPO doesn't work well with this online version of A2C
# without PPO, A2C is much less stable than batch A2C with PPO.
import numpy as np
import time, pickle
import torch
import gym
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
# from torch.utils import tensorboard
# w = tensorboard.SummaryWriter()

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): 
        super().__init__()
    def forward(self, input): 
        return mish(input)

# ************
render = False # False for traning, true for Testing
env_name = "CartPole-v1"
# env_name = "LunarLander-v2" #
max_timesteps = 500         # max timesteps in one episode
gamma = 0.98
eps = 0.2

max_grad_norm = 0.5
lra = 1e-3
lrc = 1e-3

EPISODES = 5000
stat_interval = 10
act_actor =  Mish() #nn.Tanh() # Mish()
act_critic = nn.ReLU() # Mish
Ppo = True
# ************

s = 0
episode_rewards = []

# helper function to convert numpy arrays to tensors
def t(x):
    return torch.from_numpy(x).float()

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, n_actions),
            nn.Softmax()
        )
    
    def forward(self, X):
        return self.model(X)

    
    
# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)


env = gym.make(env_name)

# config
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
actor = Actor(state_dim, n_actions, act_actor)
critic = Critic(state_dim, act_critic)
adam_actor = torch.optim.Adam(actor.parameters(), lr=lra)
adam_critic = torch.optim.Adam(critic.parameters(), lr=lrc)

if render:
    actor.load_state_dict(torch.load('./actor_{}.pth'.format('LunarLander'), map_location=torch.device('cpu')))
    critic.load_state_dict(torch.load('./critic_{}.pth'.format('LunarLander'), map_location=torch.device('cpu')))

torch.manual_seed(1)

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def policy_loss(old_log_prob, log_prob, advantage, eps, Ppo=False):
    if Ppo:
        ratio = (log_prob - old_log_prob).exp()
        clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage
        m = torch.min(ratio*advantage, clipped)
    else:
        m = log_prob * advantage
    return -m

running_reward = 0
total_steps = 0
tt = time.time()
for i in range(EPISODES):
    
    prev_prob_act = None
    done = False
    total_reward = 0
    timestep = 0
    state = env.reset()


    for t_ in range(max_timesteps):
        s += 1
        probs = actor(t(state)) # get probability distribution given the state
        dist = torch.distributions.Categorical(probs=probs) # ??????????
        action = dist.sample()  # sample action from this dist
        prob_act = dist.log_prob(action) # get log of the probability distribution
        
        next_state, reward, done, info = env.step(action.detach().data.numpy())
        advantage = reward + (1-done)*gamma*critic(t(next_state)) - critic(t(state))
        
        # if not render:
	       #  w.add_scalar("loss/advantage", advantage, global_step=s)
	       #  w.add_scalar("actions/action_0_prob", dist.probs[0], global_step=s)
	       #  w.add_scalar("actions/action_1_prob", dist.probs[1], global_step=s)
        #     # pass
        # else:
        #     env.render()
        
        total_reward += reward
        if done:
            break
        state = next_state
        
        if prev_prob_act:
            actor_loss = policy_loss(prev_prob_act.detach(), prob_act, advantage.detach(), eps, Ppo)
            adam_actor.zero_grad()
            actor_loss.backward()
            # clip_grad_norm_(adam_actor, max_grad_norm)
            adam_actor.step()

            critic_loss = advantage.pow(2).mean() # the critic produces V(s_t), Q(s_t,a_t) is the objective,
            # i.e. r_{t+1} + gamma * V(s_{t+1}). A(s_t, a_t) = Q(s_t, a-t) - V(s_t) SEE materials in the folder
            # we want V(s_t) to be the same as Q(s_t, a_t).


            # w.add_scalar("loss/critic_loss", critic_loss, global_step=s)
            adam_critic.zero_grad()
            critic_loss.backward()
            # clip_grad_norm_(adam_critic, max_grad_norm)
            adam_critic.step()

        # if not render:
        #     try:
        #         w.add_scalar("loss/actor_loss", actor_loss, global_step=s)
        #         w.add_histogram("gradients/actor",
        #                      torch.cat([p.grad.view(-1) for p in actor.parameters()]), global_step=s)
        #         w.add_histogram("gradients/critic",
        #                      torch.cat([p.data.view(-1) for p in critic.parameters()]), global_step=s)
        #         w.add_scalar("reward/episode_reward", total_reward, global_step=i)
        #     except:
        #         pass
        
        prev_prob_act = prob_act
    
    episode_rewards.append(total_reward)
    running_reward += total_reward
    total_steps += t_

    if i % (stat_interval * 4) == 0:
        with open('rw_online.pickle', 'wb') as f:
            pickle.dump(episode_rewards, f)
        
        print('Episode: {:03d}\t avg_RWD: {:.2f} \t avg_steps: {}\t elapsed {:.2f} s.'.format(i,
                                                                 running_reward/stat_interval,
                                                                 total_steps/stat_interval,
                                                                 time.time()-tt))
        running_reward, total_steps, tt = 0, 0, time.time()

        if not render:
            torch.save(actor.state_dict(), './actor_{}.pth'.format('LunarLander'))
            torch.save(critic.state_dict(), './critic_{}.pth'.format('LunarLander'))
