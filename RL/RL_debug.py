import gym
import numpy as np
import matplotlib.pyplot as plt

##################################################################
# illustration of how Gym works
##################################################################

# env_name = "LunarLander-v2"

# env = gym.make(env_name)

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# env.seed(0)

# done = False
# state = env.reset() 
# while not done:
#     action = np.random.choice(4)
#     state, reward, done, _ = env.step(0)
#     env.render()


##################################################################
# illustration of how rewards are discounted (the Bellman equation)
##################################################################
# class Memory:
#     def __init__(self):
#         self.rewards = [0,0,0,0,10] # rewards in chronological order
#         self.is_terminals = [False,False,False,False,True]
#         self.gamma = 0.9

# memory = Memory()
# rewards = []
# discounted_reward = 0

# for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
#     if is_terminal:
#         discounted_reward = 0
#     discounted_reward = reward + (memory.gamma * discounted_reward)
#     rewards.insert(0, discounted_reward)

# print("rewards in chronological order: ", memory.rewards)
# print("discouted rewards in chronological order: ", rewards)

##################################################################
# illustration of PPO
##################################################################

import torch
from torch.autograd import Variable

G, P, R, OP, L = [], [], [], [], []

A = 1
for p in np.linspace(0.01, 0.99, 100):
    probs = torch.tensor(p)
    old_probs = torch.tensor(0.7)

    logprobs = Variable(torch.log(probs), requires_grad=True)
    old_logprobs = torch.log(old_probs)

    ratios = torch.exp(logprobs - old_logprobs) # new probs represent what fraction of the old ones
    surr2 = torch.clamp(ratios, 0.8, 1.2)

    objective = torch.min(ratios, surr2) * A   # the we want to maximize if A is positive, or minimize if A is negative

    print(f'{ratios}, {surr2} chosen: {objective.item()}')

    objective.backward()
    print('Grad:\t\t', logprobs.grad)
    print('new probs:\t', logprobs + 0.1 * logprobs.grad)

   
    G.append(logprobs.grad.item())
    P.append(probs.item())
    OP.append(old_probs.item())
    R.append(ratios.item())
    L.append(objective.item())
    logprobs.grad.data.zero_()

plt.figure(figsize=(15,5))
plt.suptitle (f"Advantage is {'positive' if A > 0 else 'negative'}")
plt.subplot(1,2,1)
plt.plot(P, G, label='gradient', linewidth=5, alpha=0.4)
plt.plot(P, OP, label='old_prob')
plt.plot(P, R, label='prob/old_prob')
plt.xlabel('prob')
plt.legend()
plt.grid('on')
plt.subplot(1,2,2)
plt.plot(R, L)
plt.xlabel('ratio')
plt.ylabel('objective')

plt.show()

G, P, R, OP, L = [], [], [], [], []

A = -1
for p in np.linspace(0.01, 0.99, 100):
    probs = torch.tensor(p)
    old_probs = torch.tensor(0.7)

    logprobs = Variable(torch.log(probs), requires_grad=True)
    old_logprobs = torch.log(old_probs)

    ratios = torch.exp(logprobs - old_logprobs) # new probs represent what fraction of the old ones
    surr2 = torch.clamp(ratios, 0.8, 1.2)

    objective = torch.min(ratios, surr2) * A   # the we want to maximize if A is positive, or minimize if A is negative

    print(f'{ratios}, {surr2} chosen: {objective.item()}')

    objective.backward()
    print('Grad:\t\t', logprobs.grad)
    print('new probs:\t', logprobs + 0.1 * logprobs.grad)

   
    G.append(logprobs.grad.item())
    P.append(probs.item())
    OP.append(old_probs.item())
    R.append(ratios.item())
    L.append(objective.item())
    logprobs.grad.data.zero_()

plt.figure(figsize=(15,5))
plt.suptitle (f"Advantage is {'positive' if A > 0 else 'negative'}")
plt.subplot(1,2,1)
plt.plot(P, G, label='gradient', linewidth=5, alpha=0.4)
plt.plot(P, OP, label='old_prob')
plt.plot(P, R, label='prob/old_prob')
plt.xlabel('prob')
plt.legend()
plt.grid('on')
plt.subplot(1,2,2)
plt.plot(R, L)
plt.xlabel('ratio')
plt.ylabel('objective')

plt.show()