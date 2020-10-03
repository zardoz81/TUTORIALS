from skimage import transform as im_tf

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# make sure you don't hog all the video memory
import os
import tensorflow.python.framework.dtypes
from keras.utils import to_categorical

import numpy as np
import scipy
import gym
import pickle

def RGB2gray(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 1/3 * R + 1/3 * G + 1/3 * B

def prepro(o, image_size=[80, 80]):
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = im_tf.resize(y, image_size, mode='constant')
    return resized
#     return np.expand_dims(resized.astype(np.float32), axis=2).ravel()

def preprocess(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class Agent_PG:
    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.S = None
        self.mean = 0.
        self.std = 1.
        self.nda = []
        self.batch_size = 32
        self.__init_game_setting()
        self.brain = self.Net()
        self.lrate = 0.001
        
    def __init_game_setting(self):
        self.observation = self.env.reset()
        
    
    class Net(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.conv1 = nn.Conv2d(1, 16, 8, stride=4)
            self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
            self.fc1 = nn.Linear(2048, 128)
            self.fc2 = nn.Linear(128, 2)
            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

        def forward(self, x):
            x = x.view(-1,1,80,80)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = x.view(-1, 2048)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
            return x

        def update_parameters(self, lrate):
            for f in self.parameters():
                f.data.sub_(f.grad.data*lrate)
        

    def fit(self, S, A, discount_reward):
        action_onehot = torch.tensor(to_categorical(A.reshape(-1), num_classes=2))
        X_pt = torch.tensor(S.reshape(-1,80,80,1)).float()
        pred = self.brain.forward(X_pt)
        
        objective = torch.sum(action_onehot*pred, dim=1) # likelihood of actions taken
        objective = torch.log(objective)                 # log-likelihood of actions take
        # maximize the log-likelihood of actions with the biggest reward,
        # minimize the log-likelihood of actions with the smallest (possibly negative) reward
        objective = -objective * torch.tensor(discount_reward)
        objective = torch.sum(objective)

        self.brain.zero_grad()
        objective.backward(retain_graph=True)
        self.brain.update_parameters(self.lrate)
        
        return objective.detach().numpy()
    
    def run_episode(self,i):  ####### playing one episode
        state = self.observation
        done = False
        episode_reward = 0.0
        S = np.zeros([10000, 80, 80])
        A = np.zeros([10000,])
        R = np.zeros([10000,])
        j = 0
        while not done:
            action = self.make_action(state, test=False)
            state, reward, done, info = self.env.step(action)
            episode_reward += reward
            S[j] = self.S
            A[j] = 0 if action == 2 else 1
            R[j] = reward
            j = j + 1
        self.nda = sum(A)/j

        def compute_discounted_R(R, discount_rate=.99):
            # we take sparse rewards and fill the times when there's no reward
            # (neither positive, nor negative) with discounted rewards, starting
            # from the future and moving backwards into the past
            discounted_r = np.zeros_like(R, dtype=np.float32)
            running_add = 0
            for t in reversed(range(R.shape[0])):
                if R[t] != 0:
                    running_add = 0
                running_add = running_add * discount_rate + R[t]
                discounted_r[t] = running_add
            discounted_r = (discounted_r-discounted_r.mean()) / (discounted_r.std()+0.00001)
            return discounted_r
        RR = R[:j]
        RR = compute_discounted_R(RR)
        return S[:j], A[:j], RR-0.01, episode_reward
#         return S[:j], A[:j], RR, episode_reward

    def train(self, n_episodes):
        reward_history = []
        for i in range(n_episodes):
            self.__init_game_setting()
            S, A, discount_reward, episode_reward = self.run_episode(i)
            loss = self.fit(S, A, discount_reward)

            ########### print and save
            print('Episode: {} \t Reward {} \t Mean action {:0.2f} \t Frames {}'.format(i, episode_reward, np.mean(A), A.shape[0]))
            with open("log_PG_PYTORCH.txt", "a") as myfile:
                myfile.write("episode " + str(i) + "\t" +
                             "loss " + str(loss) + "\t" +
                             " episode reward " + str(episode_reward) + "\t" +
                             " number of down act " + str(self.nda) + "\t" +
                             " game_len " + str(len(discount_reward)) + "\t" +
                             "\n")
            reward_history.append(episode_reward)
            torch.save(self.brain.state_dict(), 'checkpoint.pth')

    def make_action(self, observation, test=True):
        prev_observation = observation
        observation = prepro(observation - self.observation)
        
        pi_action = self.brain.forward(torch.tensor(observation.reshape(1,80,80,1)).float())
        pi_action = np.squeeze(pi_action.detach().numpy(), axis=0)

        if test:
            action = pi_action.argmax()
        else:
            action = np.random.choice(2, p=pi_action)
        self.observation = prev_observation
        self.S = observation
        return 2 if action == 0 else 3

agent = Agent_PG()

print(agent.brain)
n_episodes = 15000
agent.train(n_episodes)