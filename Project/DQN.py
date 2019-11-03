# DQN `
# will need epsilon greedy policy
# target network 
# experience replay
# We will use CNN to train the agent to learn wich actions are the best

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np

# we will save fixed number of experiences for training purposes
# Each experience will have the current state, action, reward and the next state in it
# Each experience will be a list
class Experience_Replay:
    def __init__(self, len_memory = 10000):
        self.len_memory = len_memory
        self.experience = []
        self.batch_size = 5
        
    # this function adds an experience to the Replay buffer
    def add(self, in_state, action, reward, nx_state, done, image, nx_image):
        # if the length of the buffer is not max, we will just attach an experience to the end
        if len(self.experience) < self.len_memory:
            self.experience.append([in_state, action, reward, nx_state, done, image, nx_image])
        # if the length of the buffer is the max it can take, we will add an experience and remove the first experience
        elif len(self.experience) == self.len_memory:
            # remove the first experience
            self.experience.pop(0)
            # add the new one
            self.experience.append([in_state, action, reward, nx_state, done, image, nx_image])
        return self.experience
    # we can pick a batch of random experiences from the replay buffer to train
    def pick_experience(self):
        # we pick a batch of random experiences from the replay buffer to train
        experience_batch = random.sample(self.experience, self.batch_size)
        return experience_batch

        
class eps_greedy:       
    def update_epsilon(self,grid_size, eps,start_e, end_e, state, decay_rate, decay_step, dqn_train_model):
        random_num = np.random.rand()
        if random_num < eps:
            # actions can be from range of [0,1,2,3]
            action = random.choice([0,1,2,3])
        else: 
            # state is the image as a tensor
            state = state.view([1,1,grid_size,grid_size])
            action_tensor = dqn_train_model.forward(state) # this gives 4 Q values corresponding to the 4 actions
            # we pick the action that gives the maximum
            
            # peak action that gives the max Q value
            # this will use DQN
            action = int(torch.argmax(action_tensor))
            
        eps = end_e + (start_e - end_e) * np.exp(-decay_rate * decay_step)
        
        if eps < end_e:
            eps = end_e
            
        return action, eps
        
        
     
# this is the model of neural network
# we use CNN 
class dqn(nn.Module):
    def __init__(self):
        super(dqn, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,stride=1, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,stride=1,padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,stride=1, padding = 1)
        self.linear = nn.Linear(64*4*4, 4) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.linear(x.view(x.size(0), -1))
        # return a tensor of 4 by 1, each is the Q value of actions 


class dueling_dqn(nn.Module):
    def __init__(self):
        super(dueling_dqn, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,stride=1, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,stride=1,padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,stride=1, padding = 1)
        self.linear1 = nn.Linear(64*4*4, 4) 
        self.linear2 = nn.Linear(64*4*4,1)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # value
        value_state = self.linear2(x)
        # advantage
        advantage = self.linear1(x)
        
        # put the value and the advantage together and return
        return value_state+advantage-advantage.mean()