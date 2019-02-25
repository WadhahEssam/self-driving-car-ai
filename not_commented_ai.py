# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim as optim
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class Network(nn.Module):
    def __init__(self, input_size, nb_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.full_conncection1 = nn.Linear(input_size, 30)
        self.full_conncection2 = nn.Linear(30, nb_actions)
    
    def forward(self, state):
        x = F.relu(self.full_conncection1(state))
        qValues = self.full_conncection2(x)
        return qValues
        
# Implementing Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity :
            del self.memory[0]
    
    def sample(self, batch_size):
        sample = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), sample)

# Implementing Deep Q Learning
class Dql():
    def __init__(self, input_size, nb_actions, gamma):
        self.gamma = gamma
        self.model = Network(input_size, nb_actions)
        self.memory = ReplayMemory(100)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

        