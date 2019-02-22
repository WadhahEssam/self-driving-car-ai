# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class Network(nn.Module):
    def __init__(self, input_size, nb_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.full_conncection1 = nn.Linear(input_size, 30)
        self.full_conncection2 = nn.Linear(30, nb_actions)
        
# Implementing Experience Replay

# Implementing Deep Q Learning
