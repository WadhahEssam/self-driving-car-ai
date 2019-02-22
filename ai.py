# allow you to play and interact with arrays 
import numpy as np 
# just for using a random numbers
import random 
# this one is because we are going to save the model and
# load it so there will be some interaction with the 
# operating system.
import os 
# the nueral network library that we are going to use
import torch
# importing the most important module from the torch
# library that deals with the neural networks, and this
# will take the three signals of the three sensors + the 
# oreintation and will return the action that the car 
# should take.
import torch.nn
# giving a shortcut for the funcational module inside the
# nn module because it contains the neural network functiuons
# that we need to use.
import torch.nn.functional as F
# importing the torch optimizer to perform the stochastic 
# gradient descent.
import torch.optim as optim
# something technical related to torch maybe i will understand
# it later in the course, and it will be used to convert something
# to something else.
import torch.autograd
from torch.autograd import Variable