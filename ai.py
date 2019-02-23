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
import torch.nn as nn
# giving a shortcut for the funcational module inside the
# nn module because it contains allthe neural network 
# functiuons that we need to use.
import torch.nn.functional as F
# importing the torch optimizer to perform the stochastic 
# gradient descent.
import torch.optim as optim
# something technical related to torch maybe i will understand
# it later in the course, and it will be used to convert something
# to something else.
import torch.autograd
from torch.autograd import Variable

# creating the Network class that is inhereting the nn.Module Class
# and this class is going to hold every thing our neural network needs
class Network(nn.Module):
  # this is how to write the constructor in python ( __init__ )
  #
  # self refers to the object of this class that we are going to be
  # created, it is like this in java except in python you have 
  # to specify and put it in one the constuctors auguments
  #
  # the input size is 5 and it is the signals coming from the 
  # sensors and the plus and minus orentation, and those are 
  # coming from the may.py line 
  def __init__(self, input_size, nb_actions):
    # this is related to features given by the nn module
    # and first you need to specify the Network class,
    # then pass the Network's self object, and then 
    # chain this function with the .__init__()
    # so take this for granted
    super(Network, self).__init__()
    # attache the arguments with the object attributes
    self.input_size = input_size
    self.nb_actions = nb_actions
    # now we start defining the neural network, by defining
    # the all the neural network layers and connecting them
    # together.
    #
    # full connection means that all the nerurons in the first
    # layer is connected to all the neuorons to the second layer.
    #
    # so we will start by defingin the neurons for the input layers
    # and connecting it to the layers of the neurons of the hidden 
    # layer and this connection is going to be linear which means 
    # that all neurons in all layers are conntected.
    #
    # then we pass the function the numbers of neorons in the input
    # layer and the number of neurons in the hidden layer.
    #
    # the numbes of neurons in the hidden layer is totally expermintal 
    # so you can play with it until you get a result that satisfy you
    # the most.
    #
    # then we do the same to define the full connection between the 
    # hidden layer and the output layer 
    self.full_connection1 = nn.Linear(input_size, 30)
    self.full_connection2 = nn.Linear(30, nb_actions)

  # this is the functin that does the forward propagation, and takes 
  # the state ( which is the five input values ) and will return
  # the three output actions or in another word it will return the
  # three Q values that will be used as an output for the our neural 
  # netowrk.
  def forward(self, state):
    # first we need to activate the hidden neurons, by getting the 
    # full connection 1 and applying a rectifier function on them
    #
    # F.relu is the rectifier function and we pass to it a full 
    # connection containing with its left side values and it applies
    # the rectifier fucntion to it and it returns the values of the
    # left side neurons of the layer ( in our case the values of the 
    # hidden layer neurons ).
    # 
    # SideNote : ( state != input_size ) because the first one is the 
    # actual values and the second one is only the number of inputs
    # we are expecting.
    x = F.relu(self.full_connection1(state))
    # now in order to get the values of the output layer we call the
    # second full connection and passing it the values of the hidden 
    # layers neurons values.
    qValues = self.full_connection2(x)
    return qValues
  
# implementing the experience replay in our neural network
# 
# a small reminder about experience replay, if the agent is going
# from a state to another state and the two states are co-related
# he will not learn any thnng, so we need to implement the expereince 
# replay to train the agent to understant long-term co-relations
# 
# how it works ? instead of considering only one state at time t
# , we are going to consider more in the past so our series of events
# will not only be st and st+1 but it will be the last 100 states 
# that happened in the past ex: st-100 st-99 up to st -1 and st, and 
# we put all of those to the memory so we have a logn term memory 
# instead of an instant memory
# 
# once we create this memory of the last 100 events we will take 
# some random batches of these transitions to make our next update
# which is our next move (action)
# 
# the object passed this inicates that this class is inhereting
# the class object just like what java does, bnut in java 
# you don't have to tell it and it does it by default, i searched
# and found out that this way was used in the past and you don't 
# have to use it in python versions > 2.2.
class ReplayMemory(object):
  def __init__(self, capacity):
    # is the maximum number of events ( states ) that
    # we want to have in out memory and we can experement with it 
    # by increasing or decreasing the number when calling this funcion.
    self.capacity = capacity
    # is the list containing the values of the last 100 events ( depending
    # on the capacity that you put ), and it events will be added to it 
    # by the push function.
    self.memory = []

  