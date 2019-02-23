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
# the Variable function is a funcation that will convert a tensor 
# to a pytorch variable that contains the tensor and the gradient
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

  # this functin will be used to push a new state to the memory 
  # object variable and will make sure that this memory variable  is
  # containing 100 events at maximum ( depending on the capacity variable )
  #
  # the event that we are pushing to the memeory is containing four elements
  # (last state, new state, last action, last reward obtained.)
  # and still I don't know why exaclty we need to save this specific values
  def push(self, event):
    self.memory.append(event);
    if len(self.memory) > self.capacity :
      # the del function in paython is used to delete an object
      # and since almost everything in python is a variable, we can
      # use the del keyword to delete an array element.
      del self.memory[0]
  
  # this function will  select a random sample from the memory and 
  # will return this random sample
  # 
  # the batchSize is going to be the number of events that we are going
  # to take from the memroy and return it
  #
  # remember that this function will return a one one dimentional list
  # that contains batches that are aligned in a way that pytorch can 
  # understand which values are related to the same time t.
  def sample(self, batchSize):
    # random.sample : it takes a random batch from a list and this
    # random batch is of size batchSize
    # 
    # zip (*) : it reshapes the array, collectes the first element of 
    # every array and put it in one array, and the same for other arrays
    # ex : array = [[1,2],[a,b]] , zip(*array) = [[1,a], [2,b]]
    # and we do this so we can wrap this array into a pytorch variable
    # which contains both a tensor and a gradient, and a tensor is a
    # multidimensional matrix that contains elements of the same data type
    # and that is why we do zip the array we have so we can create multiable
    # arrayes that contain the same data type..
    #
    # after zipping the batching we are going to put every data set into 
    # one pytorch variable, which each one will get a gradient so we  
    # can eventually diferentiate between each of them
    #
    # if you want to read more about tensors you can check this 
    # in the pytorch documentation https://pytorch.org/docs/master/tensors.html
    # and in the test.py file I made an example of how it works
    #
    # the star operator in python does a similar jop to the ( ... ) in 
    # java, you send multidimenstional array, and the function will 
    # take every element inside it and will assign it to one value
    # of the arguments. see example 4 in test.py 
    samples = zip(*random.sample(self.memory, batchSize))
    # torch.cat is using to cancat arrays together and alignthem 
    # according to a specified element, in our case the element
    # is 0, so it will align them according to the first element
    # visit the link and search for cat to learn more.
    # https://pytorch.org/docs/stable/torch.html
    #
    # the Variable function returns a torch variarble that contains
    # both tensors and gradient, 
    # x : is one of the samples which is an array containing
    # elements witht the same data type
    #
    # the reason to concat is for us later to deferentiat which
    # value this relates to, when we apply the stochastic graident
    # descent to update the weigths
    return map(lambda x: Variable(torch.cat(x, 0), samples))

    
  
  