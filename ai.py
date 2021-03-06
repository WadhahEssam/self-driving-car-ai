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
  def sample(self, batch_size):
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
    samples = zip(*random.sample(self.memory, batch_size))
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

# The dee q learning class which is the main class that we are 
# going to be implementing and the we will define all the above 
# classes inside this class
class Dqn():
  # input_size and nb_actions weill be passed to consturct the
  # network object, and the gamma parameter is used in the bellman
  def __init__(self, input_size, nb_actions, gamma):
    self.gamma = gamma
    # this reward_window is going to keep track of the last 100
    # rewards he got throw the past time 
    self.reward_window = []
    # creting our neural network fot the deep q learning
    self.model = Network(input_size, nb_actions)
    # an object of the replay memory class, and we can pass the 
    # capacity from the constructor parameters but we did this way
    self.memory = ReplayMemory(100)
    # creating the optimizer, inside the optim object there are many
    # optimizers and they are all good and each one is used for a 
    # specific porpuse, and we are going to be using the adam 
    # algorithm optimizer, maybe later I can try the rmsprop,
    #
    # then in order to connect the adam parameter to our neural
    # network we pass the model (our neural network ) parameters
    # to the Adam algorithm 
    #
    # lr : is the learning rate, always try not to make it a large 
    # number because the agent will not learn properly
    #
    # remember that the optimizer is now an object of the adam class
    #
    # note that the model.parameters is only available on the network
    # class because it inherents the nn.Module class, and this function
    # returns an iterator just like what the Adam optimizer is accepting
    # 
    # the job of the optimizer is to take the loss that we will calculate
    # and then it will update all the weights of the neural network
    self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
    # this attribute will represent our last state, remember that the last
    # state is a vector of 5 values/dimensions ( the three signals, and 
    # orientation ), and since we are working with pytorch so this last
    # state needs to be a torch tensor ( a tensor is a multi-dimenstioal array
    # of values that has the same type ), so our last state is going to be
    # taking place as one array/dimension inside this multidimenstional array
    # and we also need to have an extra array/dimenstion that is reperesnting 
    # the batch and this is required by pytorch and also other libraries require
    # it too,
    # 
    # and it is doesn't only has to be 
    # torch tensor, but it has to have an extera dimension which is going 
    # to be a Batch, because the neural networks can only accept batches
    # even in other libraries so we also have to create this fake extra 
    # dimension that corrosponds to the batch
    #
    # the unsqueeze function adds an a dimension for the Batch to be inserted
    # at the first position of the tensor 
    #
    # I added more explaination in the text document of this course
    self.last_state = torch.Tensor(input_size).unsqueeze(0)
    # actions will be inside the range [ -1, 0, 1 ] and then those actions
    # will be converted to be corrosponging to the oreintation of the car
    # which are in the range of [ -20, 0, 20 ], and we initialize it to 0
    self.last_action = 0
    # the reward is a floating number that is again between -1 and +1
    self.last_reward = 0

  # this fucntion is the function that will make the decistion of the car
  # after each step
  # so remember that the selected action will be one of the three Q values
  # that we get from the output of the neural network which will recieve 
  # a state in order to pass it to the hidden layer and then to the output
  # layer, and finally we will feed the q values to a softmax function that
  # will choose one of the available q values.
  #
  # state : reperesents the current state of the env so we can pass it to the 
  # self.model variable, and this state should be converted as a tensor 
  # variable that has a batch cuz as we said that this is how pytorch acceps
  # the state variables 
  def select_action(self, state):
    # applying the softmax function to the q values we get from the neural 
    # network, remember that softmax selectes the final value depending on
    # the propabilities so if the qValues are [0.1 , 0.7, 0.2] , there will
    # be a 70% chance that the output of the softmax function is going to be
    # the second action, there are other functions that takes the hieghest 
    # values but this means that you don't want your agent to keep exploring 
    # the environment 
    #
    # we pass the state tensor to the sotfmax afterwrapping it with a torch
    # Variable class
    # 
    # we pass it a volatile argument with a value of true to tell the nn 
    # module that we don't want the gradient on the graph of all the computations
    # of the nn module, which means that you will not be including the gradient
    # of the state into the graph, and this will save you some memory
    # 
    # Tempreture parameter : this parameter is used to heigher the chanse of
    # getting the most sure action for example multiplay this [1,2,3] by 7
    # and see the result propability and you will get it, in this project 
    # the tempreture parameter will help you to smooth the movement of the 
    # car which will return in a more realistic movement rather that the 
    # insect like movement that we have
    #
    # where is the forward function, the forward function is not directly
    # accessed otherwise it is called by the nn module when you call it with
    # the actual states that you have > more discussion on that here
    # https://www.udemy.com/artificial-intelligence-az/learn/v4/questions/3337464
    props = F.softmax(self.model(Variable(state, volatile = True))*7)
    # the multinomial function is related to numby and it gives you a random
    # draw from the distribution that is returned by the softmax fucntion
    action = props.multinomial()
    # this will come with a batch so in order to take the action out of the
    # btach, we will need to write the following line, and it will return 
    # the action that we want to take which is 0, 1, or 2
    return action.data[0,0]

  # training the neural network process :
  # forward propagation > get output > get target > compar output with target 
  # to compute the last error > backpropagate the last error to the network, 
  # and using schotastic gradient descent to update the wights of the network
  # according to how much they contributed to the last error
  #
  # batchstate is going to be the output of the sample function that is in the
  # Replay memory state.
  #
  # batch_state : is the variable that we got from the sample function in the 
  # replay memory class which represents 100 random state samples wraped in a
  # batch.
  #
  # the batch_next_state : is the actual state that you became in after doing 
  # the action, because this function is actually going to be played after the 
  # action has been played.
  def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
    # this will return the output of the possiable actions 0, 1, 2 but we are
    # only interested on the actions that were actually taken, and that's why 
    # we chained the gather function along with one and the batch action.
    # but notice that we added a fake extra dimenstion to the batch_state in 
    # the replay memory class, so we need to add this dimention to the batch 
    # action also, and we put one and not zero beacuse zero corrosponds to the
    # the state and one corrosponds to the actions,
    # also you don't need the output to be in the form of a batch so you will 
    # need to squeeze it and return 
    outputs = self.model(batch_state).gather(1, batch_action.unsqueez(1)).squeeze(1) 
    # i don't know why is this called the outputs because this is only one output
    # which is the real output that we will compare with the previous ouput
    next_outputs = self.model(batch_next_state).detach().max(1)[0]
    # this is exaclty like what the equation is presenting
    target = self.gamma*next_outputs + batch_reward
    # td reperesents the temporal deference loss and I talked about that in
    # the text document, and in order to calculate this we will use a function
    # and there are others but the instructor for somereason that I don't know 
    # he has chosen this one to advised that we should use it too.
    td_loss = F.smooth_l1_loss(outputs, target)
    # back propagating the loss to the network in order to get update the wights 
    # and this is all done by the optimizer that we created and attached to the 
    # Dqn class which is the Adam optimaizer 
    #
    # you have to reinitialize the optimizer on every iteration of the loop 
    # which is the job of the zero_grad() function
    self.optimizer.zero_grad()
    # the job of the function retain variables is to free some memory and this 
    # will improve the backward propagation cuz you are going to go several times
    # on the loss 
    td_loss.backword(retain_variables = True)
    # here is the line that update the weights of the neural netwok
    self.optimizer.step()

  # the update function updates everything that needs to be updated as long as the 
  # AI has reached a new state.
  # things we need to update : last action > action that was just played, also the
  # last state becomes the state that was just played, and the last reward variable
  # also becomes the reward that we get.
  # it is important to know that this function is your interface to the outside world
  # because it takes the last reward and the last signal and it provides you with 
  # an action, so the output of this function is an action.
  def update(self, reward, new_signal):
    # the signal is the state and we are going to convert the signal that includes 
    # five values into a torch tensor so it can be compatiable with torch library
    # and again we are going to make the fake dimention for the batch
    new_state = torch.Tensor(new_signal).float().unsqueez(0)
    # now we need to append this trnsition to the memory to improve the experince 
    # replay, so we pass the compo that we make to the experince replay class
    # Notice : that this is a multi a tow prackets inside each other
    # all the values that we save inside the memory are a torch Tensors so we need 
    # to change the last reward and the last action into a torch tensor cuz they 
    # are the only ones that are not updated
    # LongTensor : is a type of tensor that contains integer numbers and that is 
    # what we want for our action because it is going to be either 0, 1, 2
    # we put the prackets [] because in order to change a variable into a tenosr
    # this variable should be a list, and states as we know they are already a 
    # variables inside a list.
    self.memory.push((self.last_state, new_state, self.last_action, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))


    

    
  
  