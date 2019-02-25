# 1) example of zip function
# a = ("John", "Charles", "Mike")
# b = ("Jenny", "Christy", "Monica")
# print('before :')
# print(tuple(a))
# print(tuple(b))
# x = zip(a, b)
# print('after :')
# print(tuple(x))

# 2) example of map with lambda
# a = [1, 2, 3]
# mappedA = list(map(lambda x: x+1, a))
# print(mappedA)

# 3) example of the cat fucntion of the torch libraray
# import torch
# old_state = torch.Tensor(( 11, 22, 33, 44, 55)).float()
# new_state = torch.Tensor((121,232,343,353,363)).float()
# actions =   torch.Tensor(( 90, 80, 70, 60, 50)).float()
# rewards =   torch.Tensor((666,777,888,999,159)).float()
# print(torch.cat((old_state, new_state, actions, rewards), 0))

# 4) the star operator 
# def functionA(a, b):
#   print(a)
#   print(b)
#
# list=[[1, 2, 3, 4], ['a', 'b', 'c', 'd']]
# functionA(*list) 

# 5) sequeeze exmple 
# import torch 
# x = torch.Tensor(5)
# x.unsqueeze(0)
# x.unsqueeze(0)
# print(x)