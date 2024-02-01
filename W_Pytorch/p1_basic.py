import torch

# Initializing with zeros
zero = torch.zeros([5,7])
print(zero)

x = torch.Tensor([[4, 4, 8, 0,0]])
y = torch.Tensor([8,2, 0,0,0])
print(x*y)

# Getting Shape 
s = x.shape
print(s)

# Reshaping
n = torch.Tensor([[1,2,3,4,0],
                  [6,7,8,9,0]])
n = n.view([1, 10])
print("n", n)

zz = torch.zeros([2,6])
zz = zz.view([1, 12])
print(zz)

