import torch
from torch import nn
from pytorch_apis import linear_new

M=3
P=4
N=2

x=torch.rand((M,P))
w=torch.rand((P,N))
b=torch.rand(N)

print('x.shape: ', x.shape)
print('w.shape: ', w.shape)
print('b.shape: ', b.shape)

device=torch.device('cuda')

x=x.to(device)
w=w.to(device)
b=b.to(device)

# our method
c = linear_new(x,w,b)

print(c)

# pytorch method

d = torch.mm(x,w) + b

print(d)