import torch
from torch import nn
from pytorch_apis import linear_new

M=2
P=3
N=1

#x=torch.rand((M,P), requires_grad=True)
#w=torch.rand((P,N), requires_grad=True)
#b=torch.rand(N, requires_grad=True)

x=torch.tensor([[1.0,1.0,1.0],[1.0,1.0,1.0]], requires_grad=True)
w=torch.tensor([[1.0,1.0],[1.0,1.0], [1.0,1.0]], requires_grad=True)
#b=torch.tensor([[0.01, 0.01],[0.01, 0.01]], requires_grad=True)
b=torch.rand(N, requires_grad=True)

print('x.shape: ',x,  x.shape)
print('w.shape: ',w,  w.shape)
print('b.shape: ',b,  b.shape)

device=torch.device("cuda")

x=x.to(device)
w=w.to(device)
b=b.to(device)

# our method
c = linear_new(x,w,b)

print('c:',c, 'c shape: ', c.shape)

# pytorch method

#d = torch.mm(x,w) + b

#print('d:',d)

e = c.sum()#torch.ones((M,N), requires_grad = True)
#e = d.sum()
print("------------")
#print('c:', c)
print('e:', e)
print("------------")

print("checking backward")

dd, dx, dw, db = torch.autograd.grad(e, [c,x, w, b])
#ddx, ddw, ddb = torch.autograd.grad(e, [x, w, b])

print('dd:' , dd)
print('dx:', dx)
#print('ddx:', ddx)
print('dw:', dw)
#print('ddw:', ddw)
print('db:', db)
#print('ddb:', ddb)