import torch
from pytorch_apis import mat_dot, mat_transpose, mat_add, mat_mul

M =2
P=2#0000
N =3

#a = torch.rand((M,P), dtype=torch.float32)
#b = torch.rand((P,N), dtype=torch.float32)
#o = torch.rand(dim_0)

a= torch.tensor([[1., 1.],
        [1., 1.],], requires_grad=True)

b =torch.tensor([[2.3400, 4.0000, 3.9000],
        [5.0000, 4.0000, 3.8000],], requires_grad=True)
device = torch.device("cuda")

a = a.to(device)
b = b.to(device)
#o = o.to(device)

c = mat_dot(a,b,M,N,M,P,N, device)
print('a\n',a)
print('b\n',b)
print('c\n',c)
print('c shape:', c.shape)

d = torch.mm(a,b)
print('d\n',d)
print('d shape: ', d.shape)
print(torch.equal(c,d))
#c = addition(a, b, 49 ,dim_0, device)
#if torch.allclose(c, a + b): print("Computation on GPU is correct")
#else: print("Computation on GPU is wrong")
print('----------------')
print('addition check')
a= torch.rand((M,N))
b=torch.rand((M,N))

a=a.to(device)
b=b.to(device)

c= mat_add(a,b,M,N, M, N, device)
d= a+b
print('c:\n',c)
print('d:\n',d)

print('addition check: ', torch.equal(c,d))

print('--------------')
print('transpose check')
t=mat_transpose(c, N, M, M, N, device)
tr=c.T
print(torch.equal(t, tr))