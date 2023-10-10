import torch
#from pytorch_apis import mat_dot, mat_transpose
from gp_apis import gp_mat_dot, gp_mat_transpose

M =2
P=2#0000
N =3

#a = torch.rand((M,P), dtype=torch.float32)
#b = torch.rand((P,N), dtype=torch.float32)
#o = torch.rand(dim_0)

a= torch.tensor([[1., 1.],
        [1., 1.],
        [5.0,4.0]],)

b =torch.tensor([[2.3400, 4.0000, 3.9000, 100],
        [5.0000, 4.0000, 3.8000, -999.9],],)
device = torch.device("cuda")

a = a.to(device)
b = b.to(device)
#o = o.to(device)

c = gp_mat_dot(a, b,a.shape[0] , b.shape[1], device)
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
#print('addition check')
#a= torch.rand((M,N))
#b=torch.rand((M,N))

#a=a.to(device)
#b=b.to(device)

#c= mat_add(a,b,M,N, M, N, device)
#d= a+b
#print('c:\n',c)
#print('d:\n',d)

#print('addition check: ', torch.equal(c,d))

print('--------------')
print('transpose check')
t=gp_mat_transpose(a, a.shape[1], a.shape[0], device)
tr=a.T
print('t')
print(t)
print('tr')
print(tr)
print(torch.equal(t, tr))
print('---------------')
gpmat=gp_mat_dot(t,c, t.shape[0], c.shape[1], device)
print('gpmat', gpmat, 'shape:', gpmat.shape)

tmat =torch.mm(t,c)
print('tmat', tmat, 'shape:', tmat.shape)
print(torch.equal(gpmat,tmat))