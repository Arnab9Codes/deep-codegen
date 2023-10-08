import torch
from gp_apis import gp_mat_dot
from pytorch_apis import mat_dot

a = torch.tensor([[2.3400, 4.0000, 3.9000]])
c= torch.tensor([[1.0],[1.0]])
b = torch.tensor([[1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0]])

M = c.shape[0]
P = c.shape[1]
N = a.shape [1]

device =torch.device("cuda")

a=a.to(device)
b=b.to(device)
c=c.to(device)

d = mat_dot(c,a, M,N, M,P,N, device)
print(d)