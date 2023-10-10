import gp_apis
from gp_apis import gp_mat_dot, gp_mat_transpose
import torch
import torch.nn as nn

class LinearNewfunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, b, device):

        #print('x.shape:', x.shape, type(x))
        #print('w.shape:', w.shape, w)

        ctx.save_for_backward(x,w,b)

        device = torch.device("cuda") 
        xw = gp_mat_dot(x, w, x.shape[0], w.shape[1] , device)

        #b = torch.broadcast_to(b, (xw.shape[0], xw.shape[1])) # change this 

        y = xw + b # gp_apis.gp_mat_add(xw, b, M, N, M, N, device)
        #y =  mat_add(xw, b, M, N, M, N, device)
        #print("y", y)

        return y

    @staticmethod
    def backward(ctx, dL_dY):
        x, w ,b  = ctx.saved_tensors
        #print(dL_dY.shape)
        device = torch.device("cuda") 

        wt = gp_mat_transpose(w, w.shape[1],w.shape[0], device)
   
        d_x = gp_mat_dot(dL_dY, wt, dL_dY.shape[0], wt.shape[1], device)
        
        xt = gp_mat_transpose(x, x.shape[1], x.shape[0], device)

        #print('-----------------dX--------------------')
        #print('dx tensor:', d_x)
        d_w = gp_apis.gp_mat_dot(xt, dL_dY,  xt.shape[0], dL_dY.shape[1], device)
        '''
        print('wt',wt, 'wt shape:', wt.shape)
        print('x tensor:',x,' x shape:', x.shape)
        print('xt tensor:',xt,' x shape:', xt.shape)
        print('dL_dy:',dL_dY, 'shape: ', dL_dY.shape)
        print('xt tensor:',xt,' x shape:', xt.shape[0],' ', xt.shape[1])
        print('dx', d_x)
        print('dw', d_w)
        '''
        db = dL_dY.sum(0)

        #print('---------------')
        return d_x, d_w, db, None


#def linear_new(x,w,b):
#    return LinearNew.apply(x,w,b)

class LinearNew(nn.Module):
    def __init__(self, features, hidden, device):
        super(LinearNew, self).__init__()
        self.w = nn.Parameter(torch.rand(features, hidden))
        self.b = nn.Parameter(torch.rand(hidden))
        self.device = device

        # Initialize weights and biases
        #nn.init.xavier_uniform_(self.weight)
        #nn.init.uniform_(self.bias)


    def forward(self, x):
        return LinearNewfunc.apply(x, self.w, self.b, self.device)