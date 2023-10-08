import torch as th
import gp_apis
import torch

class gspmmv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, dim_0, dim_1, reverse, norm, device0):
        res = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0):
    return gspmmv_impl.apply(graph, input1, dim_0, dim_1, reverse, norm, device0)

class gspmmve_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gspmmve(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmve(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
    return gspmmve_impl.apply(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0)

class gspmme_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, edge_input, dim_0, op, reverse, device0):
        res = gp_apis.gp_gspmme(graph, edge_input, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmme(graph, edge_input, dim_0, op, reverse, device0):
    return gspmme_impl.apply(graph, edge_input, dim_0, op, reverse, device0)

class gspmme2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, edge_input, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gspmme2d(graph, edge_input, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmme2d(graph, edge_input, dim_0, dim_1, op, reverse, device0):
    return gspmme2d_impl.apply(graph, edge_input, dim_0, dim_1, op, reverse, device0)

class gspmmve2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
        res = gp_apis.gp_gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
    return gspmmve2d_impl.apply(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0)

class gsddmmve_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, op, reverse, device0):
        res = gp_apis.gp_gsddmmve(graph, input_left, input_right, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmve(graph, input_left, input_right, dim_0, op, reverse, device0):
    return gsddmmve_impl.apply(graph, input_left, input_right, dim_0, op, reverse, device0)

class gsddmmve2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    return gsddmmve2d_impl.apply(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)

class gsddmmvv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, op, reverse, device0):
        res = gp_apis.gp_gsddmmvv(graph, input_left, input_right, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmvv(graph, input_left, input_right, dim_0, op, reverse, device0):
    return gsddmmvv_impl.apply(graph, input_left, input_right, dim_0, op, reverse, device0)

class gsddmmvv2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    return gsddmmvv2d_impl.apply(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)

class test_2out_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
        res1, res2 = gp_apis.gp_test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res1, res2

    @staticmethod
    def backward(ctx, dZ1, dZ2):
        pass #must be implemented

def test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    return test_2out_impl.apply(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)

class test3_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
        res1, res2 = gp_apis.gp_test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res1, res2

    @staticmethod
    def backward(ctx, dZ1, dZ2):
        pass #must be implemented

def test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    return test3_impl.apply(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)

class test4_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0):
        res = gp_apis.gp_test4(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def test4(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0):
    return test4_impl.apply(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0)

class mat_dot_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, M, P, N, device0):
        res = gp_apis.gp_mat_dot(input1, input2, dim1_0, dim1_1, M, P, N, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        #pass #must be implemented
        return None

def mat_dot(input1, input2, dim1_0, dim1_1, M, P, N, device0):
    return mat_dot_impl.apply(input1, input2, dim1_0, dim1_1, M, P, N, device0)

class mat_transpose_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, dim1_0, dim1_1, M, N, device0):
        res = gp_apis.gp_mat_transpose(input1, dim1_0, dim1_1, M, N, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        #pass #must be implemented
        return None

def mat_transpose(input1, dim1_0, dim1_1, M, N, device0):
    return mat_transpose_impl.apply(input1, dim1_0, dim1_1, M, N, device0)

class mat_add_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, M, N, device0):
        res = gp_apis.gp_mat_add(input1, input2, dim1_0, dim1_1, M, N, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def mat_add(input1, input2, dim1_0, dim1_1, M, N, device0):
    return mat_add_impl.apply(input1, input2, dim1_0, dim1_1, M, N, device0)

class mat_mul_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, M, N, device0):
        res = gp_apis.gp_mat_mul(input1, input2, dim1_0, dim1_1, M, N, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented
        return None

def mat_mul(input1, input2, dim1_0, dim1_1, M, N, device0):
    return mat_mul_impl.apply(input1, input2, dim1_0, dim1_1, M, N, device0)

class mat_norm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, M, N, device0):
        res = gp_apis.gp_mat_norm(input1, M, N, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def mat_norm(input1, M, N, device0):
    return mat_norm_impl.apply(input1, M, N, device0)


class LinearNew(th.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, b):
        
        M = x.shape[0]
        print('x.shape:', x.shape, type(x))
        P = x.shape[1]
        N = w.shape[1]
        print('w.shape:', w.shape, type(w))
        device = torch.device("cuda") 
        #(input1, input2, dim1_0, dim1_1, M, P, N, device0)
        # x- M x P
        # w- P x N
        # b - N 
        #xw = gp_apis.gp_mat_dot(x, w, M, N, M, P, N, device)
        xw = mat_dot(x, w, M, N, M, P, N, device)

        # will not effect the graph, as torch graph is disabled in this context
        b = torch.broadcast_to(b, (M, N)) # change this problem

        #y =  gp_apis.gp_mat_add(xw, b, M, N, M, N, device)
        y =  mat_add(xw, b, M, N, M, N, device)

        ctx.save_for_backward(x,w,b)

        return y

    @staticmethod
    def backward(ctx, dL_dY):
        x, w ,b  = ctx.saved_tensors
        M=dL_dY.shape[0]
        P=x.shape[1]
        N=dL_dY.shape[1]

        #print(M)
        #print(P)
        #print(N)
        print(dL_dY.shape)

        assert M == dL_dY.shape[0], "did not match"
        assert N == dL_dY.shape[1], "did not match"

        device = torch.device("cuda") 
        #print('dL_dy:',dL_dY, 'shape: ', dL_dY.shape)

        #wt = gp_apis.gp_mat_transpose(w, w.shape[1], w.shape[0], w.shape[0], w.shape[1], device)
        
        wt = mat_transpose(w, N, P, P, N, device)
        xt = mat_transpose(x, P, M, M, P, device)

        #wt = mat_transpose(w, n, p, p, n, device)
        #xt = mat_transpose(x, p, m, m, p, device)
        #print('w tensor:',w, 'w shape:', w.shape)

        print('wt',wt, 'wt shape:', wt.shape)
        #print('x tensor:',x,' x shape:', x.shape)
        #print('xt tensor:',xt,' x shape:', xt.shape)
        #print('dL_dy:',dL_dY, 'shape: ', dL_dY.shape)
        #d_x = gp_apis.gp_mat_dot(dL_dY, wt, x.shape[0], x.shape[1], dL_dY.shape[0], dL_dY.shape[1], wt.shape[1], device)

        print('-----------------dX--------------------')
        d_x = mat_dot(dL_dY, wt, M, P, M, N, P, device)
        #d_x = mat_dot(dL_dY, wt, m, p, m, n, p, device)
        print('dx tensor:', d_x)
        #d_w = gp_apis.gp_mat_dot(xt, dL_dY,  w.shape[0], w.shape[1], x.shape[0], x.shape[1], dL_dY.shape[1], device)
        #print('xt tensor:',xt,' x shape:', xt.shape[0],' ', xt.shape[1])
        #print('dl', dL_dY)
        print('-----------------dW--------------------')
        d_w = mat_dot(xt, dL_dY, P, N, P, M, N, device)
        #d_w = mat_dot(xt, dL_dY, p, n, p, m, n, device)

        db = torch.sum(dL_dY, dim = 0)
        
        return d_x, d_w, db


def linear_new(x,w,b):
    return LinearNew.apply(x,w,b)

