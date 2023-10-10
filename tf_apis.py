import tensorflow as tf
import gp_apis

def mat_dot(input1, input2, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return mat_dot_real(X1, X2, dim;_0, dim;_1, device0)
    return _lambda(input1, input2)

def mat_dot_real(input1, input2, dim;_0, dim;_1, device0):
    out = gp_apis.gp_mat_dot(input1, input2, dim;_0, dim;_1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_mat_dot(dZ1, dZ2, dim;_0, dim;_1, device0)
    return out, grad

def mat_transpose(input1, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1):
        return mat_transpose_real(X1, dim;_0, dim;_1, device0)
    return _lambda(input1)

def mat_transpose_real(input1, dim;_0, dim;_1, device0):
    out = gp_apis.gp_mat_transpose(input1, dim;_0, dim;_1, device0)
    def grad(dZ1):
        return gp_apis.gp_mat_transpose(dZ1, dim;_0, dim;_1, device0)
    return out, grad

def mat_add(input1, input2, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return mat_add_real(X1, X2, dim;_0, dim;_1, device0)
    return _lambda(input1, input2)

def mat_add_real(input1, input2, dim;_0, dim;_1, device0):
    out = gp_apis.gp_mat_add(input1, input2, dim;_0, dim;_1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_mat_add(dZ1, dZ2, dim;_0, dim;_1, device0)
    return out, grad

