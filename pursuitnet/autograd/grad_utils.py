import numpy as np

def backward(tensor, gradient=None):
    if tensor.val is not None:
        if gradient is None:
            gradient = np.ones_like(tensor.data)
        tensor.val.backward(gradient)
        update_grad(tensor)

def zero_grad(tensor):
    if tensor.val is not None:
        tensor.val.zero_grad()
    tensor._grad = None

def update_grad(tensor):
    if tensor.val is not None and tensor.val.grad is not None:
        tensor.grad = np.array(tensor.val.grad)