import numpy as np

def update_grad(tensor):
    if tensor.val is not None and tensor.val.grad is not None:
        tensor.grad = np.array(tensor.val.grad)

def backward(tensor, gradient=None):
    if tensor.requires_grad:
        tensor.backward(gradient)

def zero_grad(tensor):
    tensor.zero_grad()