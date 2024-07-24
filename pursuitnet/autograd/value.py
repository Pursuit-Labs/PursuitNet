import numpy as np

class Value:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def backward(self, gradient=None):
        if not self.requires_grad:
            return
        if gradient is None:
            gradient = np.ones_like(self.data)
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad += gradient
        if self.grad_fn:
            self.grad_fn(gradient)

    def zero_grad(self):
        self.grad = None