import numpy as np

class Parameter:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        # Provide a string representation for easier debugging
        return f'Parameter(data={self.data}, grad={self.grad})'
