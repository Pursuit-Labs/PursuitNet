import numpy as np
from .module import Module

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def __repr__(self):
        return "ReLU()"
