import numpy as np
import pursuitnet as pn

class Parameter(pn.Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
    
    def zero_grad(self):
        self.grad = None