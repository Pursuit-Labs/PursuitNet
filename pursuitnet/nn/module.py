import numpy as np
import pursuitnet as pn
from ..autograd.parameter import Parameter

class Module:
    def __init__(self):
        self._parameters = []
        self._modules = {}

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):  # Correctly register Parameters
            self._parameters.append(value)
        object.__setattr__(self, name, value)

    def __repr__(self):
        module_str = self.__class__.__name__ + '('
        for name, module in self._modules.items():
            module_str += '\n  (' + name + '): ' + repr(module)
        module_str += '\n)'
        return module_str

    def parameters(self):
        # Collect parameters from this module and all sub-modules
        params = self._parameters[:]
        for name, module in self._modules.items():
            params += module.parameters()
        return params

    def zero_grad(self):
        # Zero the gradients of all parameters
        for param in self.parameters():
            param.zero_grad()
