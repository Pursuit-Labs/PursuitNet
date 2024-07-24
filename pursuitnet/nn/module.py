import numpy as np

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
        elif isinstance(value, Parameter):
            self._parameters.append(value)
        object.__setattr__(self, name, value)

    def __repr__(self):
        module_str = self.__class__.__name__ + '('
        for name, module in self._modules.items():
            module_str += '\n  (' + name + '): ' + repr(module)
        module_str += '\n)'
        return module_str

    def parameters(self):
        params = self._parameters[:]
        for name, module in self._modules.items():
            params += module.parameters()
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = np.ones_like(self.output)
        for param in self.parameters():
            param.grad = self._compute_gradient(param, grad_output)

    def step(self, lr=0.01):
        for param in self.parameters():
            param.data -= lr * param.grad

    def _compute_gradient(self, param, grad_output):
        # This method should be overridden by subclasses to compute the gradient
        raise NotImplementedError

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

    def __repr__(self):
        return f'Parameter(data={self.data}, grad={self.grad})'
