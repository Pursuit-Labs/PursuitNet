# pursuitnet/optim/optimizers.py

import numpy as np
from pursuitnet.tensor import Tensor

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters  # This should be a list of Tensor objects
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.fill(0)

    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})"
