import numpy as np
from pursuitnet.autograd.parameter import Parameter
import pursuitnet as pn

class Optim:
    def __init__(self):
        pass

    def __del__(self):
        pass

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

class Adam(Optim):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, clip_value=None):
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.clip_value = clip_value
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data if isinstance(p.grad, pn.Tensor) else p.grad
            
            # Gradient clipping
            if self.clip_value is not None:
                g = np.clip(g, -self.clip_value, self.clip_value)

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (g ** 2)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= update
            print(f"Adam step: param {i} grad norm = {np.linalg.norm(g)}, update norm = {np.linalg.norm(update)}")

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad.fill(0)

# This function will be accessible as optim.Adam
def create_adam_optimizer(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, clip_value=None):
    return Adam(params, lr, betas, eps, clip_value)