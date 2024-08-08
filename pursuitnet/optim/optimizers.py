import numpy as np
from pursuitnet.tensor import Tensor

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.betas = (beta1, beta2)
        self.epsilon = epsilon
        self.t = 0
        self.m = [np.zeros_like(param.data, dtype=np.float32) for param in parameters]
        self.v = [np.zeros_like(param.data, dtype=np.float32) for param in parameters]

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.fill(0)
            else:
                param.grad = np.zeros_like(param.data)

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            m_t = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            v_t = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            self.m[i] = m_t
            self.v[i] = v_t
            
            m_hat = m_t / (1 - self.beta1 ** self.t)
            v_hat = v_t / (1 - self.beta2 ** self.t)
            
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Ensure update has the same shape as param.data
            if update.shape != param.data.shape:
                if len(param.data.shape) == 1:
                    update = np.sum(update, axis=0)
                else:
                    raise ValueError(f"Unexpected shape mismatch: param {param.data.shape}, update {update.shape}")
            
            param.data -= update

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})"