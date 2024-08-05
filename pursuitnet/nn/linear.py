# pursuitnet/nn/linear.py

import numpy as np
from pursuitnet.tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features, initial_weights=None, initial_bias=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            initial_weights if initial_weights is not None else np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2. / in_features),
            requires_grad=True
        )
        self.bias = Tensor(
            initial_bias if initial_bias is not None else np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        )

    def forward(self, input):
        self.input = input
        return input @ self.weight + self.bias

    def backward(self, grad_output):
        # Compute gradient for weights and bias
        self.weight.grad = self.input.data.T @ grad_output
        self.bias.grad = np.sum(grad_output, axis=0)
        
        # Return gradient for previous layer
        return grad_output @ self.weight.data.T

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"
