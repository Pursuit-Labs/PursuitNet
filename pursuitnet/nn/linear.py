import numpy as np
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Ensure weights are initialized with the same method as PyTorch
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2. / in_features)
        self.bias = np.zeros(out_features).astype(np.float32)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"
