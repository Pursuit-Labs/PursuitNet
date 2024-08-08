from pursuitnet.nn.module import Module
from pursuitnet.autograd.parameter import Parameter
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases as Parameter objects
        self.weight = Parameter(np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2. / in_features))
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32))

    def forward(self, input):
        return input @ self.weight + self.bias

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"

