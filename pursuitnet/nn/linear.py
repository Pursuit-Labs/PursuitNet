import numpy as np
from .module import Module
import pursuitnet as pn
from ..autograd.parameter import Parameter

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32) * np.sqrt(2.0 / in_features))
        if bias:
            self.bias = Parameter(np.zeros(out_features).astype(np.float32))
        else:
            self.bias = None

    def forward(self, x: pn.Tensor) -> pn.Tensor:
        if not isinstance(x, pn.Tensor):
            raise TypeError("Input is not a Tensor")

        print(f"Linear forward: weight norm = {np.linalg.norm(self.weight.data)}, bias norm = {np.linalg.norm(self.bias.data) if self.bias is not None else 0}")

        # Perform linear transformation
        output = x @ pn.Tensor(self.weight.data.T, requires_grad=True)
        if self.bias is not None:
            output = output + pn.Tensor(self.bias.data, requires_grad=True)

        return output

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"