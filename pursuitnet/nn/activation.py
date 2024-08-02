import numpy as np
from .module import Module
import pursuitnet as pn

class ReLU(Module):
    def forward(self, x: pn.Tensor) -> pn.Tensor:
        if not isinstance(x, pn.Tensor):
            raise TypeError("Input to ReLU is not a Tensor")

        # Create a zero tensor with the same shape as x
        zero_tensor = pn.Tensor(np.zeros_like(x.data), requires_grad=False)
        result = x.max(other=zero_tensor)  # Element-wise max with zero

        if not isinstance(result, pn.Tensor):
            raise TypeError("Output from ReLU is not a Tensor")

        return result

    def __repr__(self):
        return "ReLU()"
