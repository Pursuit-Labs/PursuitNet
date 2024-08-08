# pursuitnet/nn/activation.py

import numpy as np
from pursuitnet.tensor import Tensor

class ReLU:
    def forward(self, input: Tensor) -> Tensor:
        # Ensure input is a Tensor
        if not isinstance(input, Tensor):
            raise TypeError("Input to ReLU must be a Tensor")

        # Element-wise maximum with zero
        output_data = np.maximum(0, input.data)

        # Create output Tensor
        output = Tensor(output_data, dtype=input._pursuitnet_dtype, device=input.device, requires_grad=input.requires_grad)

        # Save input for backpropagation
        self.input = input

        return output

    def backward(self, grad_output):
        # Compute gradient
        grad_input = grad_output * (self.input.data > 0)
        
        # Propagate gradient to input tensor
        self.input.backward(grad_input)
        
        return grad_input
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

    def __repr__(self):
        return "ReLU()"
