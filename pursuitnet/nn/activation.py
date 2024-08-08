import numpy as np
from pursuitnet.tensor import Tensor

class ReLU:
    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, Tensor):
            raise TypeError("Input to ReLU must be a Tensor")

        output_data = np.maximum(0, input.data)
        output = Tensor(output_data, dtype=input._pursuitnet_dtype, device=input.device, requires_grad=input.requires_grad)

        if output.requires_grad:
            def _backward(grad_output):
                grad_input = grad_output * (input.data > 0)
                input.backward(grad_input)
            
            output._grad_fn = _backward

        return output

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

    def __repr__(self):
        return "ReLU()"