import numpy as np
from pursuitnet.tensor import Tensor

def softmax(x):
    x = np.atleast_2d(x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class CrossEntropyLoss:
    def __init__(self):
        self.input = None
        self.target = None
        self.softmax_output = None

    def forward(self, input_tensor: Tensor, target) -> Tensor:
        self.input = input_tensor
        self.target = target

        probs = softmax(input_tensor.data)
        self.softmax_output = probs

        batch_size = input_tensor.shape[0]
        target_data = target.data if isinstance(target, Tensor) else target
        target_indices = target_data.astype(int)
        loss = -np.sum(np.log(probs[np.arange(batch_size), target_indices] + 1e-9)) / batch_size

        loss = np.array(loss).reshape(())

        output = Tensor(loss, dtype=input_tensor._pursuitnet_dtype, device=input_tensor.device, requires_grad=input_tensor.requires_grad)
        
        if output.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = np.array(1.0)
                dx = self.softmax_output.copy()
                dx[np.arange(batch_size), target_indices] -= 1
                dx /= batch_size
                dx *= grad  # Scale by incoming gradient
                self.input.backward(dx)
            
            output._grad_fn = _backward

        return output

    def __call__(self, input_tensor: Tensor, target) -> Tensor:
        return self.forward(input_tensor, target)