import numpy as np
from pursuitnet.tensor import Tensor
from pursuitnet.autograd.value import Value

class CrossEntropyLoss:
    def __init__(self):
        self.input = None
        self.target = None
        self.output = None
        self.softmax_output = None

    def forward(self, input_tensor: Tensor, target: Tensor) -> Tensor:
        self.input = input_tensor
        self.target = target

        # Compute softmax
        exp_input = np.exp(input_tensor.data - np.max(input_tensor.data, axis=1, keepdims=True))
        self.softmax_output = exp_input / np.sum(exp_input, axis=1, keepdims=True)

        # Compute cross entropy loss
        batch_size = input_tensor.shape[0]
        target_indices = target.data.astype(int)  # Ensure target indices are integers
        loss = -np.sum(np.log(self.softmax_output[range(batch_size), target_indices])) / batch_size

        # Ensure loss is a scalar
        loss = np.array(loss).reshape(())

        self.output = Tensor(loss, dtype=input_tensor.dtype, device=input_tensor.device, requires_grad=input_tensor.requires_grad)

        if self.output.requires_grad:
            self.output.val = Value(self.output.data, requires_grad=True)
            self.output.val.grad_fn = self.backward

        return self.output

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = 1.0

        batch_size = self.input.shape[0]
        
        # Compute cross entropy loss gradients
        dx = self.softmax_output.copy()
        target_indices = self.target.data.astype(int)  # Ensure target indices are integers
        dx[range(batch_size), target_indices] -= 1
        dx /= batch_size

        # Apply grad_output
        dx *= grad_output

        if self.input.requires_grad:
            if self.input.grad is None:
                self.input.grad = dx
            else:
                self.input.grad += dx

    def __call__(self, input_tensor: Tensor, target: Tensor) -> Tensor:
        return self.forward(input_tensor, target)