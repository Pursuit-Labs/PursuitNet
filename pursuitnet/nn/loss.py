import numpy as np
from pursuitnet.tensor import Tensor

def softmax(x):
    # Handle both 1D and 2D arrays
    x = np.atleast_2d(x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class CrossEntropyLoss:
    def __init__(self):
        self.input = None
        self.target = None
        self.softmax_output = None

    def forward(self, input_tensor: Tensor, target) -> Tensor:
        # Store input and target
        self.input = input_tensor
        self.target = target

        # Compute softmax
        probs = softmax(input_tensor.data)
        self.softmax_output = probs

        # Compute cross-entropy loss
        batch_size = input_tensor.shape[0]
        target_data = target.data if isinstance(target, Tensor) else target
        target_indices = target_data.astype(int)  # Ensure target indices are integers
        loss = -np.sum(np.log(probs[np.arange(batch_size), target_indices] + 1e-9)) / batch_size

        # Ensure loss is a scalar
        loss = np.array(loss).reshape(())

        # Create output tensor for the loss
        output = Tensor(loss, dtype=input_tensor._pursuitnet_dtype, device=input_tensor.device, requires_grad=input_tensor.requires_grad)
        output._backward = self.backward  # Attach backward method to the output tensor

        return output

    def backward(self, grad=None) -> np.ndarray:
        print("Backward method called")
        print(f"Input requires_grad: {self.input.requires_grad}")
        print(f"Initial input grad: {self.input.grad}")
        # Compute cross-entropy loss gradients
        batch_size = self.input.shape[0]

        dx = self.softmax_output.copy()
        target_data = self.target.data if isinstance(self.target, Tensor) else self.target
        target_indices = target_data.astype(int)
        dx[np.arange(batch_size), target_indices] -= 1
        dx /= batch_size

        print(f"CrossEntropyLoss backward: input grad norm = {np.linalg.norm(dx)}")

        if self.input.requires_grad:
            if self.input.grad is None:
                self.input.grad = dx
            else:
                self.input.grad += dx

        print(f"CrossEntropyLoss backward: final input grad norm = {np.linalg.norm(self.input.grad)}")

        return dx

    def __call__(self, input_tensor: Tensor, target) -> Tensor:
        return self.forward(input_tensor, target)