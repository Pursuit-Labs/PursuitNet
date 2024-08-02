class no_grad:
    def __enter__(self):
        self.prev = Tensor.requires_grad
        Tensor.requires_grad = False

    def __exit__(self, exc_type, exc_value, traceback):
        Tensor.requires_grad = self.prev

from .tensor import Tensor