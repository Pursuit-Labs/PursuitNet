# pursuitnet/tensor.py

import numpy as np
import pursuitnet.ops as ops

class Tensor:
    def __init__(self, data, dtype=np.float32, device='cpu', requires_grad=False):
        self.device = device
        self._pursuitnet_dtype = dtype
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros_like(self.data)
        self._backward = lambda grad=None: None

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad.fill(0)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        self._backward(grad)

    def __getitem__(self, key):
        return Tensor(self.data[key], dtype=self._pursuitnet_dtype, device=self.device, requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        return ops.add(self, other)

    def __sub__(self, other):
        return ops.sub(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __truediv__(self, other):
        return ops.div(self, other)

    def __matmul__(self, other):
        return ops.matmul(self, other)
    
    def reshape(self, *shape):
        reshaped_data = self.data.reshape(shape)
        return Tensor(reshaped_data, dtype=self._pursuitnet_dtype, device=self.device, requires_grad=self.requires_grad)

    @classmethod
    def zeros(cls, *shape, dtype=np.float32, device='cpu', requires_grad=False):
        return cls(np.zeros(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype=np.float32, device='cpu', requires_grad=False):
        return cls(np.ones(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def random(cls, *shape, dtype=np.float32, device='cpu', requires_grad=False):
        return cls(np.random.randn(*shape), dtype=dtype, device=device, requires_grad=requires_grad)
