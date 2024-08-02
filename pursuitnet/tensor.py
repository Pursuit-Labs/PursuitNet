import numpy as np
import pursuitnet as pn
import pursuitnet.ops as ops
from pursuitnet.dtype import dtypes
from pursuitnet.device import device_utils

class Size:
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f'pursuitnet.Size({list(self.shape)})'

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return iter(self.shape)

    def __getitem__(self, idx):
        return self.shape[idx]

    def __len__(self):
        return len(self.shape)

class Tensor:
    def __init__(self, data, dtype=dtypes.float32, device='cpu', requires_grad=False):
        self.device = device
        self._pursuitnet_dtype = dtype
        self._numpy_dtype = dtype.numpy_dtype

        if data is None:
            self._data = None
        elif isinstance(data, Tensor):
            self._data = data.data.astype(self._numpy_dtype)
        else:
            self._data = np.array(data, dtype=self._numpy_dtype)

        if requires_grad:
            self.val = pn.Value(self._data, requires_grad=True)
        else:
            self.val = None
        self.requires_grad = requires_grad
        self._grad = None
    
    @property
    def shape(self):
        return Size(self._data.shape) if self._data is not None else None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.array(value)
        if self.val is not None:
            self.val.data = self._data

    @property
    def dtype(self):
        return self._pursuitnet_dtype

    @property
    def requires_grad(self):
        return self.val is not None and self.val.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if value and self.val is None:
            self.val = pn.Value(self.data, requires_grad=True)
        elif not value:
            self.val = None

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if isinstance(value, Tensor):
            self._grad = value.data
        else:
            self._grad = value

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        
        if hasattr(self, 'backward_hook'):
            grad = self.backward_hook(grad)
        
        if self.requires_grad:
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
        
        if self.val is not None and self.val.grad_fn is not None:
            self.val.grad_fn(grad)

        return grad

    def zero_grad(self):
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            else:
                self.grad.fill(0)
            if self.val is not None:
                self.val.zero_grad()

    def __getitem__(self, key):
        return Tensor(self.data[key], dtype=self._pursuitnet_dtype, device=self.device)

    def __repr__(self):
        return ops.tensor_repr(self)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.shape[0]

    def __eq__(self, other):
        return ops.eq(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __matmul__(self, other):
        return ops.matmul(self, other)

    def __truediv__(self, other):
        return ops.div(self, other)

    def __sub__(self, other):
        return ops.sub(self, other)

    def sum(self, axis=None, keepdims=False):
        """
        Sum of tensor elements over a given axis.
        """
        return ops.sum(self, axis=axis, keepdims=keepdims)

    def __add__(self, other):
        return ops.add(self, other)

    def max(self, axis=None, keepdims=False, other=None):
        if other is not None:
            if isinstance(other, Tensor):
                # Ensure the shapes are compatible for broadcasting
                if not np.broadcast_shapes(self.shape, other.shape):
                    raise ValueError("Shapes are not compatible for broadcasting")
                return Tensor(np.maximum(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
            else:
                return Tensor(np.maximum(self.data, other), requires_grad=self.requires_grad)
        else:
            max_data = np.max(self.data, axis=axis, keepdims=keepdims)
            return Tensor(max_data, requires_grad=self.requires_grad)


    def min(self, axis=None, keepdims=False):
        return ops.min(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return ops.mean(self, axis, keepdims)

    def reshape(self, *shape):
        return ops.reshape(self, *shape)

    def transpose(self, *axes):
        return ops.transpose(self, *axes)

    def to(self, device):
        return device_utils.to_device(self, device)

    @classmethod
    def zeros(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return ops.zeros(cls, *shape, dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return ops.ones(cls, *shape, dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def random(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return ops.random(cls, *shape, dtype=dtype, device=device, requires_grad=requires_grad)
