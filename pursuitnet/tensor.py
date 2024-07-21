try:
    import cupy as np
    HAS_CUPY = True
except ImportError:
    import numpy as np
    HAS_CUPY = False

from pursuitnet import dtypes, ops, device_utils, Value, grad_utils

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
            self.data = None
            self.shape = None
        elif isinstance(data, Tensor):
            self.data = data.data.astype(self._numpy_dtype)
            self.shape = Size(self.data.shape)
        else:
            self.data = np.array(data, dtype=self._numpy_dtype)
            self.shape = Size(self.data.shape)
        
        self.val = Value(self.data, requires_grad=requires_grad) if requires_grad else None
        self._grad = None
    
    @property
    def dtype(self):
        return self._pursuitnet_dtype

    @property
    def requires_grad(self):
        return self.val is not None and self.val.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if value and self.val is None:
            self.val = Value(self.data, requires_grad=True)
        elif not value:
            self.val = None

    @property
    def grad(self):
        if self.val is not None:
            return self.val.grad
        return self._grad
    
    @grad.setter
    def grad(self, value):
        if self.val is not None:
            self.val.grad = value
        else:
            self._grad = value
    
    def backward(self, gradient=None):
        grad_utils.backward(self, gradient)

    def zero_grad(self):
        grad_utils.zero_grad(self)

    def _update_grad(self):
        grad_utils.update_grad(self)

    def __getitem__(self, key):
        return Tensor(self.data[key], dtype=self._pursuitnet_dtype, device=self.device)

    def __repr__(self):
        return ops.tensor_repr(self)

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        return ops.add(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __matmul__(self, other):
        return ops.matmul(self, other)

    def __truediv__(self, other):
        return ops.div(self, other)

    def __sub__(self, other):
        return ops.sub(self, other)
    
    def sum(self):
        return ops.sum(self)

    def max(self, axis=None, keepdims=False):
        return ops.max(self, axis, keepdims)

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