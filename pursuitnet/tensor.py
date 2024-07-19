try:
    import cupy as np
    HAS_CUPY = True
except ImportError:
    import numpy as np
    HAS_CUPY = False

from . import dtypes
from .autograd import Value

class Tensor:
    def __init__(self, data, dtype=dtypes.float32, device='cpu', requires_grad=False):
        self.device = device
        self._pursuitnet_dtype = dtype
        
        if hasattr(dtype, 'numpy_dtype'):
            self._numpy_dtype = dtype.numpy_dtype
        else:
            self._numpy_dtype = dtype  # Assume it's already a numpy dtype
        
        if data is None:
            self.data = None
            self.shape = None
        elif isinstance(data, Tensor):
            self.data = data.data.astype(self._numpy_dtype)
            self.shape = self.data.shape
        else:
            self.data = np.array(data, dtype=self._numpy_dtype)
            self.shape = self.data.shape
        
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
        if self.val is not None:
            if gradient is None:
                gradient = np.ones_like(self.data)
            self.val.backward(gradient)
            self._update_grad()
    
    def _update_grad(self):
        if self.val is not None and self.val.grad is not None:
            self.grad = np.array(self.val.grad)

    def zero_grad(self):
        if self.val is not None:
            self.val.zero_grad()

    def __getitem__(self, key):
        return Tensor(self.data[key], dtype=self._numpy_dtype, device=self.device)

    def __repr__(self):
        def format_element(x):
            if np.issubdtype(self._numpy_dtype, np.bool_) and x == True:
                return f" {str(x)}"
            elif np.issubdtype(self._numpy_dtype, np.floating):
                if abs(x) < 1e-4 and x != 0:
                    return f"{x:.4e}"
                elif abs(x) >= 1e4:
                    return f"{x:.4e}"
                elif x == int(x):
                    return f"{int(x)}."
                else:
                    return f"{x:.4f}".rstrip('0').rstrip('.') + ('0' * (4 - len(f"{x:.4f}".rstrip('0').rstrip('.').split('.')[-1])))
            elif np.issubdtype(self._numpy_dtype, np.complexfloating):
                return f"{x.real:.4f}{x.imag:+.4f}j"
            elif np.issubdtype(self._numpy_dtype, np.integer):
                return f"{x}"
            else:
                return str(x)

        data_str = np.array2string(
            self.data,
            separator=', ',
            formatter={'all': format_element},
            threshold=1000,
            edgeitems=3,
            max_line_width=75
        )
        
        data_str = data_str.replace('\n ', '\n        ')
        
        if data_str.endswith('\t'):
            data_str = data_str[:-1]
        
        if len(self.data.shape) == 1:
            data_str = data_str.replace('\t', '')

        dtype_str = f"pursuitnet.{self._pursuitnet_dtype.__name__}"

        if self._pursuitnet_dtype == dtypes.float32 or self._pursuitnet_dtype == dtypes.bool:
            return f"Tensor({data_str})"
        else:
            return f"Tensor({data_str}, dtype={dtype_str})"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        from .autograd.functions import Add
        if isinstance(other, Tensor):
            result = Tensor(np.add(self.data, other.data), dtype=self._numpy_dtype, device=self.device)
            result.val = Add.apply(self.val, other.val) if (self.val is not None or other.val is not None) else None
        else:
            result = Tensor(np.add(self.data, other), dtype=self._numpy_dtype, device=self.device)
            result.val = Add.apply(self.val, other) if self.val is not None else None
        return result

    def __mul__(self, other):
        from .autograd.functions import Mul
        if isinstance(other, Tensor):
            result = Tensor(np.multiply(self.data, other.data), dtype=self._numpy_dtype, device=self.device)
            result.val = Mul.apply(self.val, other.val) if (self.val is not None or other.val is not None) else None
        else:
            result = Tensor(np.multiply(self.data, other), dtype=self._numpy_dtype, device=self.device)
            result.val = Mul.apply(self.val, other) if self.val is not None else None
        return result

    def __matmul__(self, other):
        from .autograd.functions import MatMul
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication is only supported between Tensors")
        result = Tensor(np.matmul(self.data, other.data), dtype=self._numpy_dtype, device=self.device)
        result.val = MatMul.apply(self.val, other.val) if (self.val is not None or other.val is not None) else None
        return result

    def sum(self):
        result = Tensor(np.sum(self.data), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward(grad_output):
                self.backward(np.full_like(self.data, grad_output.data))
            result.val.grad_fn = _backward
        return result
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(np.divide(self.data, other.data), dtype=self._numpy_dtype, device=self.device)
            result.val = self.val / other.val if (self.val is not None and other.val is not None) else None
        else:
            result = Tensor(np.divide(self.data, other), dtype=self._numpy_dtype, device=self.device)
            result.val = self.val / other if self.val is not None else None
        return result

    def __sub__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(np.subtract(self.data, other.data), dtype=self._numpy_dtype, device=self.device)
            result.val = self.val - other.val if (self.val is not None and other.val is not None) else None
        else:
            result = Tensor(np.subtract(self.data, other), dtype=self._numpy_dtype, device=self.device)
            result.val = self.val - other if self.val is not None else None
        return result
    
    def max(self, axis=None, keepdims=False):
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims), dtype=self._numpy_dtype, device=self.device)

    def min(self, axis=None, keepdims=False):
        return Tensor(np.min(self.data, axis=axis, keepdims=keepdims), dtype=self._numpy_dtype, device=self.device)

    def mean(self, axis=None, keepdims=False):
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), dtype=self._numpy_dtype, device=self.device)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(self.data.reshape(shape), dtype=self._numpy_dtype, device=self.device)

    def transpose(self, *axes):
        if not axes:
            axes = None
        return Tensor(np.transpose(self.data, axes), dtype=self._numpy_dtype, device=self.device)

    def to(self, device):
        if device == self.device:
            return self
        else:
            if device == 'cpu' and HAS_CUPY:
                new_tensor = Tensor(self.data.get(), dtype=self._numpy_dtype, device=device)
            elif device == 'gpu' and HAS_CUPY:
                new_tensor = Tensor(np.array(self.data), dtype=self._numpy_dtype, device=device)
            else:
                new_tensor = Tensor(self.data, dtype=self._numpy_dtype, device=device)
            new_tensor.val = self.val
            return new_tensor

    @classmethod
    def zeros(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return cls(np.zeros(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return cls(np.ones(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def random(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return cls(np.random.random(shape), dtype=dtype, device=device, requires_grad=requires_grad)