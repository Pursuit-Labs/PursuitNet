try:
    import cupy as np
    HAS_CUPY = True
except ImportError:
    import numpy as np
    HAS_CUPY = False
    
from . import dtypes

class Tensor:
    def __init__(self, data, dtype=dtypes.float32, device='cpu', requires_grad=False):
        self.device = device
        self._pursuitnet_dtype = dtype
        self.requires_grad = requires_grad
        
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
        
        self.grad = None
        self.grad_fn = None
    
    @property
    def dtype(self):
        return self._pursuitnet_dtype

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
        
        # Remove extra spaces at the start of lines and replace newlines with tabs
        data_str = data_str.replace('\n ', '\n        ')
        
        # Remove the last tab if it exists
        if data_str.endswith('\t'):
            data_str = data_str[:-1]
        
        # For 1D tensors, remove the tab
        if len(self.data.shape) == 1:
            data_str = data_str.replace('\t', '')

        dtype_str = f"pursuitnet.{self._pursuitnet_dtype.__name__}"

        if self._pursuitnet_dtype == dtypes.float32 or self._pursuitnet_dtype == dtypes.bool:
            return f"Tensor({data_str})"
        else:
            return f"Tensor({data_str}, dtype={dtype_str})"

    def __str__(self):
        return self.__repr__()

    def backward(self, gradient=None):
        if not self.requires_grad:
            return

        if gradient is None:
            gradient = np.ones_like(self.data)

        if self.grad is None:
            self.grad = np.array(gradient)
        else:
            self.grad += np.array(gradient)

        if self.grad_fn:
            self.grad_fn()

    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.grad)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(np.add(self.data, other.data), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.backward(result.grad)
                if other.requires_grad:
                    other.backward(result.grad)
            result.grad_fn = _backward
        
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(np.subtract(self.data, other.data), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.backward(result.grad)
                if other.requires_grad:
                    other.backward(-result.grad)
            result.grad_fn = _backward
        
        return result

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(np.multiply(self.data, other.data), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.backward(np.multiply(result.grad, other.data))
                if other.requires_grad:
                    other.backward(np.multiply(result.grad, self.data))
            result.grad_fn = _backward
        
        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(np.divide(self.data, other.data), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.backward(np.divide(result.grad, other.data))
                if other.requires_grad:
                    other.backward(-np.divide(np.multiply(result.grad, self.data), np.square(other.data)))
            result.grad_fn = _backward
        
        return result

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication is only supported between Tensors")
        result = Tensor(np.matmul(self.data, other.data), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.backward(np.matmul(result.grad, other.data.T))
                if other.requires_grad:
                    other.backward(np.matmul(self.data.T, result.grad))
            result.grad_fn = _backward
        
        return result

    def __getitem__(self, key):
        return Tensor(self.data[key], dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)

    def __setitem__(self, key, value):
        self.data[key] = value

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(self.data.reshape(shape), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)

    def transpose(self, *axes):
        if not axes:
            axes = None
        return Tensor(np.transpose(self.data, axes), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)

    def to(self, device):
        if device == self.device:
            return self
        else:
            if device == 'cpu' and HAS_CUPY:
                return Tensor(self.data.get(), dtype=self._numpy_dtype, device=device, requires_grad=self.requires_grad)
            elif device == 'gpu' and HAS_CUPY:
                return Tensor(np.array(self.data), dtype=self._numpy_dtype, device=device, requires_grad=self.requires_grad)
            else:
                return Tensor(self.data, dtype=self._numpy_dtype, device=device, requires_grad=self.requires_grad)

    @property
    def T(self):
        return self.transpose()

    def mean(self, axis=None, keepdims=False):
        result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    scale = np.ones_like(self.data) / self.data.size
                    self.backward(result.grad * scale)
            result.grad_fn = _backward
        
        return result

    def sum(self, axis=None, keepdims=False):
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.backward(np.ones_like(self.data) * result.grad)
            result.grad_fn = _backward
        
        return result

    def max(self, axis=None, keepdims=False):
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)

    def min(self, axis=None, keepdims=False):
        return Tensor(np.min(self.data, axis=axis, keepdims=keepdims), dtype=self._numpy_dtype, device=self.device, requires_grad=self.requires_grad)

    @classmethod
    def zeros(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return cls(np.zeros(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return cls(np.ones(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def random(cls, *shape, dtype=dtypes.float32, device='cpu', requires_grad=False):
        return cls(np.random.random(shape), dtype=dtype, device=device, requires_grad=requires_grad)