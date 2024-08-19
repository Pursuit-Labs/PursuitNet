import numpy as np
import pursuitnet as pn
import pursuitnet.ops as ops
import pursuitnet.dtype as pn_dtype

class Tensor:
    def __init__(self, data, dtype=pn_dtype.float32, device='cpu', requires_grad=False):
        self.device = device
        self._pursuitnet_dtype = dtype

        if dtype is None:
            self.data = np.array(data)
        elif dtype == bool:
            self.data = np.array(data, dtype=bool)
        else:
            self.data = np.array(data, dtype=dtype())

        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
        self._backward_hooks = []
        self._grad_fn = None
        

    @property
    def shape(self):
        return self.data.shape
    
    def __len__(self):
        return len(self.data)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("Gradients can only be implicitly created for scalar outputs")
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        # Ensure grad has the same shape as self.grad
        if self.grad.shape != grad.shape:
            if self.grad.size == grad.size:
                grad = grad.reshape(self.grad.shape)
            else:
                # If sizes don't match, we need to sum over the extra dimensions
                sum_axes = tuple(range(grad.ndim - self.grad.ndim))
                grad = grad.sum(axis=sum_axes).reshape(self.grad.shape)

        self.grad += grad

        if self._grad_fn:
            self._grad_fn(grad)

        for hook in self._backward_hooks:
            hook(grad)

    def register_hook(self, hook):
        self._backward_hooks.append(hook)

    def __getitem__(self, key):
        return Tensor(self.data[key], dtype=self._pursuitnet_dtype, device=self.device, requires_grad=self.requires_grad)

    def argmax(self, axis=None):
        result = np.argmax(self.data, axis=axis)
        return Tensor(result, dtype=self._pursuitnet_dtype, device=self.device, requires_grad=False)
    
    def item(self):
        if self.data.size != 1:
            raise ValueError("Only one element tensors can be converted to Python scalars")
        return self.data.item()

    def __repr__(self):
        def format_element(x):
            if np.issubdtype(self.data.dtype, np.bool_) and x:
                return f" {str(x)}"
            elif np.issubdtype(self.data.dtype, np.floating):
                if abs(x) < 1e-4 and x != 0:
                    return f"{x:.4e}"
                elif abs(x) >= 1e4:
                    return f"{x:.4e}"
                elif x == int(x):
                    return f"{int(x)}."
                else:
                    return f"{x:.4f}".rstrip('0').rstrip('.') + ('0' * (4 - len(f"{x:.4f}".rstrip('0').rstrip('.').split('.')[-1])))
            elif np.issubdtype(self.data.dtype, np.complexfloating):
                return f"{x.real:.4f}{x.imag:+.4f}j"
            elif np.issubdtype(self.data.dtype, np.integer):
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

        if self._pursuitnet_dtype in [pn_dtype.float32, pn_dtype.bool]:
            return f"Tensor({data_str})"
        else:
            return f"Tensor({data_str}, dtype={dtype_str})"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        return ops.add(self, other)

    def __radd__(self, other):
        return ops.add(self, other)

    def __sub__(self, other):
        return ops.sub(self, other)

    def __rsub__(self, other):
        return ops.sub(other, self)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __rmul__(self, other):
        return ops.mul(self, other)

    def __truediv__(self, other):
        return ops.div(self, other)

    def __rtruediv__(self, other):
        return ops.div(other, self)

    def __matmul__(self, other):
        return ops.matmul(self, other)

    def reshape(self, *shape):
        reshaped_data = self.data.reshape(*shape)
        return Tensor(reshaped_data, dtype=self._pursuitnet_dtype, device=self.device, requires_grad=self.requires_grad)
    
    def astype(self, dtype):
        return Tensor(self.data.astype(dtype), dtype=dtype, device=self.device, requires_grad=self.requires_grad)

    def sum(self, axis=None, keepdims=False):
        if self.data.dtype == bool:
            summed_data = np.sum(self.data.astype(int), axis=axis, keepdims=keepdims)
        else:
            summed_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(summed_data, dtype=pn.float32, device=self.device, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                if axis is not None:
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        for ax in (axis if isinstance(axis, tuple) else (axis,)):
                            grad_shape[ax] = 1
                    grad = np.broadcast_to(grad.reshape(grad_shape), self.data.shape)
                self.backward(grad)
            result._grad_fn = _backward
        
        return result

    def mean(self, axis=None, keepdims=False):
        if self.data.dtype == bool:
            mean_data = np.mean(self.data.astype(float), axis=axis, keepdims=keepdims)
        else:
            mean_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(mean_data, dtype=pn.float32, device=self.device, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                if axis is not None:
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        for ax in (axis if isinstance(axis, tuple) else (axis,)):
                            grad_shape[ax] = 1
                    grad = np.broadcast_to(grad.reshape(grad_shape), self.data.shape)
                
                # Calculate the scaling factor for the gradient
                if axis is None:
                    scale = 1 / self.data.size
                else:
                    scale = 1 / np.prod([self.data.shape[ax] for ax in (axis if isinstance(axis, tuple) else (axis,))])
                
                self.backward(grad * scale)
            result._grad_fn = _backward
        
        return result

    
    def max(self):
        max_data = self.data.max()
        result = Tensor(max_data, dtype=self._pursuitnet_dtype, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward(grad):
                grad_input = np.zeros_like(self.data)
                grad_input[self.data == max_data] = grad
                self.backward(grad_input)
            result.register_hook(_backward)
        return result
    
    def min(self):
        min_data = self.data.min()
        result = Tensor(min_data, dtype=self._pursuitnet_dtype, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward(grad):
                grad_input = np.zeros_like(self.data)
                grad_input[self.data == min_data] = grad
                self.backward(grad_input)
            result.register_hook(_backward)
        return result
    
    def transpose(self):
        transposed_data = self.data.T
        return Tensor(transposed_data, dtype=self._pursuitnet_dtype, device=self.device, requires_grad=self.requires_grad)

    @classmethod
    def zeros(cls, *shape, dtype=pn_dtype.float32, device='cpu', requires_grad=False):
        return cls(np.zeros(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype=pn_dtype.float32, device='cpu', requires_grad=False):
        return cls(np.ones(shape), dtype=dtype, device=device, requires_grad=requires_grad)

    @classmethod
    def random(cls, *shape, dtype=pn_dtype.float32, device='cpu', requires_grad=False):
        return cls(np.random.randn(*shape), dtype=dtype, device=device, requires_grad=requires_grad)
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data, dtype=bool, device=self.device)
        else:
            return Tensor(self.data == other, dtype=bool, device=self.device)

    def __ne__(self, other):
        return ~(self == other)

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data, dtype=bool, device=self.device)
        else:
            return Tensor(self.data < other, dtype=bool, device=self.device)

    def __le__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data, dtype=bool, device=self.device)
        else:
            return Tensor(self.data <= other, dtype=bool, device=self.device)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data, dtype=bool, device=self.device)
        else:
            return Tensor(self.data > other, dtype=bool, device=self.device)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data >= other.data, dtype=bool, device=self.device)
        else:
            return Tensor(self.data >= other, dtype=bool, device=self.device)
