# pursuitnet/ops/basic_ops.py

try:
    import cupy as np
    HAS_CUPY = True
except ImportError:
    import numpy as np
    HAS_CUPY = False

from ..autograd import operations
import pursuitnet as pn
from pursuitnet.autograd.value import Value
from pursuitnet.autograd.parameter import Parameter

def add(a, b):
    if not isinstance(a, pn.Tensor):
        a, b = b, a  # Ensure 'a' is always a Tensor
    if not isinstance(b, pn.Tensor):
        b_data = np.array(b)
    else:
        b_data = b.data

    result = pn.Tensor(a.data + b_data, requires_grad=a.requires_grad)
    if result.requires_grad:
        def _backward(grad):
            if a.requires_grad:
                a.backward(grad)
            if isinstance(b, pn.Tensor) and b.requires_grad:
                b.backward(grad)
        result._grad_fn = _backward
    return result

def mul(a, b):
    if not isinstance(a, pn.Tensor):
        a, b = b, a  # Ensure 'a' is always a Tensor
    if not isinstance(b, pn.Tensor):
        b_data = np.array(b)
    else:
        b_data = b.data

    result = pn.Tensor(a.data * b_data, requires_grad=a.requires_grad)
    if result.requires_grad:
        def _backward(grad):
            if a.requires_grad:
                a.backward(grad * b_data)
            if isinstance(b, pn.Tensor) and b.requires_grad:
                b.backward(grad * a.data)
        result._grad_fn = _backward
    return result

def matmul(a, b):
    result = pn.Tensor(np.matmul(a.data, b.data), requires_grad=(a.requires_grad or b.requires_grad))
    if result.requires_grad:
        def _backward(grad):
            if a.requires_grad:
                a.backward(np.matmul(grad, b.data.T))
            if b.requires_grad:
                b.backward(np.matmul(a.data.T, grad))
        result._grad_fn = _backward
    return result

def div(a, b):
    if isinstance(b, a.__class__):
        result = a.__class__(np.divide(a.data, b.data), dtype=a._pursuitnet_dtype, device=a.device)
        if a.requires_grad or b.requires_grad:
            result.requires_grad = True
            result.val = Value(result.data, requires_grad=True)
            def _backward(grad_output):
                if a.requires_grad:
                    a.backward(grad_output / b.data)
                if b.requires_grad:
                    b.backward(-grad_output * a.data / (b.data ** 2))
            result.val.grad_fn = _backward
    else:
        result = a.__class__(np.divide(a.data, b), dtype=a._pursuitnet_dtype, device=a.device)
        if a.requires_grad:
            result.requires_grad = True
            result.val = Value(result.data, requires_grad=True)
            def _backward(grad_output):
                a.backward(grad_output / b)
            result.val.grad_fn = _backward
    return result

def sub(a, b):
    if isinstance(b, a.__class__):
        result = a.__class__(np.subtract(a.data, b.data), dtype=a._pursuitnet_dtype, device=a.device)
        if a.requires_grad or b.requires_grad:
            result.requires_grad = True
            result.val = Value(result.data, requires_grad=True)
            def _backward(grad_output):
                if a.requires_grad:
                    a.backward(grad_output)
                if b.requires_grad:
                    b.backward(-grad_output)
            result.val.grad_fn = _backward
    else:
        result = a.__class__(np.subtract(a.data, b), dtype=a._pursuitnet_dtype, device=a.device)
        if a.requires_grad:
            result.requires_grad = True
            result.val = Value(result.data, requires_grad=True)
            def _backward(grad_output):
                a.backward(grad_output)
            result.val.grad_fn = _backward
    return result

def sum(a, axis=None, keepdims=False):
    result_data = np.sum(a.data, axis=axis, keepdims=keepdims)
    result = a.__class__(result_data, dtype=a._pursuitnet_dtype, device=a.device)
    
    if a.requires_grad:
        result.requires_grad = True
        result.val = Value(result.data, requires_grad=True)
        
        def _backward(grad_output):
            # Ensure grad_output has the same shape as the original input
            if axis is not None:
                grad_shape = list(a.data.shape)
                if not keepdims:
                    if isinstance(axis, (list, tuple)):
                        for ax in sorted(axis, reverse=True):
                            grad_shape.insert(ax, 1)
                    else:
                        grad_shape.insert(axis, 1)
                grad_output = np.broadcast_to(grad_output.reshape(grad_shape), a.data.shape)
            else:
                # If axis is None, the result is a scalar, so we broadcast to the original shape
                grad_output = np.full(a.data.shape, grad_output)
            
            a.backward(grad_output)
        
        result.val.grad_fn = _backward
    
    return result

def max(a, axis=None, keepdims=False):
    result = a.__class__(np.max(a.data, axis=axis, keepdims=keepdims), dtype=a._pursuitnet_dtype, device=a.device)
    if a.requires_grad:
        result.requires_grad = True
        result.val = Value(result.data, requires_grad=True)
        def _backward(grad_output):
            mask = a.data == np.max(a.data, axis=axis, keepdims=True)
            a.backward(grad_output * mask)
        result.val.grad_fn = _backward
    return result

def min(a, axis=None, keepdims=False):
    result = a.__class__(np.min(a.data, axis=axis, keepdims=keepdims), dtype=a._pursuitnet_dtype, device=a.device)
    if a.requires_grad:
        result.requires_grad = True
        result.val = Value(result.data, requires_grad=True)
        def _backward(grad_output):
            mask = a.data == np.min(a.data, axis=axis, keepdims=True)
            a.backward(grad_output * mask)
        result.val.grad_fn = _backward
    return result

def mean(a, axis=None, keepdims=False):
    result = a.__class__(np.mean(a.data, axis=axis, keepdims=keepdims), dtype=a._pursuitnet_dtype, device=a.device)
    if a.requires_grad:
        result.requires_grad = True
        result.val = Value(result.data, requires_grad=True)
        def _backward(grad_output):
            shape = np.array(a.data.shape)
            if axis is not None:
                shape[axis] = 1
            a.backward(grad_output / np.prod(shape) * np.ones_like(a.data))
        result.val.grad_fn = _backward
    return result

def reshape(a, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    result = a.__class__(a.data.reshape(shape), dtype=a._pursuitnet_dtype, device=a.device)
    if a.requires_grad:
        result.requires_grad = True
        result.val = Value(result.data, requires_grad=True)
        def _backward(grad_output):
            a.backward(grad_output.reshape(a.data.shape))
        result.val.grad_fn = _backward
    return result

def transpose(a, *axes):
    if not axes:
        axes = None
    result = a.__class__(np.transpose(a.data, axes), dtype=a._pursuitnet_dtype, device=a.device)
    if a.requires_grad:
        result.requires_grad = True
        result.val = Value(result.data, requires_grad=True)
        def _backward(grad_output):
            a.backward(np.transpose(grad_output, np.argsort(axes) if axes else None))
        result.val.grad_fn = _backward
    return result

def zeros(cls, *shape, dtype, device, requires_grad):
    return cls(np.zeros(shape), dtype=dtype, device=device, requires_grad=requires_grad)

def ones(cls, *shape, dtype, device, requires_grad):
    return cls(np.ones(shape), dtype=dtype, device=device, requires_grad=requires_grad)

def random(cls, *shape, dtype, device, requires_grad):
    return cls(np.random.random(shape), dtype=dtype, device=device, requires_grad=requires_grad)

def tensor_repr(tensor):
    def format_element(x):
        if np.issubdtype(tensor._numpy_dtype, np.bool_) and x == True:
            return f" {str(x)}"
        elif np.issubdtype(tensor._numpy_dtype, np.floating):
            if abs(x) < 1e-4 and x != 0:
                return f"{x:.4e}"
            elif abs(x) >= 1e4:
                return f"{x:.4e}"
            elif x == int(x):
                return f"{int(x)}."
            else:
                return f"{x:.4f}".rstrip('0').rstrip('.') + ('0' * (4 - len(f"{x:.4f}".rstrip('0').rstrip('.').split('.')[-1])))
        elif np.issubdtype(tensor._numpy_dtype, np.complexfloating):
            return f"{x.real:.4f}{x.imag:+.4f}j"
        elif np.issubdtype(tensor._numpy_dtype, np.integer):
            return f"{x}"
        else:
            return str(x)

    data_str = np.array2string(
        tensor.data,
        separator=', ',
        formatter={'all': format_element},
        threshold=1000,
        edgeitems=3,
        max_line_width=75
    )
    
    data_str = data_str.replace('\n ', '\n        ')
    
    if data_str.endswith('\t'):
        data_str = data_str[:-1]
    
    if len(tensor.data.shape) == 1:
        data_str = data_str.replace('\t', '')

    dtype_str = f"pursuitnet.{tensor._pursuitnet_dtype.__name__}"

    if tensor._pursuitnet_dtype.numpy_dtype == np.float32 or tensor._pursuitnet_dtype.numpy_dtype == np.bool_:
        return f"Tensor({data_str})"
    else:
        return f"Tensor({data_str}, dtype={dtype_str})"
