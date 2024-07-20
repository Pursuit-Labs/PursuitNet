try:
    import cupy as np
    HAS_CUPY = True
except ImportError:
    import numpy as np
    HAS_CUPY = False

from ..autograd import functions

def add(a, b):
    if isinstance(b, a.__class__):
        result = a.__class__(np.add(a.data, b.data), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = functions.Add.apply(a.val, b.val) if (a.val is not None or b.val is not None) else None
    else:
        result = a.__class__(np.add(a.data, b), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = functions.Add.apply(a.val, b) if a.val is not None else None
    return result

def mul(a, b):
    if isinstance(b, a.__class__):
        result = a.__class__(np.multiply(a.data, b.data), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = functions.Mul.apply(a.val, b.val) if (a.val is not None or b.val is not None) else None
    else:
        result = a.__class__(np.multiply(a.data, b), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = functions.Mul.apply(a.val, b) if a.val is not None else None
    return result

def matmul(a, b):
    if not isinstance(b, a.__class__):
        raise TypeError("Matrix multiplication is only supported between Tensors")
    result = a.__class__(np.matmul(a.data, b.data), dtype=a._pursuitnet_dtype, device=a.device)
    result.val = functions.MatMul.apply(a.val, b.val) if (a.val is not None or b.val is not None) else None
    return result

def div(a, b):
    if isinstance(b, a.__class__):
        result = a.__class__(np.divide(a.data, b.data), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = a.val / b.val if (a.val is not None and b.val is not None) else None
    else:
        result = a.__class__(np.divide(a.data, b), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = a.val / b if a.val is not None else None
    return result

def sub(a, b):
    if isinstance(b, a.__class__):
        result = a.__class__(np.subtract(a.data, b.data), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = a.val - b.val if (a.val is not None and b.val is not None) else None
    else:
        result = a.__class__(np.subtract(a.data, b), dtype=a._pursuitnet_dtype, device=a.device)
        result.val = a.val - b if a.val is not None else None
    return result

def sum(a):
    result = a.__class__(np.sum(a.data), dtype=a._pursuitnet_dtype, device=a.device, requires_grad=a.requires_grad)
    if a.requires_grad:
        def _backward(grad_output):
            a.backward(np.full_like(a.data, grad_output))
        result.val.grad_fn = _backward
    return result

def max(a, axis=None, keepdims=False):
    return a.__class__(np.max(a.data, axis=axis, keepdims=keepdims), dtype=a._pursuitnet_dtype, device=a.device)

def min(a, axis=None, keepdims=False):
    return a.__class__(np.min(a.data, axis=axis, keepdims=keepdims), dtype=a._pursuitnet_dtype, device=a.device)

def mean(a, axis=None, keepdims=False):
    return a.__class__(np.mean(a.data, axis=axis, keepdims=keepdims), dtype=a._pursuitnet_dtype, device=a.device)

def reshape(a, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return a.__class__(a.data.reshape(shape), dtype=a._pursuitnet_dtype, device=a.device)

def transpose(a, *axes):
    if not axes:
        axes = None
    return a.__class__(np.transpose(a.data, axes), dtype=a._pursuitnet_dtype, device=a.device)

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