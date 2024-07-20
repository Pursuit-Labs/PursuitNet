try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def to_device(tensor, device):
    if device == tensor.device:
        return tensor
    else:
        if device == 'cpu' and HAS_CUPY:
            new_tensor = tensor.__class__(tensor.data.get(), dtype=tensor._pursuitnet_dtype, device=device)
        elif device == 'gpu' and HAS_CUPY:
            new_tensor = tensor.__class__(cupy.array(tensor.data), dtype=tensor._pursuitnet_dtype, device=device)
        else:
            new_tensor = tensor.__class__(tensor.data, dtype=tensor._pursuitnet_dtype, device=device)
        new_tensor.val = tensor.val
        return new_tensor