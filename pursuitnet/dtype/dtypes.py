# pursuitnet/dtype/dtypes.py

import numpy as np

class dtype:
    def __init__(self, numpy_dtype, name):
        self.numpy_dtype = numpy_dtype
        self.__name__ = name

    def __repr__(self):
        return f"pursuitnet.{self.__name__}"

    def __call__(self):
        return self.numpy_dtype

# Floating-point types
float32 = dtype(np.float32, 'float32')
float64 = dtype(np.float64, 'float64')
float16 = dtype(np.float16, 'float16')

# Complex types
complex64 = dtype(np.complex64, 'complex64')
complex128 = dtype(np.complex128, 'complex128')

# Integer types
uint8 = dtype(np.uint8, 'uint8')
int8 = dtype(np.int8, 'int8')
int16 = dtype(np.int16, 'int16')
int32 = dtype(np.int32, 'int32')
int64 = dtype(np.int64, 'int64')
uint16 = dtype(np.uint16, 'uint16')
uint32 = dtype(np.uint32, 'uint32')
uint64 = dtype(np.uint64, 'uint64')

# Boolean type
bool = dtype(np.bool_, 'bool')  # Changed from np.bool to np.bool_

# Aliases
float = float32
double = float64
half = float16
short = int16
int = int32
long = int64
cfloat = complex64
cdouble = complex128

# Quantized types (these don't have direct NumPy equivalents, so we'll create custom dtypes)
quint8 = dtype(np.dtype([('quint8', np.uint8)]), 'quint8')
qint8 = dtype(np.dtype([('qint8', np.int8)]), 'qint8')
qint32 = dtype(np.dtype([('qint32', np.int32)]), 'qint32')
quint4x2 = dtype(np.dtype([('quint4x2', np.uint8)]), 'quint4x2')  # This is a placeholder

# 8-bit floating point types (these don't have NumPy equivalents, so we'll create custom dtypes)
float8_e4m3fn = dtype(np.dtype([('float8_e4m3fn', np.uint8)]), 'float8_e4m3fn')
float8_e5m2 = dtype(np.dtype([('float8_e5m2', np.uint8)]), 'float8_e5m2')

# Mapping of PursuitNet dtypes to their string representations
dtype_map = {dtype: str(dtype) for dtype in [
    float32, float64, float16, complex64, complex128,
    uint8, int8, int16, int32, int64, uint16, uint32, uint64,
    bool, quint8, qint8, qint32, quint4x2, float8_e4m3fn, float8_e5m2
]}

def get_pursuitnet_dtype(dtype):
    return dtype_map.get(dtype, str(dtype))
