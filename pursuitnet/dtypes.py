import numpy as np

# Floating-point types
float32 = np.float32
float64 = np.float64
float16 = np.float16

# Complex types
complex64 = np.complex64
complex128 = np.complex128

# Integer types
uint8 = np.uint8
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

# Boolean type
bool = np.bool_

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
quint8 = np.dtype([('quint8', np.uint8)])
qint8 = np.dtype([('qint8', np.int8)])
qint32 = np.dtype([('qint32', np.int32)])
quint4x2 = np.dtype([('quint4x2', np.uint8)])  # This is a placeholder

# 8-bit floating point types (these don't have NumPy equivalents, so we'll create custom dtypes)
float8_e4m3fn = np.dtype([('float8_e4m3fn', np.uint8)])
float8_e5m2 = np.dtype([('float8_e5m2', np.uint8)])

# Mapping of NumPy dtypes to pursuitnet dtypes
dtype_map = {
    float32: 'pursuitnet.float32',
    float64: 'pursuitnet.float64',
    float16: 'pursuitnet.float16',
    complex64: 'pursuitnet.complex64',
    complex128: 'pursuitnet.complex128',
    uint8: 'pursuitnet.uint8',
    int8: 'pursuitnet.int8',
    int16: 'pursuitnet.int16',
    int32: 'pursuitnet.int32',
    int64: 'pursuitnet.int64',
    uint16: 'pursuitnet.uint16',
    uint32: 'pursuitnet.uint32',
    uint64: 'pursuitnet.uint64',
    bool: 'pursuitnet.bool',
    quint8: 'pursuitnet.quint8',
    qint8: 'pursuitnet.qint8',
    qint32: 'pursuitnet.qint32',
    quint4x2: 'pursuitnet.quint4x2',
    float8_e4m3fn: 'pursuitnet.float8_e4m3fn',
    float8_e5m2: 'pursuitnet.float8_e5m2',
}

def get_pursuitnet_dtype(dtype):
    return dtype_map.get(dtype, str(dtype))
