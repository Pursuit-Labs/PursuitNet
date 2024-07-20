# pursuitnet/__init__.py
__version__ = '0.1.0'

# Importing main components of the library
from .random import Random
from .nn import NN
from .optim import Optim

# Importing dtype, ops, device, and autograd modules
from .dtype import *
from .ops import ops
from .device import device_utils
from .autograd import grad_utils, functions, Value

# Initializing main components
random = Random()
nn = NN()
optim = Optim()
dtype = dtype(None, 'None')

# Defer the import of Tensor to avoid circular dependency
from .tensor import Tensor

# Initialize tensor after the import
tensor = Tensor(None)

