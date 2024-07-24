# pursuitnet/__init__.py
__version__ = '0.1.0'

# Importing main components of the library
from .rand import Rand
from . import optim

# Importing dtype, ops, device, and autograd modules
from .dtype import *
from .ops import ops
from .device import device_utils
from .autograd import grad_utils, Value, operations

from .nn import *

# Initializing main components
rand = Rand()
dtype = dtype(None, 'None')

# Defer the import of Tensor to avoid circular dependency
from .tensor import Tensor

# Initialize tensor after the import
tensor = Tensor(None)

