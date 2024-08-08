# pursuitnet/__init__.py

__version__ = '0.1.0'

# Import Tensor directly
from .tensor import Tensor

# Import main components of the library
from .autograd import grad_utils, Value, operations, parameter
from .nn import *
from .optim import *
from .dtype import *

from .rand import Rand
from .optim import optimizers
from .device import device_utils
from .ops import *

# Initialize rand and dtype
rand = Rand()