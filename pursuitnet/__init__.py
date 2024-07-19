# pursuitnet/__init__.py
__version__ = '0.1.0'

from .random import Random
from .nn import NN
from .optim import Optim
from .tensor import Tensor
from .dtypes import *

random = Random()
nn = NN()
optim = Optim()
tensor = Tensor(None)
