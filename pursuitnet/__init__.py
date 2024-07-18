# pursuitnet/__init__.py
__version__ = '0.1.0'

from .random import Random
from .nn import NN
from .optim import Optim

random = Random()
nn = NN()
optim = Optim()

