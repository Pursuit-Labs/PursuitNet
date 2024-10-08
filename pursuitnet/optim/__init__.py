# optim/__init__.py
from .optimizers import Adam
import numpy as np

def print_random_number():
    rand = np.random.rand()
    print(f"Random number from pursuitnet.optim: {rand}")
    return rand