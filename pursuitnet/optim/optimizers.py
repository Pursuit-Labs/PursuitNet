# optim/optimizers.py

import numpy as np

class Optim:
    def __init__(self):
        print("Optim initialized")

    def __del__(self):
        print("Optim deleted")

    def print_random_number(self):
        print(f"Random number from pursuitnet.optim: {np.random.rand()}")
