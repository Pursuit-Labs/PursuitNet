# rand.py

import numpy as np
import time

class Rand:
    def __init__(self):
        self.seed(int(time.time()))

    def __del__(self):
        pass

    def seed(self, seed_value):
        np.random.seed(seed_value)

    def print_random_number(self):
        rand = np.random.rand()
        print(f"Random number from pursuitnet.random: {rand}")
        return rand