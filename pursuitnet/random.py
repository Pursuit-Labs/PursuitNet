# random.py

import numpy as np
import time

class Random:
    def __init__(self):
        self.seed(int(time.time()))
        print("Random initialized")

    def __del__(self):
        print("Random deleted")

    def seed(self, seed_value):
        np.random.seed(seed_value)

    def print_random_number(self):
        print(f"Random number from pursuitnet.random: {np.random.rand()}")