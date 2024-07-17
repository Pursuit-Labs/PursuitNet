# nn/modules.py
import numpy as np

class NN:
    def __init__(self):
        print("NN initialized")

    def __del__(self):
        print("NN deleted")

    def print_random_number(self):
        print(f"Random number from pursuitnet.nn: {np.random.rand()}")

