import numpy as np

def relu(x):
    return np.maximum(0, x)

def view(x, shape):
    return x.reshape(shape)
