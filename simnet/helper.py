import math
import numpy as np
import imageio
import glob
from os import path

def suffled_digits(pathname):
    digits = []
    for d in range(10):
        for file in glob.glob(path.join(pathname, str(d), "*.png")):
            digits.append((d, file))
    np.random.shuffle(digits)
    return digits
    

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_arr(arr):
    return np.asarray([sigmoid(x) for x in arr])
