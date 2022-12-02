from collections import namedtuple
import logging

import matplotlib.pyplot as plt
import numpy as np


# A data class to group activation functions and their derivatives
ActivationFunction = namedtuple("ActivationFunction", ["name", "function", "derivative"])

# All activation functions are from lecture 5, page 61.

sig_func = lambda x: 1 / (1 + np.exp(-x))
sigmoid = ActivationFunction(
    name="Sigmoid",
    function=sig_func,
    derivative=lambda x: sig_func(x) * (1 - sig_func(x))
)

htan_func = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
hyper_tan = ActivationFunction(
    name="Hyperbolic Tangent",
    function=htan_func,
    derivative=lambda x: 1 - (htan_func(x) ** 2)
)

relu = ActivationFunction(
    name="Rectified Linear Unit",
    function=lambda x: np.maximum(x, 0),
    derivative=lambda x: np.where(x > 0, 1, 0)
)

# Global list to be used in GUI
ACTIVATION_FUNCTIONS = {
    "Sigmoid": sigmoid,
    "Hyperbolic Tangent": hyper_tan,
    "Rectified Linear Unit": relu
}

def get_logger(name):
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    date_fmt = "%H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=date_fmt,
        filename='run.log',
        filemode='w'
    )
    return logging.getLogger(name)

def plot_mses(mses):
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.plot(range(len(mses)), mses)

    plt.show()

class ConfusionMatrix:

    def __init__(self, dimension):
        self.dimension = dimension
        self.matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]

    def add(self, actual, predicted):
        self.matrix[actual][predicted] += 1

    def __repr__(self):
        slots = len(str(max(map(max, self.matrix))))
        aloc = lambda x: str(x) + (" " * (slots - len(str(x))))
        result = ""
        result += "  " + " ".join(map(aloc, range(self.dimension)))

        for i, vals in enumerate(self.matrix):
            result += "\n" + str(i) + " " + " ".join(map(aloc, vals))

        return result
