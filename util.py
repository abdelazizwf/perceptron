from collections import namedtuple
import numpy as np

ActivationFunction = namedtuple("ActivationFunction", ["name", "function", "derivative"])

sig_func = lambda x: 1 / (1 + np.exp(-x))

sigmoid = ActivationFunction(
    name="Sigmoid",
    function=sig_func,
    derivative=lambda x: sig_func(x) * (1 - sig_func(x))
)

# https://www.wolframalpha.com/input?key=&i=derivative+%281+-+e%5E-x%29+%2F+%281+%2B+e%5E-x%29
hyper_tan = ActivationFunction(
    name="Hyperbolic Tangent",
    function=lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x)),
    derivative=lambda x: (2 * np.exp(x)) / ((np.exp(x) + 1) ** 2)
)

ACTIVATION_FUNCTIONS = [
    sigmoid,
    hyper_tan,
]
