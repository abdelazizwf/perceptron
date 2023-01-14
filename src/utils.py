import logging
from pathlib import Path

import matplotlib.pyplot as plt


def get_logger(name):
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    date_fmt = "%H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=date_fmt,
        filename=Path('tmp/run.log'),
        filemode='w'
    )
    return logging.getLogger(name)

def plot_mses(mses, name):
    plt.title(name + " training")
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
