from pathlib import Path

from perceptron.data_handlers import *
from perceptron.model import MLP
from perceptron.model.activations import ACTIVATION_FUNCTIONS
from perceptron.utils import plot_mses


def run_model(h_layers, mse, eta, dataset, activation, bias, epochs):
    handler = None
    if dataset == "Iris":
        handler = Iris(Path("data/iris.csv"))
    elif dataset == "Penguins":
        handler = Penguins(Path("data/penguins.csv"))
    elif dataset == "MNIST":
        handler = MNIST(Path("data/mnist_train.csv"), Path("data/mnist_test.csv"))

    model = MLP(
        *handler.partition_data(),
        h_layers,
        bias=bias,
        mse_threshold=mse,
        eta=eta,
        activation=ACTIVATION_FUNCTIONS[activation],
        epochs=epochs,
    )

    mses, train_acc = model.train()
    plot_mses(mses, dataset)
    print(f"{dataset} training accuracy: {train_acc}")

    conf_matrix, test_acc = model.test()
    print(f"{dataset} testing accuracy: {test_acc}\nConfusion matrix:\n{conf_matrix}")

    return test_acc
