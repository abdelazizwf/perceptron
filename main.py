import matplotlib.pyplot as plt

from GUI import gui
from data_handlers import Iris, MNIST, Penguins
from model import MLP
from util import ACTIVATION_FUNCTIONS, get_logger, plot_mses

def run(h_layers, mse, eta, dataset, activation, bias, epochs):
    handler = None
    if dataset == "Iris":
        handler = Iris("data/iris.csv")
    elif dataset == "Penguins":
        handler = Penguins("data/penguins.csv")
    elif dataset == "MNIST":
        handler = MNIST("data/mnist_train.csv", "data/mnist_test.csv")

    model = MLP(
        *handler.partition_data(),
        h_layers,
        bias=bias,
        mse_threshold=mse,
        eta=eta,
        activation=ACTIVATION_FUNCTIONS[activation],
        epochs=epochs
    )

    mses, train_acc = model.train()
    plot_mses(mses)
    print(f"{dataset} training accuracy: {train_acc}")

    test_acc = model.test()
    print(f"{dataset} testing accuracy: {test_acc}")

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Starting...")

    plt.style.use("ggplot")

    gui(run)

    logger.info("Finished.")
