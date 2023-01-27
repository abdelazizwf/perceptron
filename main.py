from pathlib import Path
import matplotlib.pyplot as plt

from src.gui import run_gui
from src.data_handlers import Iris, MNIST, Penguins
from src.model import MLP, ACTIVATION_FUNCTIONS
from src.utils import get_logger, make_log_file, plot_mses, LOG_PATH


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

if __name__ == "__main__":
    make_log_file()
    
    logger = get_logger(__name__)
    logger.info("Starting...")

    plt.style.use("ggplot")

    run_gui(run_model)

    logger.info("Finished.")
