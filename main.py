import matplotlib.pyplot as plt

from perceptron.gui import run_gui
from perceptron.utils import make_log_file, get_logger
from perceptron.runner import run_model


if __name__ == "__main__":
    make_log_file()
    
    logger = get_logger(__name__)
    logger.info("Starting...")

    plt.style.use("ggplot")

    run_gui(run_model)

    logger.info("Finished.")
