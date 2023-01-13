import tkinter as tk
from tkinter import messagebox

from src.data_handlers import DATASETS
from src.utils import ACTIVATION_FUNCTIONS, get_logger
from src.gui.widgets import *


def h_layers_info():
    messagebox.showinfo(
        "Hidden layers",
        "Hidden layers are written as integers seperated by spaces. For example, the input '2 8 16' " +
        "will create a network with 3 hidden layers, the first has 2 nodes, the second has 8, and the " +
        "the third has 16."
    )

def run_gui(runner):
    root = tk.Tk()
    root.resizable(width=False, height=False)
    root.title('Task 3')

    dataset_inp = OptionMenu(root, "Select dataset: ", 0, DATASETS)

    hidden_layers_inp = Text(root, "Enter hidden layers: ", 1)
    mse_inp = Text(root, "Enter MSE threshold: ", 2)
    eta_inp = Text(root, "Enter learning rate: ", 3)
    epochs_inp = Text(root, "Enter number of epochs: ", 4)

    activation_inp = OptionMenu(root, "Select activation function: ", 5, list(ACTIVATION_FUNCTIONS.keys()))

    bias_var = tk.IntVar(root)
    tk.Checkbutton(root, text=" Bias", onvalue=1, offvalue=0, variable=bias_var).grid(column=0, row=6)

    logging_widget = LoggingWidget(root, 3, 7, "tmp/run.log")

    tk.Button(root, text="?", command=h_layers_info).grid(column=2, row=1)

    working = tk.Label(root, text="Running.....")


    def submit():
        try:
            h_layers_val = hidden_layers_inp.get()
            assert h_layers_val != ''
            assert all(num.isdigit() for num in h_layers_val.split())
            h_layers = [int(num) for num in h_layers_val.split()]
        except:
            messagebox.showerror(
                "Error",
                "Please enter a correct value for the hidden layers. " +
                "Consult the info button (?) for more information."
            )
            return

        mse = float(mse_inp.get())
        eta = float(eta_inp.get())
        epochs = int(epochs_inp.get())
        dataset = dataset_inp.get()
        activation = activation_inp.get()
        bias = int(bias_var.get())

        logger = get_logger(__name__)
        logger.info(f"GUI submitted for {dataset} with {h_layers} hidden layers, bias: {bias}, epochs: {epochs}, " +
                    f"activation: {activation}, learning rate: {eta}, MSE threshold: {mse}")

        working.grid(column=3, row=7, sticky=tk.W, padx=10, pady=10)
        root.update()

        acc = runner(h_layers, mse, eta, dataset, activation, bias, epochs)

        working.grid_forget()
        root.update()

        logging_widget.update(acc)


    tk.Button(root, text="Run", height=2, width=10, command=submit).grid(column=1, row=7, pady=10)

    logging_widget.update()
    root.mainloop()
