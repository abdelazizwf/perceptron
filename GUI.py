import tkinter as tk

from util import ACTIVATION_FUNCTIONS
from data_handlers import DATASETS



def gui(runner):
    root = tk.Tk()
    root.title('Task 3')
    root.geometry("500x200")

    tk.Label(root, text="Enter hidden layers: ").grid(column=0, row=2, sticky=tk.W)
    hlayers_text = tk.Text(root, height=1, width=10)
    hlayers_text.grid(column=1, row=2)

    tk.Label(root, text="Enter MSE threshold: ").grid(column=0, row=3, sticky=tk.W)
    mse_text = tk.Text(root, height=1, width=10)
    mse_text.grid(column=1, row=3)

    tk.Label(root, text="Enter learning rate: ").grid(column=0, row=5, sticky=tk.W)
    eta_text = tk.Text(root, height=1, width=10)
    eta_text.grid(column=1, row=5)

    tk.Label(root, text="Enter number of epochs: ").grid(column=0, row=6, sticky=tk.W)
    epochs_text = tk.Text(root, height=1, width=10)
    epochs_text.grid(column=1, row=6)

    tk.Label(root, text="Select dataset: ").grid(column=0, row=7, sticky=tk.W)
    dataset_v = tk.StringVar()  
    dataset_v.set("Penguins")
    tk.OptionMenu(root, dataset_v, *DATASETS.keys()).grid(column=1, row=7)

    tk.Label(root, text="Select activation function: ").grid(column=0, row=8, sticky=tk.W)
    af_v = tk.StringVar()
    af_v.set("Sigmoid")
    tk.OptionMenu(root, af_v, *ACTIVATION_FUNCTIONS.keys()).grid(column=1, row=8)

    bias_var = tk.IntVar(root)
    tk.Checkbutton(root, text='Bias', onvalue=1, offvalue=0, variable=bias_var).grid(column=0, row=9)


    def submit():
        h_layers = [int(num) for num in hlayers_text.get(1.0, "end-1c").split()]
        mse = float(mse_text.get(1.0, "end-1c"))
        eta = float(eta_text.get(1.0, "end-1c"))
        dataset = dataset_v.get()
        af = af_v.get()
        bias = int(bias_var.get())
        print(h_layers, mse, eta, dataset, af, bias)
        runner(h_layers, mse, eta, dataset, af, bias)


    tk.Button(root, text="Run", height=2, width=10, command=submit).grid(column=1, row=10)

    root.mainloop()
