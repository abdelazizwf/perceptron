import os
import tkinter as tk
from tkinter.scrolledtext import ScrolledText


class Text:

    def __init__(self, root, text, row):
        tk.Label(root, text=text).grid(column=0, row=row, padx=10, pady=10, sticky=tk.W)

        self.widget = tk.Text(root, height=1, width=15)
        self.widget.grid(column=1, row=row)

    def get(self):
        return self.widget.get(1.0, "end-1c")


class OptionMenu:

    def __init__(self, root, text, row, values):
        tk.Label(root, text=text).grid(column=0, row=row, padx=10, pady=10, sticky=tk.W)
        
        self.var = tk.StringVar()
        self.var.set(values[0])
        tk.OptionMenu(root, self.var, *values).grid(column=1, row=row)

    def get(self):
        return self.var.get()


class LoggingWidget:

    def __init__(self, root, column, rowspan, path):
        self.widget = ScrolledText(root, state='disabled', wrap=tk.WORD)
        self.widget.grid(rowspan=rowspan, column=column, row=0, padx=10, pady=10)

        self.path = path

    def update(self, acc=-1):
        logs = []

        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                logs = "\n".join(f.readlines())

        self.widget.configure(state='normal')
        self.widget.delete('1.0', tk.END)
        self.widget.insert(tk.INSERT, logs)
        if acc > -1:
            self.widget.insert(tk.INSERT, f"\n\nAccuracy: {acc}\n\n")
        self.widget.configure(state='disabled')
        self.widget.yview(tk.END)

