import tkinter as tk
from tkinter import *


root = Tk()
root.title('Task 3')
root.geometry("500x200")

myLabelT1 = Label(root, text = "Enter number of hidden layers: ")
myLabelT1.grid(column = 0, row = 2, sticky=W)
T1 = Text(root, height = 1, width = 10)
T1.grid(column = 1, row = 2)

myLabelT2 = Label(root, text = "Enter number of neurons in each layer: ")
myLabelT2.grid(column = 0, row = 3, sticky=W)
T2 = Text(root, height = 1, width = 10)
T2.grid(column = 1, row = 3)

myLabelT3 = Label(root, text = "Enter learning rate: ")
myLabelT3.grid(column = 0, row = 5, sticky=W)
T3 = Text(root, height = 1, width = 10)
T3.grid(column = 1, row = 5)

myLabelT4 = Label(root, text = "Enter number of epochs: ")
myLabelT4.grid(column = 0, row = 6, sticky=W)
T4 = Text(root, height = 1, width = 10)
T4.grid(column = 1, row = 6)

c1_v=tk.IntVar(root)
c1 = tk.Checkbutton(root, text='Bias', onvalue=1, variable= c1_v)
c1.grid(column = 0, row = 7)

frame = Frame(root) 
frame.grid(column = 1, row = 7)

v = StringVar(root, "1")

functions = {"Sigmoid" : "1", "Hyperbolic Tangent" : "2"}

for (text, value) in functions.items():
    Radiobutton(frame, text = text, variable = v,
                value = value).pack(side = LEFT, ipady = 5)
    
    
def Listing():
    
 
myButton = Button( root, text = "run", height= 2,width = 10, command = Listing )
myButton.grid(column = 1, row = 10)

root.mainloop()
