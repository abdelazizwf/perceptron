# Perceptron
This repository is adapted from an assignment in a university Artificial Neural Networks class. The goal was to build a Multilayer Perceptron in python using NumPy and Pandas, with a GUI interface using Tkinter, and use it to classify data in the penguins dataset (`data/penguins.csv`) and optionally the MNIST dataset.

I decided to use that code to apply several new topics I learned in Git, GitHub, Python, AI, and Neural Networks, as well as writing better code and refactoring old code. All the new work starts from the `rebirth` tag.

## Usage

### Linux:

First clone the repository, and navigate to the project's directory:
```bash
git clone https://github.com/abdelazizwf/perceptron.git && cd perceptron
```
Then, use `venv` to create a virtual environment:
```bash
python3 -m venv --prompt "perceptron-venv" .venv
```
Activate the new environment:
```bash
source .venv/bin/activate && python -m ensurepip
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Run the project using `main.py`:
```bash
python main.py
```

