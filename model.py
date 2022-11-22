import numpy as np
from util import sigmoid

class Layer:

    def __init__(self, nodes_num, previous_num, bias=1, activation=sigmoid, eta=0.4):
        # Initialize a matrix N x M, where N is the number of nodes in the current layer,
        # and M is the number of nodes in the previous layer. Each row holds all the weights
        # connected to a node, and each column correspond to the weight of a node in the previous
        # layer that is connected to nodes in this layer. The bias is treated as a weight for each node.
        self.weights = np.random.standard_normal((nodes_num, previous_num + bias))

        self.activation = activation   
        self.eta = eta
        self.bias = bias

        self.deltas = None # Hold the delta values calculated in the backward pass
        self.inputs = None # Hold the inputs recieved in the first forward pass
        self.nets = None # Hold the net values calculated in the first forward pass

    def first_forward(self, inputs):
        if self.bias == 1:
            inputs = np.append(inputs, 1)

        self.inputs = np.array([inputs])

        # Calculate the net values usin X W^T
        # (X W^T is different than W^T X since X is represented as a 1 x M matrix
        # instead of M x 1 matrix)
        self.nets = self.inputs @ self.weights.T

        # Return the activations of the net values
        return self.activation.function(self.nets).flatten()

    def backward(self, back_deltas, back_weights):
        # Calculate the sums using matrix multiplication
        sums = back_deltas @ back_weights

        # If bias is enabled, add a bias output to match the dimensions of both matrices
        if self.bias == 1:
            self.nets = np.array([np.append(self.nets, 0)])

        # Calculate the delta values
        self.deltas = self.activation.derivative(self.nets) * sums

        # Return both the delta values and weights so they can be used in the previuos layer
        forward_deltas = self.deltas
        if self.bias == 1:
            forward_deltas = np.array([self.deltas[0][:-1]])

        return forward_deltas, self.weights

    def second_forward(self):
        # Convert the deltas and inputs matrices into an array
        self.deltas = self.deltas.flatten()
        self.inputs = self.inputs.flatten()

        # Update each row of weights according to the delta rule
        for i, row in enumerate(self.weights):
            row += self.eta * self.deltas[i] * self.inputs


class OutputLayer(Layer):

    def backward(self, costs):
        # Output layers calculate the delta values based on the cost of the outputs
        self.deltas = costs * self.activation.derivative(self.nets)
        return self.deltas, self.weights


class Model:
    
    def __init__(self, x_train, y_train, x_test, y_test, hidden_layers, bias=1, activation=sigmoid, eta=0.4, epochs=1000, mse_threshold=0.05):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        self.num_inputs = len(x_train.columns)
        self.num_outputs = len(y_train.unique())
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.bias = bias

        self.layers = []
        prev = self.num_inputs
        for num in hidden_layers:
            self.layers.append(
                Layer(num, prev, bias, activation, eta)
            )
            prev = num

        self.layers.append(
            OutputLayer(self.num_outputs, prev, bias, activation, eta)
        )

        self.accuracy = 0

    def train(self):
        for _ in range(self.epochs):
            mse = 0

            for inputs, target in zip(self.x_train.values, self.y_train.values):

                ys = inputs
                for layer in self.layers:
                    ys = layer.first_forward(ys)

                target_vector = np.array(
                    [(1 if i == target else 0) for i in range(self.num_outputs)]
                )

                total_cost = np.sum((ys - target_vector) ** 2)
                mse += total_cost ** 2

                costs = target_vector - ys

                deltas, weights = self.layers[-1].backward(costs)
                for layer in reversed(self.layers[:-1]):
                    deltas, weights = layer.backward(deltas, weights)

                for layer in self.layers:
                    layer.second_forward()

            mse *= 1 / len(self.y_train)

            if mse < self.mse_threshold:
                break

    def test(self):
        correct = 0

        for inputs, target in zip(self.x_test.values, self.y_test.values):

            ys = inputs
            for layer in self.layers:
                ys = layer.first_forward(ys)

            y = ys.argmax()

            correct += 1 if y == target else 0

        self.accuracy = (correct / len(self.y_test)) * 100

        return self.accuracy
