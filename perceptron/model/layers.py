import numpy as np

from perceptron.model.activations import sigmoid


class Layer:

    def __init__(self, nodes_num, previous_num, bias=1, activation=sigmoid, eta=0.4):
        # Initialize a matrix N x M, where N is the number of nodes in the current layer,
        # and M is the number of nodes in the previous layer. Each row holds all the weights
        # connected to a node, and each column correspond to the weight of a node in the previous
        # layer that is connected to nodes in this layer. The bias is treated as a weight for each node.
        self.weights = np.random.standard_normal((nodes_num, previous_num + bias))

        self.activation = activation
        self.bias = bias
        self.eta = eta

        self.deltas = None # Hold the delta values calculated in the backward pass
        self.inputs = None # Hold the inputs recieved in the first forward pass
        self.nets = None # Hold the net values calculated in the first forward pass

    def get_outputs(self, inputs):
        # if bias is enabled, add 1 as a bias input
        if self.bias == 1:
            inputs = np.append(inputs, 1)

        # Reshape the inputs from 1D array to a 2D matrix
        self.inputs = np.array([inputs])

        # Calculate the net values using X W^T
        # (X W^T is different than W^T X since X is represented as a 1 x M matrix
        # instead of M x 1 matrix)
        self.nets = self.inputs @ self.weights.T

        # Return the activations of the net values as a 1D array
        return self.activation.function(self.nets).flatten()

    def calculate_deltas(self, back_deltas, back_weights):
        # Calculate the sums using matrix multiplication
        sums = back_deltas @ back_weights

        # If bias is enabled, add a bias output to match the dimensions of the sums matrix
        if self.bias == 1:
            self.nets = np.array([np.append(self.nets, 0)])

        # Calculate the delta values
        self.deltas = self.activation.derivative(self.nets) * sums

        # Because we calculate a delta for the bias weights in the current layer, that delta
        # must be removed since the previous layer doesn't provide outputs to the bias unit.
        forward_deltas = self.deltas
        if self.bias == 1:
            forward_deltas = np.array([self.deltas[0][:-1]])

        # Return both the delta values and weights so they can be used in the previuos layer
        return forward_deltas, self.weights

    def update_weights(self):
        # Convert the deltas and inputs matrices into an array
        self.deltas = self.deltas.flatten()
        self.inputs = self.inputs.flatten()

        # Update each row of weights according to the delta rule
        for i, row in enumerate(self.weights):
            row += self.eta * self.deltas[i] * self.inputs


class OutputLayer(Layer):

    def calculate_deltas(self, costs):
        # Output layers calculate the delta values based on the cost of the outputs
        self.deltas = costs * self.activation.derivative(self.nets)
        return self.deltas, self.weights

