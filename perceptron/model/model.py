import numpy as np

from perceptron.utils import ConfusionMatrix, get_logger
from perceptron.model.activations import sigmoid
from perceptron.model.layers import Layer, OutputLayer


class MLP:
    
    def __init__(self, x_train, y_train, x_test, y_test, hidden_layers, bias=1, activation=sigmoid, eta=0.4, epochs=400, mse_threshold=0.05):
        # Store training and testing data
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        # The number of input nodes is the number of features (columns) in the training data
        self.num_inputs = len(x_train.columns)
        # The number of output nodes is the number of target unique values of the training data 
        self.num_outputs = len(y_train.unique())
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.bias = bias

        # Create the hidden layers of the network
        self.layers = []
        prev = self.num_inputs
        for num in hidden_layers:
            self.layers.append(
                Layer(num, prev, bias, activation, eta)
            )
            prev = num

        # Create the output layer of the network
        self.layers.append(
            OutputLayer(self.num_outputs, prev, bias, activation, eta)
        )

        self.test_accuracy = -1
        self.train_accuracy = -1
        self.confusion_matrix = ConfusionMatrix(self.num_outputs)

        self.logger = get_logger(__name__ + "." + self.__class__.__name__)

        self.logger.info(f"MLP created with {hidden_layers} hidden layers, bias: {bias}, epochs: {epochs}, " +
                         f"activation: {activation.name}, learning rate: {eta}, MSE threshold: {mse_threshold}")

    def train(self):
        mses = []
        correct = 0

        for i in range(self.epochs):
            mse = 0

            for inputs, target in zip(self.x_train.values, self.y_train.values):

                # First forward pass to calculate the outputs
                ys = inputs
                for layer in self.layers:
                    ys = layer.get_outputs(ys)

                y = ys.argmax()
                correct += 1 if y == target else 0

                # Calculate the target array based on the value of the target. For example, if the target
                # is 2 out of a total of 6 classes (0-based), the target vector will be [0, 0, 1, 0, 0, 0]
                target_array = np.array(
                    [(1 if i == target else 0) for i in range(self.num_outputs)]
                )

                # Calculate the total cost to update the MSE
                total_cost = np.sum((ys - target_array) ** 2)
                mse += total_cost ** 2

                # Calculate the cost array for each output node (i.e: t - y for every node)
                costs = target_array - ys

                # Backward pass. Starts with the output layer, then goes backwards through the hidden layers
                deltas, weights = self.layers[-1].calculate_deltas(costs)
                for layer in reversed(self.layers[:-1]):
                    deltas, weights = layer.calculate_deltas(deltas, weights)

                # Second forward pass to update the weights
                for layer in self.layers:
                    layer.update_weights()

            # Calculate the MSE for the whole epoch and finish training if it's below the threshold
            mse *= 1 / len(self.y_train)

            mses.append(mse)

            self.logger.info(f"MSE at epoch {i + 1} is {mse}")

            if mse < self.mse_threshold:
                break

        self.train_accuracy = (correct / (len(mses) * len(self.y_train))) * 100

        self.logger.info(f"Finished training with accuracy {self.train_accuracy}")

        return mses, round(self.train_accuracy, 3)

    def test(self):
        correct = 0

        for inputs, target in zip(self.x_test.values, self.y_test.values):

            # Get the output of the network
            ys = inputs
            for layer in self.layers:
                ys = layer.get_outputs(ys)

            # Get the index of the maximum value, the index correspond to which class that output represents
            y = ys.argmax()

            self.confusion_matrix.add(target, y)

            # Determine of the network's choice is correct or not
            correct += 1 if y == target else 0

        # Calculate the accuracy
        self.test_accuracy = (correct / len(self.y_test)) * 100

        self.logger.info(f"Finished testing with accuracy {self.test_accuracy}")
        self.logger.info(f"The confusion matrix is:\n{self.confusion_matrix}")

        return self.confusion_matrix, round(self.test_accuracy, 3)

    def test_sample(self, sample):
        # Return the result of the network on a user-provided sample
        ys = sample
        for layer in self.layers:
            ys = layer.get_outputs(ys)

        return ys.argmax()
