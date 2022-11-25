from model import MLP
from data_handlers import MNIST, Penguins

def penguins_run():
    penguins = Penguins("data/penguins.csv")
    model = MLP(
        *penguins.partition_data(),
        [8, 12, 5],
        bias=1,
        eta=0.5,
        mse_threshold=0.05,
        epochs=500
    )
    model.train()
    acc = model.test()
    print("Penguins Accuracy: ", acc)

def mnist_run():
    mnist = MNIST("data/mnist_train.csv", "data/mnist_test.csv")
    model = MLP(
        *mnist.partition_data(),
        [8, 14],
        bias=1,
        eta=0.4,
        mse_threshold=0.15,
        epochs=5
    )
    model.train()
    acc = model.test()
    print("MNIST Accuracy: ", acc)

if __name__ == "__main__":
    penguins_run()
