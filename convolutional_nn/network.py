import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

import sys
import os
sys.path.append(os.path.abspath('../neural_network'))
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Tanh, Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, num_classes=None)
    y = y.reshape(len(y), 2, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 100)

kernel_count = 10
kernel_size = 5

""" network = [
    Convolutional((1, 28, 28), kernel_size, kernel_count),
    Sigmoid(),
    Convolutional((kernel_count, 24, 24), kernel_size, kernel_count),
    Sigmoid(),
    Convolutional((kernel_count, 20, 20), kernel_size, kernel_count),
    Sigmoid(),
    Reshape((kernel_count, 16, 16), (kernel_count * 16 * 16, 1)),
    Dense(kernel_count * 16 * 16, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
] """

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.00625
# learning_rate = 0.1

for e in range(epochs):
    """ if e % 10 == 0:
        learning_rate = learning_rate / 2 
    print(f"learning_rate: {learning_rate}")"""
    error = 0
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)
            """ print(layer.__class__.__name__)
            print(output.size)
            print(output.shape) """

        # error
        error += binary_cross_entropy(y, output)

        #backward
        grad = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error={error}")

for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    if np.argmax(output) != np.argmax(y):
        print("FALSE")