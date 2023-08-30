from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 10000
learning_rate = 0.1

for e in range(epochs):
    error = 0
    for x, y in zip(X,Y):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += mse(y, output)

        # backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print('%d/%d, error=%f' % (e + 1, epochs, error))


# test
def xor(a, b):
    return a ^ b

num_samples = 300
test_set_X = np.random.randint(0, 2, (num_samples, 2, 1))
test_set_Y = np.array([[xor(a[0], a[1])] for a in test_set_X])



for x, y in zip(test_set_X, test_set_Y):
    output = x
    for layer in network:
        output = layer.forward(output)

    if y == np.round(output):
        print(f"SUCCESS! Given: {x}, result: {y}, pred: {np.round(output)}")
    else:
        print(f"FAIL! Given: {x}, result: {y}, pred: {np.round(output)}")