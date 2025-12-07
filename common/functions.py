import numpy as np
from numpy.typing import NDArray


def identity_function(x: NDArray):
    return x


def step_function_(x: NDArray):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x: NDArray):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: NDArray):
    x = sigmoid(x)
    return (1.0 - x) * x


def relu(x: NDArray):
    return np.maximum(0, x)


def relu_grad(x: NDArray):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def softmax(a: NDArray):
    exp_a = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return exp_a / np.sum(exp_a, axis=-1, keepdims=True)


def sum_squared_error(y: NDArray, t: NDArray):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y: NDArray, t: NDArray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X: NDArray, t: NDArray):
    return cross_entropy_error(softmax(X), t)
