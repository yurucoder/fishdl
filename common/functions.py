import numpy as np
from numpy.typing import NDArray


# 계단 함수
def step_function_(x: NDArray):
    return np.array(x > 0, dtype=np.int32)


# 시그모이드 함수
def sigmoid(x: NDArray):
    return 1 / (1 + np.exp(-x))


# ReLU 함수
def ReLU(x: NDArray):
    return np.maximum(0, x)


# 항등 함수
def identity_function(x: NDArray):
    return x


# 소프트맥스 함수
def softmax(a: NDArray):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# 교차 엔트로피 오차
def cross_entropy_error(y: NDArray, t: NDArray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
