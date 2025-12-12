import sys, os
import numpy as np
from numpy.typing import NDArray

sys.path.append(os.getcwd())
from common.functions import *


class Relu:
    def __init__(self):
        # 순전파에서 0보다 작은 것은 True, 나머지는 False로 저장됨
        self.mask = None

    # 0 이하는 0으로, 0 이상은 값 그대로
    def forward(self, x: NDArray):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    # 0 이하는 0으로, 0 이상은 값 그대로
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W: NDArray, b: NDArray):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: NDArray):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout: NDArray):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# 소프트맥스 함수와 교차 엔트로피오차는 미분의 결과로 출력과 정답 레이블의 오차로 깔끔히 떨어진다.
# 이는 설계상 의도적인 결과로서, 항등함수와 오차제곱합의 조합도 마찬가지이다.
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: NDArray, t: NDArray):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape
        dx = (self.y - self.t) / batch_size

        return dx
