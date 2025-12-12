import numpy as np
from numpy.typing import NDArray


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
