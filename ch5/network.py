import sys, os
import numpy as np
from collections import OrderedDict

sys.path.append(os.getcwd())
from common.layers import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(
        self,
        input_size: int,  # 입력층의 뉴런 수
        hidden_size: int,  # 은닉층의 뉴런 수
        output_size: int,  # 출력층의 뉴런 수
        weight_init_std=0.01,
    ):
        # 가중치 초기화
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }

        # 계층 생성
        self.layers = OrderedDict(
            {
                "Affine1": Affine(self.params["W1"], self.params["b1"]),
                "Relu1": Relu(),
                "Affine2": Affine(self.params["W2"], self.params["b2"]),
            }
        )

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        # W는 더미이지만, self.params[] 부분이 레퍼런스 전달이기 때문에 predict() 결과가 바뀐다.
        return {
            "W1": numerical_gradient(loss_W, self.params["W1"]),
            "b1": numerical_gradient(loss_W, self.params["b1"]),
            "W2": numerical_gradient(loss_W, self.params["W2"]),
            "b2": numerical_gradient(loss_W, self.params["b2"]),
        }

    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        return {
            "W1": self.layers["Affine1"].dW,
            "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db,
        }
