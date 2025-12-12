import sys, os
import numpy as np
from numpy.typing import NDArray

sys.path.append(os.getcwd())
from common.layers import *
from common.gradient import numerical_gradient


# 이층 네트워크
class TwoLayerNet:
    def __init__(
        self,
        input_size: int,  # 입력층의 뉴런 수
        hidden_size: int,  # 은닉층의 뉴런 수
        output_size: int,  # 출력층의 뉴런 수
        weight_init_std=0.01,
    ):
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }

    # 추론 실행
    def predict(self, x: NDArray):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 손실함수 값을 구함
    def loss(self, x: NDArray, t: NDArray):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # 정확도를 구한다
    def accuracy(self, x: NDArray, t: NDArray):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    # 가중치 매개변수의 기울기를 구한다
    def numerical_gradient(self, x: NDArray, t: NDArray):
        loss_W = lambda W: self.loss(x, t)

        # W는 더미이지만, self.params[] 부분이 레퍼런스 전달이기 때문에 predict() 결과가 바뀐다.
        return {
            "W1": numerical_gradient(loss_W, self.params["W1"]),
            "b1": numerical_gradient(loss_W, self.params["b1"]),
            "W2": numerical_gradient(loss_W, self.params["W2"]),
            "b2": numerical_gradient(loss_W, self.params["b2"]),
        }
