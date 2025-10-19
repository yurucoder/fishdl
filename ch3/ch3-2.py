# 신경망 2 - 3층 신경망 구현하기

from common.functions import sigmoid, identity_function
import sys
import os
import numpy as np

sys.path.append(os.getcwd())


# 퍼셉트론 복습
# b + w1x1 + w2x2
# 퍼셉트론이란: x1, x2와 같이 여러 가지 입력 값으로 하나의 결과값을 내는 논리 단위
# 가중치(w1, w2)란: 각 신호가 가지는 영향력을 나타냄
# 편향(b)란: 뉴런이 얼마나 쉽게 활성화될 수 있는지를 제어

# 신경망이란?
# 퍼셉트론에서는 우리가 원하는 결과(OR, XOR 게이트 등)을 얻기 위해 그 가중치를 수동으로 설정했다.
# 신경망에서는 우리가 원하는 결과를 얻기 위해 가중치를 자동으로 학습하는 능력을 가진다.

# 활성화함수란?
# 퍼셉트론에서는, 가중치가 적용된 모든 값을 합한 결과를 기준치 0을 기준으로 판단하였다.
# 이는 0을 기준으로 분기를 나누는 함수 h(x)로 표현할 수 있다.
# h(x)와 같이 입력 신호의 총합을 출력 신호로 변환하는 함수를 활성화 함수라 한다.

# 행렬을 사용하여 신경망 표현하기

# 입력층 -> 1층
X = np.array([1.0, 0.5])  # 입력값
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 가중치
B1 = np.array([0.1, 0.2, 0.3])  # 편향
A1 = np.dot(X, W1) + B1  # 1층
Z1 = sigmoid(A1)  # 활성화함수 적용

# 1층 -> 2층도 동일한 방법이다.
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 2층 -> 출력층도 동일한 방법이다.
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # identity_function은 항등 함수로 Y = A3이다.

# 출력층의 활성화함수 sigma(x)는 문제의 성질에 따라 바뀐다.

# 구현 정리


def init_network():
    return {
        "W1": np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        "b1": np.array([0.1, 0.2, 0.3]),
        "W2": np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        "b2": np.array([0.1, 0.2]),
        "W3": np.array([[0.1, 0.3], [0.2, 0.4]]),
        "b3": np.array([0.1, 0.2]),
    }


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
