# 신경망 학습 3 - 신경망에서의 기울기

import sys, os
import numpy as np

sys.path.append(os.getcwd())
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# 신경망
class simpleNet:
    # 초기 가중치를 정규분포로 초기화
    def __init__(self):
        self.W = np.random.randn(2, 3)

    # 입력값 x를 받아 신경망 1층 진행
    def predict(self, x):
        return np.dot(x, self.W)

    # 손실함수로 값 평가
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)


x = np.array([0.6, 0.9])  # 입력 레이블
t = np.array([0, 0, 1])  # 정답 레이블

# 단순한 신경망 학습 프로세스
net = simpleNet()
print("가중치 매개변수:", net.W)

p = net.predict(x)
print("1층 진행 후 결과:", p)
print("최댓값의 인덱스:", np.argmax(p))
print("손실 함수 결과:", net.loss(x, t))

# t에 대한 x의 손실함수 f(W)를 가중치 레이블 W에 대하여 편미분하자
f = lambda w: net.loss(x, t)

# 기울기를 구하여 가중치 W를 평가하자
dW = numerical_gradient(f, net.W)
print("가중치별 기울기", dW)
