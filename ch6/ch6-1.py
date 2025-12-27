# 학습 관련 기술들 1 - 매개변수 갱신 최적화

import numpy as np


# SGD: 확률적 경사 하강법
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr  # learning rate

    # 초기 가중치에 그 미분값을 기준으로 값의 변동을 준다. (미분값 * 0.01)
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


"""신경망 학습 의사 코드

network = TwoLayerNet(...)
optimizer = SGD() # 최적화 과정을 담당하는 함수

for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...)     # 미니 배치 뽑아내기
    grads = network.gradient(x_batch, t_batch) # 그라디언트 구하기
    params = network.params                    # 초기 가중치 별도 저장 (참조를 막기 위함)
    optimizer.update(params, grads)            # 최적화된 값 업데이트
    ...

"""


# 물리학의 운동량의 변화 형태를 적용하여 최적화
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        # v가 비어있을 때, 가중치 행렬 크기과 같도록 0의 값을 채워준다.
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # 현재 모멘텀 = 기존 모멘텀(a * v) - 가중치 변화율(n * dL/dW)
            # a를 곱하는 이유: 운동량이 보존되지 않고, 서서히 가라앉게 하기 위함
            # 손실함수의 값의 기울기를 속도로 생각하고 모멘텀을 갱신함.
            self.v[key] = (self.momentum * self.v[key]) - (self.lr * grads[key])

            # 가중치에 모멘텀을 적용한다.
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 현재의 손실 함수 값에 따라 학습률을 줄여나간다.
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            # 0으로 나누는 일이 없도록 1e-7만큼 더해준다.
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
