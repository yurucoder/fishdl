# 신경망 1 - 활성화 함수

import numpy as np
import matplotlib.pylab as plt


# 이 함수에서는 x가 실수(부동소수점)인데, 넘파이 지원에는 정수가 필요하다.
def step_function_1(x):
    if x > 0:
        return 1
    else:
        return 0


# 넘파이 배열 x를 부등식으로 비교하면 bool 배열 y가 생성된다.
# y를 astype() 메서드를 사용하여 0과 1로 이루어진 정수 배열로 변환할 수 있다.
def step_function_2(x):
    y = x > 0
    return y.astype(np.int)


# 위 함수를 조금 더 간략화할 수 있다!
def step_function_3(x):
    return np.array(x > 0, dtype=np.int32)


def 계단함수_그래프():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function_3(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y축의 범위
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def 시그모이드_그래프():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y축의 범위
    plt.show()


계단함수_그래프()
시그모이드_그래프()


# ReLu(Rectified Linear Unit)은 최근 많이 사용하는 활성화 함수이다!
# 값이 0 이하면 0을 출력하고 이상이면 그 값을 그대로 출력한다.
def ReLU(x):
    return np.maximum(0, x)
