# 신경망 학습 2 - 수치 미분

import numpy as np
import matplotlib.pylab as plt


# 나쁜 구현의 예
def numerical_diff_1(f, x):
    h = 1e-50  # 너무 작은 수는 float32 타입에서 에러를 일으킨다.
    # f(x+h) - f(x) 는 x+h 와 x 사이의 기울기로, x의 접선과는 거리가 있다.
    return (f(x + h) - f(x)) / h


# 개선된 구현
def numerical_diff(f, x):
    h = 1e-4  # 일반적으로 1e-4 정도면 좋은 수치가 나온다고 한다.
    # f(x+h) - f(x-h) 는 중심차분/중앙차분 이라고 부르며, x에 더 가까운 수치이다.
    return (f(x + h) - f(x - h)) / (2 * h)


# 다음과 같이 "진정한 미분"이 아닌 알고리즘상 수치적 근삿값을 뽑아내는 것을
# [수치미분] 이라고 하고, 이런 것을 연구하는 학문을 [수치해석학] 이라고 한다.
# 우리가 원래 배웠던 방식의 진정한 미분은 [해석적 미분] 이라고 부른다.

# 다음과 같은 함수를 미분해보자
# y = 0.01x^2 + 0.1x


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


x = np.arange(0.0, 20.0, 0.1)  # 0에서 20까지 0.1 간격의 배열 x (20 미포함)
y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()

# 해석적 해: 0.2, 수치미분 해: 0.1999999999990898
print(numerical_diff(function_1, 5))
# 해석적 해: 0.3, 수치미분 해:0.2999999999986347
print(numerical_diff(function_1, 10))

# 수치 미분을 한 해가 해석적 해와 매우 근사한 것을 알 수 있다


# 편미분을 위한 함수: f(x1, x2) = x0^2 + x1^2
def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)


# x1 = 4 일때 x0에 대한 식
def function_tmp1(x0):
    return x0 * x0 + 4.0**2.0


# x0 = 3 일때 x1에 대한 식
def function_tmp2(x1):
    return 3.0**2.0 + x1 * x1


# x0 = 3, x1 = 4 일때 x0에 대한 편미분 구하기
print(numerical_diff(function_tmp1, 3.0))  # 6.00000000000378 (6)
print(numerical_diff(function_tmp2, 4.0))  # 7.999999999999119 (8)


def numerical_gradient(f, x: np.ndarray):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / 2 * h
        x[idx] = tmp_val  # 값 복원

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
