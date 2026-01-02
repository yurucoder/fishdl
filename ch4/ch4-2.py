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
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()


# 수치 미분을 한 해가 해석적 해와 매우 근사한 것을 알 수 있다
print(
    f"f(5) 수치미분: {numerical_diff(function_1, 5)}, 해석적 해: 0.2",
    f"f(10) 수치미분: {numerical_diff(function_1, 10)}, 해석적 해: 0.3",
    sep="\n",
)


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
print(
    f"x0=3 편미분: {numerical_diff(function_tmp1, 3.0)}, 해석적 해: 6",
    f"x1=4 편미분: {numerical_diff(function_tmp2, 4.0)}, 해석적 해: 8",
    sep="\n",
)


def numerical_gradient(f, x):
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

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


print(
    f"f(3, 4) 기울기: {numerical_gradient(function_2, np.array([3.0, 4.0]))}",
    f"f(0, 2) 기울기: {numerical_gradient(function_2, np.array([0.0, 2.0]))}",
    f"f(3, 0) 기울기: {numerical_gradient(function_2, np.array([3.0, 0.0]))}",
    sep="\n",
)


# 경사법이란: 특정 x에서의 기울기를 구하여, 더 값이 더 낮은 x쪽으로 최적화 하는 방법
# 구현에서는 학습률 u(에타)에 기울기를 곱한 값을 지속적으로 빼면서 가까운 위치를 찾는다.
# 그래프가 점차 완만해질수록 감소율은 줄어들고, 적절한 지점을 찾기 위해 학습률 조절이 필요하다.


# 경사법 구현: 100번 루프 후 가장 최적화된 x값을 리턴한다.
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


# 문제 예시
print(
    "경사법 (-3, 4)에서 시작, 학습률 0.1, 100번 반복",
    gradient_descent(
        function_2,
        init_x=np.array([-3.0, 4.0]),
        lr=0.1,
        step_num=100,
    ),
    "학습률이 너무 큰 경우 (학습률 10.0)",
    gradient_descent(
        function_2,
        init_x=np.array([-3.0, 4.0]),
        lr=10.0,
        step_num=100,
    ),
    "학습률이 너무 작은 경우 (학습률 1e-10)",
    gradient_descent(
        function_2,
        init_x=np.array([-3.0, 4.0]),
        lr=1e-10,
        step_num=100,
    ),
    sep="\n",
)
