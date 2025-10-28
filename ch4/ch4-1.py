# 신경망 학습 1 - 손실 함수

import sys, os
import numpy as np

sys.path.append(os.getcwd())
from dataset.mnist import load_mnist


# 손실함수란: 신경망의 성능을 분석하기 위한 지표


# 손실함수 테스트용 데이터
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 정답은 2
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]


# 오차제곱합: 고딩 때 배운 분산과 유사
def sum_squares_error(y, t):
    # y가 신경망의 추정값, t가 정답(테스트)레이블이다.
    return 0.5 * np.sum((y - t) ** 2)


# y2의 결과가 더 높게 나온다! (오차)
print(
    "오차제곱합 계산: y1이 정답에 가까움",
    sum_squares_error(np.array(y1), np.array(t)),
    sum_squares_error(np.array(y2), np.array(t)),
    sep="\n",
)


# 교차 엔트로피 오차: -Sum(t * Log(y)) 의 구현
def cross_entropy_error_1(y, t):
    delta = 1e-7  # y가 0이 되어 -inf가 되지 않도록 방지함
    return -np.sum(t * np.log(y + delta))


# y2의 결과가 더 높게 나온다! (오차)
print(
    "교차 엔트로피 오차 계산: y1이 정답에 가까움",
    cross_entropy_error_1(np.array(y1), np.array(t)),
    cross_entropy_error_1(np.array(y2), np.array(t)),
    sep="\n",
)

# 미니배치 학습
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

np.random.choice(60000, 10)


# 배치용 교차 엔트로피 오차: 정답 레이블이 원-핫 인코딩인 경우
def cross_entropy_error_2(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 배치용 교차 엔트로피 오차: 정답 레이블이 원-핫 인코딩이 아닌 경우
def cross_entropy_error_3(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
