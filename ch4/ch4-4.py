# 신경망 학습 4 - 학습 알고리즘 구현

# 신경망 정리

# 0. 전제
# - 신경망에는 적응 가능한 가중치와 편향이 있고
# - 이 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습' 이라고 합니다.

# 1. 미니배치
# - 훈련 데이터 중 일부를 무작위로 가져옵니다.
# - 이렇게 선별한 데이터를 미니배치라 하며, 이 미니배치의 손실함수 값을 줄이는 것이 목표입니다.

# 2. 기울기 산출
# - 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다.
# - 기울기는 손실함수의 값을 가장 작게 하는 방향을 제시합니다.

# 3. 매개변수 갱신
# - 가중치 매개변수를 기울기 방향으로 아주 조금 갱신합니다.

# 4. 반복
# - 1~3 단계를 반복합니다.

import sys, os
import numpy as np

sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
from common.two_layer_net import TwoLayerNet


# 샘플 데이터 다운로드
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 하이퍼파라미터
ITERS_NUM = 1000  # 반복 횟수
TRAIN_SIZE = x_train.shape[0]  # 학습 데이터 크기
BATCH_SIZE = 100  # 미니배치 크기
LEARNING_RATE = 0.1  # 학습률

# 손실함수 값 기록
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(TRAIN_SIZE / BATCH_SIZE, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(ITERS_NUM):
    # 미니배치 획득
    batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch)  # 성능 개선판!

    # 매개변수 갱신
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= LEARNING_RATE * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")
