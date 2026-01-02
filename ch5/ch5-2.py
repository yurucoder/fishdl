# 오차역전파법 2 - 신경망 구축하기

import sys, os
import numpy as np

sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
from network import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 기울기 검증하기

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f"{key}: {diff}")


# 학습 구현하기

ITERS_NUM = 10000  # 반복 횟수
TRAIN_SIZE = x_train.shape[0]  # 학습 데이터 크기
BATCH_SIZE = 100  # 미니배치 크기
LEARNING_RATE = 0.1  # 학습률

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(TRAIN_SIZE / BATCH_SIZE, 1)

for i in range(ITERS_NUM):
    batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법으로 기울기를 구한다
    grad = network.gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= LEARNING_RATE * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")
