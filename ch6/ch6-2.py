# 학습 관련 기술들 2 - 가중치의 초깃값

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from common.functions import sigmoid, relu


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # 바람직하지 못한 초깃값
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01

    # Xavier 초깃값 (활성화 함수가 sigmoid나 tanh처럼 S자일 때)
    # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

    # HE 초깃값 (활성화 함수가 ReLU일 때)
    w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num)

    a = np.dot(x, w)
    # z = sigmoid(a)
    z = relu(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(f"{i+1}-layer")
    plt.ylim(ymin=0, ymax=7000)
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()
