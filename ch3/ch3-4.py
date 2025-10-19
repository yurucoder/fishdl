# 신경망 4 - 손글씨 숫자 인식

from common.functions import sigmoid, softmax
from dataset.mnist import load_mnist
import sys
import os
import time
import pickle
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())


# 이미 학습된 매개변수를 사용하여 학습 과정 없이 추론 과정만 구현
# 이 추론 과정을 '순전파' (forward propagation)이라고 한다


# MNIST 데이터셋 활용


# 넘파이로 저장된 이미지를 PIL 객체로 변환해야 함 (flatten=True)
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def use_mnist_image():
    # 훈련된 이미지는 넘파이 객체로 피클 파일에 저장
    # (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)

    print(
        f"x_train.shape: {x_train.shape}",
        f"t_train.shape: {t_train.shape}",
        f"x_test.shape: {x_test.shape}",
        f"t_test.shape: {t_test.shape}",
        sep="\n",
    )

    img = x_train[0]
    label = t_train[0]
    print(label)  # 5

    print(img.shape)  # (784,) - 1차원으로 압축된 이미지 데이터
    img = img.reshape(28, 28)  # 원래 이미지의 모양으로 변형
    print(img.shape)  # (28, 28) = 2차원 이미지 데이터

    img_show(img)


# MNIST 로부터 numpy 배열 데이터를 받는다. (입력층 데이터)
def get_data():
    (_, _), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


# 학습된 가중치 매개변수를 읽어온다
# 이 데이터에 따라 입력 데이터의 확률이 결정된다! (pkl 데이터를 만드는 걸 '학습한다'라고 표현한다.)
# [[a, b], [c, d]] 에서 안쪽배열(행)이 다음층의 뉴런수, 바깥배열(열)이 이전층의 뉴런수를 의미
def init_network():
    with open("dataset/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


# network 데이터에 따라 3층 신경망을 진행
def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 추론 및 평가 흐름
def normal_process():
    # 입력층 데이터는 784(28x28)개: 사진의 픽셀 개수
    # 출력층 데이터는 10개: 0~9 사이의 숫자를 구분함
    # 이 샘플 학습 모델에서는 첫 번째 은닉층에 50개, 두 번째 은닉층에 100개의 뉴런 설정됨
    # 784 -> 50 -> 100 -> 10
    x, t = get_data()
    network = init_network()

    # 모델 정확도 평가
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 가장 확률이 높은 것
        if p == t[i]:
            accuracy_cnt += 1

    # 입력 개수 대비 정답 수
    print(f"Accuracy: {float(accuracy_cnt) / len(x)}")


# batch 처리
def batch_process():
    x, t = get_data()
    network = init_network()

    batch_size = 100  # 배치 크기
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i: i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i: i + batch_size])

    print(f"Accuracy: {float(accuracy_cnt) / len(x)}")


# 배치 처리와 일반 처리 성능 비교
def check_time():
    t0 = time.time()
    normal_process()
    t1 = time.time()
    ta = t1 - t0
    print(f"A가 걸린 시간: {ta}")

    t0 = time.time()
    batch_process()
    t1 = time.time()
    tb = t1 - t0
    print(f"B가 걸린 시간: {tb}")

    print(f"A가 오래걸림: {ta - tb > 0}")


check_time()
