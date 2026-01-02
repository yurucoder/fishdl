# 퍼셉트론

import numpy as np


# AND 게이트는 (1, 1)에서만 1을 출력한다.
def AND(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.7  # 편향
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(
    "AND 게이트의 진리표",
    f"1 AND 1: {AND(1, 1)}",  # 1
    f"1 AND 0: {AND(1, 0)}",  # 0
    f"0 AND 1: {AND(0, 1)}",  # 0
    f"0 AND 0: {AND(0, 0)}",  # 0
    sep="\n",
)


# NAND 게이트는 AND게이트와 반대인 진리표를 갖는다.
def NAND(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([-0.5, -0.5])  # 가중치(음수)
    b = 0.7  # 편향
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(
    "NAND 게이트의 진리표",
    f"1 NAND 1: {NAND(1, 1)}",  # 0
    f"1 NAND 0: {NAND(1, 0)}",  # 1
    f"0 NAND 1: {NAND(0, 1)}",  # 1
    f"0 NAND 0: {NAND(0, 0)}",  # 1
    sep="\n",
)


# OR 게이트는 입력 중 하나라도 1이 있다면 1을 출력한다.
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # AND와 편향만 다르다!
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(
    "OR 게이트의 진리표",
    f"1 OR 1: {OR(1, 1)}",  # 1
    f"1 OR 0: {OR(1, 0)}",  # 1
    f"0 OR 1: {OR(0, 1)}",  # 1
    f"0 OR 0: {OR(0, 0)}",  # 0
    sep="\n",
)


# 선형 그래프로 보았을 때 위의 세 함수는 기울기가 같고 그 위치만 다르다!
# AND의 경우 x와 y 값이 둘 다 그래프의 위에 위치한 경우 1을 반환한다
# NAND의 경우 AND와 [부등호가 반대인] 경우이므로, AND와 같은 그래프의 아래에 위한 경우 1을 반환한다
# OR의 경우 AND와 편향만 다르므로, 동일하게 OR그래프 위에 위치한 경우 1을 반환한다

# 한편, XOR(배타적 논리합)은 선형 영역을 나눌 수 없는 비선형 영역이다.
# 컴퓨터로 구현할 수 있는 건 선형 영역 뿐이므로, 단독 함수로는 불가능함을 알 수 있다.


# 따라서 XOR은 다층 퍼셉트론으로 구현한다.
def XOR(x1, x2):  # 0층 (x1, x2)
    s1 = NAND(x1, x2)  # 1층 (s1, s2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)  # 2층 (y)
    return y  # NAND와 OR의 교집합(AND)이 XOR이 된다.


print(
    "XOR 게이트의 진리표",
    f"0 XOR 0: {XOR(0, 0)}",  # 0
    f"1 XOR 0: {XOR(1, 0)}",  # 1
    f"0 XOR 1: {XOR(0, 1)}",  # 1
    f"1 XOR 1: {XOR(1, 1)}",  # 0
    sep="\n",
)
