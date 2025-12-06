import numpy as np
from numpy.typing import NDArray


# 그라디언트
def numerical_gradient(f, x: NDArray):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]
        x[i] = tmp_val + h
        fxh1 = f(x)
        x[i] = tmp_val - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (h * 2)
        x[i] = tmp_val

    return grad
