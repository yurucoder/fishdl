# 합성곱 신경망

import sys, os
import numpy as np
from collections import OrderedDict

sys.path.append(os.getcwd())
from common.layers import Affine, Relu, SoftmaxWithLoss


# im2col, col2im 함수에 대해
# 이미지-행렬 변환 함수들. 행렬 계산 최적화를 위해 만든 함수.
# 3차원 데이터를 펼친 후 순서대로 붙여 2차원 데이터로 만든다.
# 각종 연산 라이브러리들은 2차원 데이터에 대해 최적화가 잘 되어있는 편이다.
# 원래 형태를 알고 있으므로 결과물은 다시 3차원으로 복구할 수 있다.


# 저자가 구현한 im2col 함수
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


# 합성곱 계층: 이미지(3차원) 데이터를 가공하여 필터로 곱하는 연산 계층.
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.pad)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.pad)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


# 풀링 계층: 데이터의 가로/세로 값을 줄이고, 원하는 형태로 성형하는 계층
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 최댓값
        out = np.max(col, axis=1)

        # 성형
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out


# Conv-ReLU-Pooling-Affine-ReLU-Affine-Softmax 순으로 진행
class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={
            "filter_num": 30,
            "filter_size": 5,
            "pad": 0,
            "stride": 1,
        },
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        self.params = {
            "W1": weight_init_std
            * np.random.randn(filter_num, input_dim[0], filter_size, filter_size),
            "b1": np.zeros(filter_num),
            "W2": weight_init_std * np.random.randn(pool_output_size, hidden_size),
            "b2": np.zeros(hidden_size),
            "W3": weight_init_std * np.random.randn(hidden_size, output_size),
            "b3": np.zeros(output_size),
        }

        self.layers = OrderedDict(
            {
                "Conv1": Convolution(
                    self.params["W1"],
                    self.params["b1"],
                    conv_param["stride"],
                    conv_param["pad"],
                ),
                "Relu1": Relu(),
                "Pool1": Pooling(pool_h=2, pool_w=2, stride=2),
                "Affine1": Affine(self.params["W2", self.params["b2"]]),
                "Relu2": Relu(),
                "Affine2": Affine(self.params["W3"], self.params["b3"]),
            }
        )

        self.last_layer = SoftmaxWithLoss

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward()
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        # 순전피
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과
        return {
            "W1": self.layers["Conv1"].dW,
            "b1": self.layers["Conv1"].db,
            "W2": self.layers["Affine1"].dW,
            "b2": self.layers["Affine1"].db,
            "W3": self.layers["Affine2"].dW,
            "b3": self.layers["Affine2"].db,
        }
