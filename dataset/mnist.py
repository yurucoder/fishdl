# mnist.py: Load MNIST data, by Saitoh Koki. koki0702@gmail.com
# https://github.com/oreilly-japan/deep-learning-from-scratch

import os
import pickle
import urllib.request
import gzip
import numpy as np


# Constants
TRAIN_IMG = "train_img"
TRAIN_LABEL = "train_label"
TEST_IMG = "test_img"
TEST_LABEL = "test_label"

# url_base = 'http://yann.lecun.com/exdb/mnist/'
url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # mirror site

key_file = {
    TRAIN_IMG: "train-images-idx3-ubyte.gz",
    TRAIN_LABEL: "train-labels-idx1-ubyte.gz",
    TEST_IMG: "t10k-images-idx3-ubyte.gz",
    TEST_LABEL: "t10k-labels-idx1-ubyte.gz",
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name: str):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"
    }
    request = urllib.request.Request(url_base + file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, mode="wb") as f:
        f.write(response)
    print("Done")


def download_mnist():
    for v in key_file.values():
        _download(v)


def _load_label(file_name: str):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name: str):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy():
    return {
        TRAIN_IMG: _load_img(key_file[TRAIN_IMG]),
        TRAIN_LABEL: _load_label(key_file[TRAIN_LABEL]),
        TEST_IMG: _load_img(key_file[TEST_IMG]),
        TEST_LABEL: _load_label(key_file[TEST_LABEL]),
    }


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X: np.ndarray):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in (TRAIN_IMG, TEST_IMG):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset[TRAIN_LABEL] = _change_one_hot_label(dataset[TRAIN_LABEL])
        dataset[TEST_LABEL] = _change_one_hot_label(dataset[TEST_LABEL])

    if not flatten:
        for key in (TRAIN_IMG, TEST_IMG):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset[TRAIN_IMG], dataset[TRAIN_LABEL]), (
        dataset[TEST_IMG],
        dataset[TEST_LABEL],
    )


if __name__ == "__main__":
    init_mnist()
