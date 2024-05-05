import numpy as np
import gzip
import os
from urllib.request import urlretrieve
from minigrad import Tensor


class DataSet:
  def __init__(self, dataset, to_tensor=False):
    if dataset in ["fashion_mnist", "mnist"]: self.dataset = dataset
    else: raise ValueError("Wrong dataset name")
    self.url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/" if dataset == "fashion_mnist" else "https://storage.googleapis.com/cvdf-datasets/mnist/"
    self.path = os.path.join(os.getcwd(), "datasets/fashion_mnist/") if dataset == "fashion_mnist" else os.path.join(os.getcwd(), "datasets/mnist/")
    self.to_tensor = to_tensor

  def _fetch_dataset(self, file, offset):
    os.makedirs(self.path, exist_ok=True)
    if not os.path.isfile(self.path + file): urlretrieve(self.url + file, self.path + file)
    with gzip.open((self.path + file)) as f:
      array = np.frombuffer(f.read()[offset:], dtype=np.uint8)
    return array

  def get_dataset(self):
    X_train = self._fetch_dataset("train-images-idx3-ubyte.gz", 16).reshape(-1, 28 * 28).astype(np.float32)
    Y_train = self._fetch_dataset("train-labels-idx1-ubyte.gz", 8).reshape(-1,).astype(np.int32)
    X_test = self._fetch_dataset("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 28 * 28).astype(np.float32)
    Y_test = self._fetch_dataset("t10k-labels-idx1-ubyte.gz", 8).reshape(-1,).astype(np.int32)
    if self.to_tensor: return Tensor(X_train), Tensor(Y_train), Tensor(X_test), Tensor(Y_test)
    return X_train, Y_train, X_test, Y_test
