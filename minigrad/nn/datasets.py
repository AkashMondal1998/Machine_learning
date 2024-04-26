import numpy as np
from minigrad.utils import _fetch_fashion, _load_mnsit,_fetch_mnist

def get_fashion_mnist():
    X_train = _fetch_fashion("train-images-idx3-ubyte.gz",16).reshape(-1,28*28).astype(np.float32)
    Y_train = _fetch_fashion("train-labels-idx1-ubyte.gz",8).reshape(-1,).astype(np.int32)
    X_test = _fetch_fashion("t10k-images-idx3-ubyte.gz",16).reshape(-1,28*28).astype(np.float32)
    Y_test = _fetch_fashion("t10k-labels-idx1-ubyte.gz",8).reshape(-1,).astype(np.int32)
    return X_train,Y_train,X_test,Y_test

def get_mnist():
    X_train = _fetch_mnist("train-images-idx3-ubyte.gz",16).reshape(-1,28*28).astype(np.float32)
    Y_train = _fetch_mnist("train-labels-idx1-ubyte.gz",8).reshape(-1,).astype(np.int32)
    X_test = _fetch_mnist("t10k-images-idx3-ubyte.gz",16).reshape(-1,28*28).astype(np.float32)
    Y_test = _fetch_mnist("t10k-labels-idx1-ubyte.gz",8).reshape(-1,).astype(np.int32)
    return X_train, Y_train, X_test, Y_test