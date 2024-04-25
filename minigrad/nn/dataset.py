import gzip,os
import numpy as np

def _load_file(file,offset):
    with gzip.open(os.path.join(os.getcwd(),"datasets/mnist",file)) as f:
        array = np.frombuffer(f.read()[offset:],dtype=np.uint8)
    return array

def get_mnist():
    X_train = _load_file("train-images-idx3-ubyte.gz",16).reshape(-1,28*28).astype(np.float32)
    Y_train = _load_file("train-labels-idx1-ubyte.gz",8).reshape(-1,).astype(np.int32)
    X_test = _load_file("t10k-images-idx3-ubyte.gz",16).reshape(-1,28*28).astype(np.float32)
    Y_test = _load_file("t10k-labels-idx1-ubyte.gz",8).reshape(-1,).astype(np.int32)
    return X_train, Y_train, X_test, Y_test