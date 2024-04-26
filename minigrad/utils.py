import gzip,os
import numpy as np
from urllib.request import urlretrieve

def _fetch_fashion(file,offset):
    p = os.path.join(os.getcwd(),"datasets/fashion_mnist/")
    os.makedirs(p,exist_ok=True)
    if not os.path.isfile(p+file): urlretrieve("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"+file,p+file)
    with gzip.open((p+file)) as f:
        array = np.frombuffer(f.read()[offset:],dtype=np.uint8)
    return array

def _fetch_mnist(file,offset):
    p = os.path.join(os.getcwd(),"datasets/mnist/")
    os.makedirs(p,exist_ok=True)
    if not os.path.isfile(p+file): urlretrieve("https://storage.googleapis.com/cvdf-datasets/mnist/"+file,p+file)
    with gzip.open((p+file)) as f:
        array = np.frombuffer(f.read()[offset:],dtype=np.uint8)
    return array