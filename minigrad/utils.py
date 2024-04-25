import gzip,os
import numpy as np
from urllib.request import urlretrieve

def _load_mnsit(file,offset):
    with gzip.open(os.path.join(os.getcwd(),"datasets/mnist",file)) as f:
        array = np.frombuffer(f.read()[offset:],dtype=np.uint8)
    return array

def _fetch_fashion(file,offset):
    p = os.path.join(os.getcwd(),"datasets/fashion_mnist/")
    os.makedirs(p,exist_ok=True)
    if not os.path.exists(p+file): urlretrieve("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"+file,p+file)
    with gzip.open((p+file)) as f:
        array = np.frombuffer(f.read()[offset:],dtype=np.uint8)
    return array