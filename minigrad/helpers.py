import numpy as np

def _expand(x,shape,axis,keepdims):
  if not keepdims: 
    x = np.expand_dims(x,axis)
  num_repeats = shape[axis]
  return np.repeat(x,num_repeats,axis)