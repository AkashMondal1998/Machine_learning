import numpy as np

def _expand(x,shape,axis,keepdims):
  if not keepdims: 
    x = np.expand_dims(x,axis)
  num_repeats = shape[axis]
  return np.repeat(x,num_repeats,axis)


def _reduce(grad_out,in_shape):
    if len(in_shape) < len(grad_out.shape): temp_shape = (1,) + in_shape
    else: temp_shape = in_shape
    sum_axis = [i for i in range(len(grad_out.shape)) if temp_shape[i] != grad_out.shape[i]]
    return grad_out.sum(axis=tuple(sum_axis)).reshape(in_shape)