import numpy as np

def _expand(grad_out,axis,in_shape):
  if len(grad_out.shape) < len(in_shape): grad_out = grad_out[None,:] if axis == 0 else grad_out[:,None]
  return np.broadcast_to(grad_out,in_shape)

def _reduce(grad_out,in_shape):
    if len(in_shape) < len(grad_out.shape): in_shape = (1,) + in_shape
    reduce_axis = tuple(i for i in range(len(grad_out.shape)) if in_shape[i] == 1 and grad_out.shape[i] > 1)
    return grad_out.sum(axis=reduce_axis,keepdims=True if len(in_shape) > 1 else False)