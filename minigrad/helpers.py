import numpy as np


def expand(x, axis, in_shape):
  if x.shape != in_shape:
    if x.shape == tuple(): x = x.reshape(1,)
    if len(x.shape) < len(in_shape) and axis is not None: x = np.expand_dims(x, axis=axis)
    x = np.broadcast_to(x, in_shape)
  return x


def reduce(x, in_shape):
  keepdims = True if len(in_shape) > 1 else False
  if len(in_shape) < len(x.shape): in_shape = (1,) * (len(x.shape) - len(in_shape)) + in_shape
  reduce_axis = tuple(i for i in range(len(x.shape)) if in_shape[i] == 1 and x.shape[i] > 1)
  return x.sum(axis=reduce_axis, keepdims=keepdims)
