import numpy as np


class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def forward(self, *args, **kwargs):
    raise NotImplementedError()

  def backward(self, *args, **kwargs):
    raise NotImplementedError()

  @classmethod
  def apply(op, *x, **kwargs):
    ctx = op(*x)
    ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs))
    ret._ctx = ctx
    return ret


from minigrad import ops


class Tensor:
  def __init__(self, data):
    self.data = np.array(data, dtype=np.float32)
    self.grad = None
    self._ctx = None

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def ndim(self):
    return self.data.ndim

  def __repr__(self):
    return f"Tensor({self.data})"

  def add(self, x):
    return ops.Add.apply(self, x)

  def neg(self):
    return ops.Neg.apply(self)

  def sub(self, x):
    return self.add(x.neg())

  def mul(self, x):
    return ops.Mul.apply(self, x)

  def reshape(self, shape=None):
    return ops.Reshape.apply(self, shape)

  def sum(self, axis=None):
    return ops.Sum.apply(self, axis=axis)

  def log(self):
    return ops.Log.apply(self)

  def dot(self, x):
    return ops.Dot.apply(self, x)

  def mean(self):
    return ops.Mean.apply(self)

  def relu(self):
    return ops.Relu.apply(self)

  def sigmoid(self):
    return ops.Sigmoid.apply(self)

  def logsoftmax(self):
    return ops.LogSoftMax.apply(self)

  def square(self):
    return self * self

  def abs(self):
    return self.relu() + self.neg().relu()

  def mean_squared_error(self, x):
    return (self - x).square().mean()

  @classmethod
  def zeros(cls, *shape):
    return cls(np.zeros(shape, dtype=np.float32))

  @classmethod
  def eye(cls, n, m=None):
    return cls(np.eye(n, m, dtype=np.float32))

  @classmethod
  def xavier_uniform(cls, in_features, out_features):
    range = np.sqrt(6 / (in_features + out_features))
    return cls(np.random.uniform(-range, +range, (in_features, out_features)))

  @classmethod
  def xavier_normal(cls, in_features, out_features):
    scale = np.sqrt(2 / (in_features + out_features))
    return cls(np.random.normal(0.0, scale, (in_features, out_features)))

  @classmethod
  def from_numpy(cls, array):
    return cls(array.astype(float))

  def backward(self, autofill=True):
    if self._ctx is None:
      return

    if self.grad is None and autofill:
      assert self.data.shape == tuple(), "Backward can only be called on the scalar tensors"
      self.grad = np.ones_like(self.data)

    grads = self._ctx.backward(self.grad)
    if len(self._ctx.parents) == 1:
      grads = [grads]

    for t, g in zip(self._ctx.parents, grads):
      assert t.shape == g.shape, f"{t.grad.shape} != {g.shape}"
      t.grad = g if t.grad is None else (t.grad + g)
      t.backward(False)
