import numpy as np

class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.requires_grad = any(t.requires_grad for t in tensors)

  def forward(self, *args, **kwargs): raise NotImplementedError()

  def backward(self, *args, **kwargs): raise NotImplementedError()

  @classmethod
  def apply(op, *x, **kwargs):
    ctx = op(*x)
    ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs), requires_grad=ctx.requires_grad)
    ret._ctx = ctx
    return ret


from minigrad import ops

class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = np.array(data, dtype=np.float32)
    self.grad = None
    self._ctx = None
    self.requires_grad = requires_grad

  @property
  def shape(self): return self.data.shape

  @property
  def dtype(self): return self.data.dtype

  @property 
  def ndim(self): return self.data.ndim

  def __repr__(self): return f"Tensor({self.data},requires_grad={self.requires_grad})"

  def add(self, x): return ops.Add.apply(self, self._const(x))

  def neg(self): return ops.Neg.apply(self)

  def sub(self, x): return self + (self._const(x)).neg()

  def mul(self, x): return ops.Mul.apply(self, self._const(x))

  def div(self,x,reverse=False): return ops.Div.apply(self, self._const(x),reverse=reverse)

  def reshape(self, shape=None): return ops.Reshape.apply(self, shape=shape)

  def sum(self, axis=None,keepdims=False): return ops.Sum.apply(self, axis=axis,keepdims=keepdims)

  def log(self): return ops.Log.apply(self)
  
  def exp(self): return ops.Exp.apply(self)

  def dot(self, x): return ops.Dot.apply(self, x)

  def mean(self): return ops.Mean.apply(self)

  def relu(self): return ops.Relu.apply(self)

  def sigmoid(self): return ops.Sigmoid.apply(self)

  def log_softmax(self): return ops.LogSoftMax.apply(self)

  def square(self): return self * self

  def abs(self): return self.relu() + self.neg().relu()

  def maximum(self,x): return ops.Maximum.apply(self,self._const_val(x))

  def mean_squared_error(self, x): return (self.sub(x)).square().mean()

  def binary_crossentropy_withlogits(self,x):
    return (self.maximum(0) - (self * x) + (1 + self.abs().neg().exp()).log()).mean()

  @classmethod
  def zeros(cls, *shape): return cls(np.zeros(shape, dtype=np.float32))

  @classmethod
  def eye(cls, n, m=None, requires_grad=False): return cls(np.eye(n, m, dtype=np.float32), requires_grad=requires_grad)

  @classmethod
  def xavier_uniform(cls, in_features, out_features, requires_grad=False):
    range = np.sqrt(6 / (in_features + out_features))
    return cls(np.random.uniform(-range, +range, (in_features, out_features)), requires_grad=requires_grad)

  @classmethod
  def xavier_normal(cls, in_features, out_features, requires_grad=False):
    scale = np.sqrt(2 / (in_features + out_features))
    return cls(np.random.normal(0.0, scale, (in_features, out_features)), requires_grad=requires_grad)

  @classmethod
  def from_numpy(cls, array): return cls(array.astype(float))

  def backward(self, autofill=True):
    if not self.requires_grad or not self._ctx: return

    if self.grad is None and autofill:
      assert self.data.shape == tuple(), "Backward can only be called on the scalar tensors"
      self.grad = np.ones_like(self.data)

    grads = self._ctx.backward(self.grad)
    grads = [grads] if len(self._ctx.parents) == 1 else grads

    for t, g in zip(self._ctx.parents, grads):
      assert t.shape == g.shape, f"{t.shape} != {g.shape}"
      if t.requires_grad:
        t.grad = g if t.grad is None else (t.grad + g)
        t.backward(False)

  def _const(self,val):
    if isinstance(val,(int,float)): return Tensor(np.full_like(self.data,val),requires_grad=False)
    else: return val


  def __add__(self,x): return self.add(x)

  def __radd__(self,x): return self.add(x)
  
  def __sub__(self,x): return self.sub(x)

  def __rsub__(self,x): return self.sub(x)
  
  def __mul__(self,x): return self.mul(x)

  def __rmul__(self,x): return self.mul(x)

  def __truediv__(self,x): return self.div(x)

  def __rtruediv__(self,x): return self.div(x,True)
  
  def __matmul__(self,x): return self.dot(x)
  
  def __neg__(self): return self.neg()

  
