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
    if ctx.requires_grad: ret._ctx = ctx
    return ret


import minigrad.function as F

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

  @property
  def size(self): return self.data.size

  def __repr__(self): return f"Tensor({self.data},requires_grad={self.requires_grad})"

  def add(self,x,reverse=False): return F.Add.apply(*self._const(x,reverse))

  def sub(self,x,reverse=False): return F.Sub.apply(*self._const(x,reverse))

  def div(self,x,reverse=False): return F.Div.apply(*self._const(x,reverse))

  def mul(self,x,reverse=False): return F.Mul.apply(*self._const(x,reverse))
  
  def pow(self,x,reverse=False): return F.Pow.apply(*self._const(x,reverse))

  def maximum(self,x,reverse=False): return F.Maximum.apply(*self._const(x,reverse))

  def reshape(self, shape=None): return F.Reshape.apply(self, shape=shape)

  def sum(self, axis=None,keepdims=False): return F.Sum.apply(self, axis=axis,keepdims=keepdims)

  def neg(self): return F.Neg.apply(self)

  def log(self): return F.Log.apply(self)
  
  def exp(self): return F.Exp.apply(self)

  def sqrt(self): return self.pow(0.5)

  def dot(self, x):
    if self.ndim == x.ndim == 1: assert self.shape[0] == x.shape[0], f"1D tensors are not aligned {self.shape[0]} (dim0) != {x.shape[0]} (dim0)"
    else: assert self.shape[1] == x.shape[0] , f"2D tensors are not aligned {self.shape[1]} (dim1) != {x.shape[0]} (dim0)"
    return F.Dot.apply(self, x)

  def matmul(self,x):
    assert self.ndim == x.ndim == 2, f"2D tensors are expected" 
    return self.dot(x)

  def mean(self): return self.sum().div(self.size)

  def relu(self): return F.Relu.apply(self)

  def sigmoid(self): return F.Sigmoid.apply(self)

  def log_softmax(self): return F.LogSoftMax.apply(self)

  def square(self): return self * self

  def abs(self): return self.relu() + self.neg().relu()

  def mean_squared_error(self, x): return (self.sub(x)).square().mean()

  def binary_crossentropy(self,x): 
    return ((-x * self.log()) - ((1 - x) * (1 - self).log())).mean()

  def binary_crossentropy_withlogits(self,x):
    return (self.maximum(0) - (self * x) + (1 + self.abs().neg().exp()).log()).mean()

  def sparse_categorical_crossentropy(self,x):
    # self is the raw logits 
    # x is the one the one hot encoded vector
    return self.log_softmax().neg().mul(x).sum(axis=1).mean()

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
  def from_numpy(cls, array): return cls(array.astype(np.float32))

  def backward(self, autofill=True):
    if not self._ctx: return

    if self.grad is None and autofill:
      assert self.data.shape == tuple(), "Backward can only be called on the scalar tensors"
      self.grad = np.array(1.0)

    grads = self._ctx.backward(self.grad)
    grads = [grads] if len(self._ctx.parents) == 1 else grads

    for t, g in zip(self._ctx.parents, grads):
      assert t.shape == g.shape, f"{t.shape} != {g.shape}"
      t.grad = g if t.grad is None else (t.grad + g)
      t.backward(False)

  def _const(self,val,reverse):
    val = Tensor(np.full_like(self.data,val),requires_grad=False) if isinstance(val,(int,float)) else val 
    if reverse: return val,self
    else: return self,val


  def __neg__(self): return self.neg()
  def __add__(self,x): return self.add(x)
  def __sub__(self,x): return self.sub(x)
  def __mul__(self,x): return self.mul(x)
  def __pow__(self,x): return self.pow(x)
  def __matmul__(self,x): return self.dot(x)
  def __truediv__(self,x): return self.div(x)

  def __radd__(self,x): return self.add(x,True)
  def __rsub__(self,x): return self.sub(x,True)
  def __rmul__(self,x): return self.mul(x,True)
  def __rpow__(self,x): return self.pow(x,True)
  def __rtruediv__(self,x): return self.div(x,True)