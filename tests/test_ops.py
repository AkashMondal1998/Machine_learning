from minigrad import Tensor
import numpy as np
import torch


def test_add():
  x, y = Tensor([1, 2, 4], requires_grad=True), Tensor([9, 2, 1], requires_grad=True)
  d = x + y
  d = d.sum()
  d.backward()

  t, m = torch.tensor([1, 2, 4], dtype=torch.float32, requires_grad=True), torch.tensor([9, 2, 1], dtype=torch.float32, requires_grad=True)
  e = t + m
  e = e.sum()
  e.backward()

  assert d.data == e.detach().numpy()
  assert (x.grad == t.grad.detach().numpy()).all()
  assert (y.grad == m.grad.detach().numpy()).all()


def test_sub():
  x, y = Tensor([1, 2, 4], requires_grad=True), Tensor([9, 2, 1], requires_grad=True)
  d = x - y
  d = d.sum()
  d.backward()

  t, m = torch.tensor([1, 2, 4], dtype=torch.float32, requires_grad=True), torch.tensor([9, 2, 1], dtype=torch.float32, requires_grad=True)
  e = t - m
  e = e.sum()
  e.backward()

  assert d.data == e.detach().numpy()
  assert (x.grad == t.grad.detach().numpy()).all()
  assert (y.grad == m.grad.detach().numpy()).all()


def test_mul():
  x, y = Tensor([1, 2, 4], requires_grad=True), Tensor([9, 2, 1], requires_grad=True)
  d = x * y
  d = d.sum()
  d.backward()

  t, m = torch.tensor([1, 2, 4], dtype=torch.float32, requires_grad=True), torch.tensor([9, 2, 1], dtype=torch.float32, requires_grad=True)
  e = t * m
  e = e.sum()
  e.backward()

  assert d.data == e.detach().numpy()
  assert (x.grad == t.grad.detach().numpy()).all()
  assert (y.grad == m.grad.detach().numpy()).all()


def test_div():
  x, y = Tensor([1, 2, 4], dtype=np.float32, requires_grad=True), Tensor([9, 2, 1], dtype=np.float32, requires_grad=True)
  d = x / y
  d = d.sum()
  d.backward()

  t, m = torch.tensor([1, 2, 4], dtype=torch.float32, requires_grad=True), torch.tensor([9, 2, 1], dtype=torch.float32, requires_grad=True)
  e = t / m
  e = e.sum()
  e.backward()

  assert d.data == e.detach().numpy()
  assert (x.grad == t.grad.detach().numpy()).all()
  assert (y.grad == m.grad.detach().numpy()).all()


def test_matmul():
  x, y = Tensor([[1, 2, 4], [6, 2, 1]], requires_grad=True), Tensor([[9], [2], [1]], requires_grad=True)
  d = x @ y
  d = d.sum()
  d.backward()

  t, m = torch.tensor([[1, 2, 4], [6, 2, 1]], dtype=torch.float32, requires_grad=True), torch.tensor(
      [[9], [2], [1]], dtype=torch.float32, requires_grad=True)
  e = t @ m
  e = e.sum()
  e.backward()

  assert d.data == e.detach().numpy()
  assert (x.grad == t.grad.detach().numpy()).all()
  assert (y.grad == m.grad.detach().numpy()).all()


def test_dot():
  x, y = Tensor([1, 2, 4], requires_grad=True), Tensor([9, 2, 1], requires_grad=True)
  d = x.dot(y)
  d = d.sum()
  d.backward()

  t, m = torch.tensor([1, 2, 4], dtype=torch.float32, requires_grad=True), torch.tensor([9, 2, 1], dtype=torch.float32, requires_grad=True)
  e = t.dot(m)
  e = e.sum()
  e.backward()

  assert d.data == e.detach().numpy()
  assert (x.grad == t.grad.detach().numpy()).all()
  assert (y.grad == m.grad.detach().numpy()).all()
