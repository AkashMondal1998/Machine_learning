import minigrad
import minigrad.nn as nn
from minigrad import Tensor
from tqdm import trange
import math


class Net(nn.Base):
  def __init__(self):
    self.l1 = nn.Linear(784, 200, dtype=minigrad.dtypes.float32)
    self.l2 = nn.Linear(200, 10, dtype=minigrad.dtypes.float32)

  def forward(self, x):
    x = self.l1(x).relu()
    x = self.l2(x)
    return x


if __name__ == "__main__":
  dataset = nn.DataSet("mnist", to_tensor=True)
  X_train, Y_train, X_test, Y_test = dataset.get_dataset()
  model = Net()
  optimizer = nn.Adam(model.parameters())
  BS = 256
  num_iters = math.ceil(X_train.shape[0] / BS)

  for i in range(10):
    for _ in (t := trange(num_iters)):
      samp = Tensor.randint(X_train.shape[0], size=(BS))
      optimizer.zero_grad()
      loss = model(X_train[samp]).sparse_categorical_crossentropy(Y_train[samp])
      loss.backward()
      optimizer.step()
      accuracy = (model(X_train[samp]).log_softmax().argmax(axis=1) == Y_train[samp]).mean()
      t.set_description(f"epoch-->{i+1}  loss={loss.item():0.2f} accuracy={accuracy.item():0.2f}")

  print(f"accuracy on test set = {(model(X_test).log_softmax().argmax(axis=1) == Y_test).mean().item()*100:0.2f}")
