# Minigrad
### A Mnist classifier
#### Inspired by [Tinygrad](https://github.com/tinygrad/tinygrad) and [Micrograd](https://github.com/karpathy/micrograd)


#### Mnist classifier using Minigrad
```python3
# Neural Net
class Net(nn.Base):
    def __init__(self):
        self.l1 = nn.Linear(784,200)
        self.l2 = nn.Linear(200,10)
        
    def forward(self,x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

model = Net()

# Get the Mnsit dataset
dataset = DataSet("mnist",to_tensor=True)
X_train,Y_train,X_test,Y_test = dataset.get_dataset()

optimizer = Adam(model.parameters())

BS = 256
num_iters = math.ceil(X_train.shape[0] / BS)

#Training
for i in range(10):
    for _ in (t:=trange(num_iters)):
        samp = Tensor.randint(X_train.shape[0],size=(BS))
        optimizer.zero_grad()
        loss = model(X_train[samp]).sparse_categorical_crossentropy(Y_train[samp])
        loss.backward()
        optimizer.step()
        accuracy = (model(X_train[samp]).log_softmax().argmax(axis=1) == Y_train[samp]).mean()
        t.set_description(f"epoch-->{i+1}  loss={loss.item():0.2f} accuracy={accuracy.item():0.2f}")
```

#### Mnist example
```bash
PYTHONPATH="." python3 examples/mnist.py
```
#### Fashion Mnist example
```bash
PYTHONPATH="." python3 examples/fashion_mnist.py
```

#### Running Test
Install pytest and torch to validate the gradients
```bash
python3 -m pytest
```