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