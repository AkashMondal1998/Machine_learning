# Minigrad
### A Mnist classifier
#### Inspired by [Tinygrad](https://github.com/tinygrad/tinygrad) and [Micrograd](https://github.com/karpathy/micrograd)

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