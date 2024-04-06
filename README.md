## Jupyter Notebooks for machine learning algorithms
### Models Done:
- Handwritten 1 or 0 prediction using MNIST DATASET
- Handwirtten digits prediction using MNIST DATASET (Tensorflow and Pytorch)
- Coffee Roasting Model (Tensorflow and Pytorch)


### Extra:
- Calculate gradient using auto back propagation


### Build neural network using numpy only
### Todo
- Back propagation


### Done
- Neuron 
- Layer of neurons
- Forward pass
- Sigmoid Activation function
- ReLU Activation function
- Mse loss function



### Forward pass and loss calucation using neural_engine
#### Neural Network 
``` 
    from neural_engine import nn

```
```
    class Net(Base):
        def __init__(self, in_features):
            self.l1 = nn.Layer(in_features, 3)
            self.relu = nn.ReLU()
            self.l2 = nn.Layer(3, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.l1(x))
            x = self.sig(self.l2(x))
            return x

    
    model = Net(x_train.shape[1])
```
#### Loss Function
```
    loss_fn  = nn.BCELoss()
    loss = loss_fn(model(x),y_train)
```