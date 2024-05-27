# torch_utils

**torch_utils**  is a PyTorch extension designed for training and building deep learning models. 

## Example 


```python
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split

# import the torch_utils 
import torch_utils as ts
```

For this example, we're going to use a simple data. which is generated randomly.

```python 
X = torch.rand(100, 3)
y = torch.mm(X, torch.Tensor([1, 2, 3]).unsqueeze(0).t())

dataset = TensorDataset(X, y)
```
Let build a simple model, just a single `Linear` layer.

```python
model = nn.Sequential(
    nn.Linear(3, 1)
)
```

Now, let create an instance of our `Trainer` class. The `Trainer` will charge of training of the model.

```python
trainer = ts.train.Trainer()
```

```
Trainer(
  (configs): TrainerConfigs(max_iters=10, train_batch=64, test_batch=64, shuffle_data=True, device=device(type='cpu'))
)
```

Let's change some configurations. For example, change the `max_ietrs` to be 100. 

```python
trainer.configs.max_iters = 100
```

```python
train_set, test_set = random_split(dataset, [0.8, 0.2])
``` 

The trainer support `DataContainer`, takes the train, and the test sets.

```python
data_container = ts.data.DataContainer(train_set, test_set)
```

aloso the trainer takes the `OptimizerContainer` rather than the torch `Optimizer`:

```python 
opt = ts.optim.OptimizerContainer(optim.Adam(model.parameters(), lr=0.01))
```

Now, let's compile the trainer. Provide all building blocks needed to train the model, such as optimizer, loss functio, datasets.

```python
trainer.compile(
    model= model, 
    opt_container= opt,
    loss_fn= nn.MSELoss(),
    data_container= data_container
)
```

Finally, train the mode, using the `trainer.train()` 

```python 
trainer.train()
```

```
100%|████████████████████████████████████████| 100/100 [00:00<00:00, 325.21it/s]
```
