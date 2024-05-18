import torch_utils as ts
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset



X = torch.rand(100, 3)
y = torch.mm(X, torch.Tensor([1, 2, 3]).unsqueeze(0).t())


dataset = TensorDataset(X, y)


print(dataset)



trainer = ts.train.Trainer()


# trainer.compile(
#     model=  
#     loss= nn.MSELoss(),
#     optimizer= optim.Adam(model.lr=)
# )
