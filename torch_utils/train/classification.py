import torch, torch.nn as nn
from torch.utils.data import DataLoader 

import torch_utils.data as ts_data
import torch_utils.optim as ts_optim
from torch_utils.base import TrainerBase

from typing import Tuple, Dict


__all__ = ["ClassificationTrainer"]

def train_step_base(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: ts_optim.OptimizerContainer,
    loader: DataLoader,
    device: torch.device,
): 
    loss_val, acc = 0, 0

    for batch, (X, y) in enumerate(loader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            loss_val += loss.item()

            # 3. Optimizer zero grad
            opt.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            opt.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    loss_val = loss_val / len(loader)
    acc = acc / len(loader)
    
    return loss, acc


def train_step_classification(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: ts_optim.OptimizerContainer,
    train_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    
    model.train()

    train_loss, train_acc = train_step_base(
        model= model,
        loss_fn= loss_fn,
        opt= opt,
        loader= train_loader,
        device= device
    )

    return train_loss, train_acc


def test_step_classifcation(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: ts_optim.OptimizerContainer,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
     
    model.eval()

    test_loss, test_acc = train_step_base(
        model= model,
        loss_fn= loss_fn,
        opt= opt,
        loader= test_loader,
        device= device
    )

    return test_loss, test_acc


class ClassificationTrainer(TrainerBase):

    model: nn.Module
    loss_fn: nn.modules.loss._Loss
    opt_container: ts_optim.OptimizerContainer
    train_loader:  DataLoader
    test_loader: DataLoader
    device: torch.device

    results: Dict[str, list]

    def __init__(self, ):
        
        self.results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
        }
        
    def compile(
            self,
            model: nn.Module,
            loss_fn: nn.modules.loss._Loss,
            opt_container: ts_optim.OptimizerContainer,
            train_loader: DataLoader,
            test_loader: DataLoader,
            device: torch.device,
    ) -> None:
         
        self.model = model
        self.opt_container = opt_container
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
         

    def train_step(self):
        
        train_loss, train_acc = train_step_classification(
            model= self.model,
            loss_fn= self.loss_fn,
            opt= self.opt_container,
            train_loader= self.train_loader,
            device= self.device
        )

        self.results["train_loss"].append(train_loss)
        self.results["train_acc"].append(train_acc)

    def test_step(self):

        test_loss, test_acc = test_step_classifcation(
            model= self.model,
            loss_fn= self.loss_fn,
            opt= self.opt_container,
            test_loader= self.test_loader,
            device = self.device
        )

        self.results["test_loss"].append(test_loss)
        self.results["test_acc"].append(test_acc) 


    