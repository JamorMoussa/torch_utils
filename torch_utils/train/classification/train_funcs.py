import torch, torch.nn as nn
from torch.utils.data import DataLoader 

import torch_utils.optim as ts_optim

from typing import Tuple

from tqdm import tqdm



def train_step_base(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: ts_optim.OptimizerContainer,
    loader: DataLoader,
    device: torch.device,
    batch: int = None
): 
    loss_val, acc = 0, 0

    is_train: bool = batch is not None

    iterator = (bar := tqdm(enumerate(loader), total= len(loader))) if is_train else enumerate(loader)

    for batch, (X, y) in iterator:
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
    
    #if batch is not None:
    #    bar.set_description(f"Loss: {loss_val:.5f} | Accuracy: {acc:.2f}%")

    mode = "Train" if is_train else "Test"
    
    print(f"| {mode} Loss: {loss_val:.5f} | {mode} Acc: {acc*100: .3f}%", end= (" "  if is_train else "| \n"))

    return loss_val, acc


def train_step_classification(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: ts_optim.OptimizerContainer,
    train_loader: DataLoader,
    device: torch.device,
    batch: int
) -> Tuple[float, float]:
    
    model.train()

    train_loss, train_acc = train_step_base(
        model= model,
        loss_fn= loss_fn,
        opt= opt,
        loader= train_loader,
        device= device,
        batch= batch
    )

    return train_loss, train_acc


def test_step_classifcation(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    opt: ts_optim.OptimizerContainer,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
     
    model.eval()

    test_loss, test_acc = train_step_base(
        model= model,
        loss_fn= loss_fn,
        opt= opt,
        loader= test_loader,
        device= device,
    )

    return test_loss, test_acc