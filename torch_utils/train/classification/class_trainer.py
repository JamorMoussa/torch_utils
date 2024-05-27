import torch, torch.nn as nn
from torch.utils.data import DataLoader 

import torch_utils.optim as ts_optim
from torch_utils.base import TrainerBase
from .train_funcs import train_step_classification, test_step_classifcation
from .plots import *

from typing import Dict


__all__ = ["ClassificationTrainer"]


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
            batch: int 
    ) -> None:
         
        self.model = model
        self.opt_container = opt_container
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.batch = batch
         

    def train_step(self):
        
        train_loss, train_acc = train_step_classification(
            model= self.model,
            loss_fn= self.loss_fn,
            opt= self.opt_container,
            train_loader= self.train_loader,
            device= self.device,
            batch= self.batch,
        )

        self.results["train_loss"].append(train_loss)
        self.results["train_acc"].append(train_acc)

    def test_step(self):

        test_loss, test_acc = test_step_classifcation(
            model= self.model,
            loss_fn= self.loss_fn,
            opt= self.opt_container,
            test_loader= self.test_loader,
            device = self.device,
        )

        self.results["test_loss"].append(test_loss)
        self.results["test_acc"].append(test_acc) 

    def plot_accuracy(self):

        plot_accuracy(
            train_acc= self.results["train_acc"],
            test_acc= self.results["test_acc"]
        )

    def plot_loss(self):

        plot_loss(
            train_loss= self.results["train_loss"],
            test_loss= self.results["test_loss"]
        )



    