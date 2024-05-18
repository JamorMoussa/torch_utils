import torch, torch.nn as nn
from torch.utils.data import DataLoader

from .. import optim as ts_optim
from .. import data as ts_data

from dataclasses import dataclass
from typing import Self

from tqdm import tqdm


__all__ = ["Trainer", "TrainerConfigs", ]


@dataclass
class TrainerConfigs:

    max_iters: int = 10
    train_batch: int = 64
    test_batch: int  = 64
    shuffle_data: bool = True

    device: torch.device = torch.device("cpu")

    @staticmethod
    def get_defaults() -> Self:
        return TrainerConfigs()


class Trainer:

    model: nn.Module
    loss_fn: nn.modules.loss._Loss
    opt_container: ts_optim.OptimizerContainer
    data_container: ts_data.DataContainer

    train_loader: DataLoader
    test_loader: DataLoader

    configs: TrainerConfigs

    device: torch.device
    
    def __init__(
        self,
        configs: TrainerConfigs = TrainerConfigs.get_defaults()
    ) -> None:    
        self.configs = configs

    def compile(
            self,
            model: nn.Module,
            loss_fn: nn.modules.loss._Loss,
            opt_container: ts_optim.OptimizerContainer,
            data_container: ts_data.DataContainer
    ) -> None:
        
        self.model = model
        self.loss_fn = loss_fn
        self.opt_container = opt_container
        self.data_container = data_container

        data_configs = ts_data.DataContainerConfigs(
                            train_batch= self.configs.train_batch,
                            test_batch= self.configs.test_batch,
                            shuffle= self.configs.shuffle_data
                        )
        self.data_container.set_configs(data_configs)


        self.train_loader: DataLoader = self.data_container.get_loader(
                    self.data_container.train_set,
                    self.data_container.configs.train_batch
                )
        
        self.test_loader: DataLoader = self.data_container.get_loader(
                    self.data_container.test_set,
                    self.data_container.configs.test_batch
                )
        
    def train_step(self):
        
        self.model.train()

        for batch, (X, y) in enumerate(self.train_loader):

            X, y = X.to(self.configs.device), y.to(self.configs.device)

            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)

            self.opt_container.opt.zero_grad()

            loss.backward()

            self.opt_container.opt.step()

    def train(self):

        for iter in (bar := tqdm(range(self.configs.max_iters))):
            self.train_step()
           # bar.set_description(f"Hello {iter}")


    def __repr__(self) -> str:
        return f"Trainer(\n{' '*2}(configs): {self.configs}\n)"

