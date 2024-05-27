import torch, torch.nn as nn
from torch.utils.data import DataLoader

from .. import optim as ts_optim
from .. import data as ts_data
from ..base import ConfigsBase, TrainerBase
from .classification import ClassificationTrainer
from .regression import RegressionTrainer

from dataclasses import dataclass, field
from typing import Self, Any

from tqdm import tqdm


__all__ = ["Trainer", "TrainerConfigs", ]


@dataclass
class TrainerConfigs(ConfigsBase):

    max_iters: int = 10
    train_batch: int = 64
    test_batch: int  = 64
    shuffle_data: bool = True

    device: torch.device = torch.device("cpu")

    types: tuple[type] = field(default=(int, int, int, bool, torch.device), repr=False)

    @staticmethod
    def get_defaults() -> Self:
        return TrainerConfigs()


class Trainer(TrainerBase):

    model: nn.Module
    loss_fn: nn.modules.loss._Loss
    opt_container: ts_optim.OptimizerContainer
    data_container: ts_data.DataContainer

    train_loader: DataLoader
    test_loader: DataLoader

    configs: TrainerConfigs

    is_classification: bool = True

    sub_trainer: TrainerBase = None
    
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
        
        self.model = model.to(self.configs.device)
        self.loss_fn = loss_fn
        self.opt_container = opt_container
        self.data_container = data_container

        self.is_classification = self._check_cls_mode(loss_fn= loss_fn)

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
        
        if self.is_classification: 
            self.sub_trainer = ClassificationTrainer()
        else:
            self.sub_trainer = RegressionTrainer()

        
        self.sub_trainer.compile(
            model= model, 
            loss_fn= loss_fn,
            opt_container= opt_container,
            train_loader= self.train_loader,
            test_loader= self.test_loader,
            device= self.configs.device
        )
        
    def train_step(self):
        self.sub_trainer.train_step()

    def test_step(self):
        self.sub_trainer.test_step()
        
    
    def train(self):

        for iter in (bar := tqdm(range(self.configs.max_iters))):
            
            self.train_step()

            self.test_step()

    def __repr__(self) -> str:
        return f"Trainer(\n{' '*2}(configs): {self.configs}\n)"
    

    def _check_cls_mode(
        self,
        loss_fn: nn.modules.loss._Loss,
    ):
        cls_loss_fns = (nn.CrossEntropyLoss, nn.BCELoss)

        return True if any(isinstance(loss_fn, loss) for loss in cls_loss_fns) else False

