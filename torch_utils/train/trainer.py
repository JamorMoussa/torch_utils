import torch.nn as nn
from .. import optim as ts_optim

from dataclasses import dataclass

from typing import Self


__all__ = ["Trainer", "TrainerConfigs", ]


@dataclass
class TrainerConfigs:

    max_iters: int

    @staticmethod
    def get_defaults() -> Self:
        return TrainerConfigs(max_iters= 10)



class Trainer:

    model: nn.Module
    loss: nn.modules.loss._Loss
    opt_container: ts_optim.OptimizerContainer

    configs: TrainerConfigs
    
    def __init__(self, configs: TrainerConfigs = TrainerConfigs.get_defaults()) -> None:
        self.configs = configs

    def compile(
            self,
            model: nn.Module,
            loss: nn.modules.loss._Loss,
            opt_container: ts_optim.OptimizerContainer
    ) -> None:
        
        self.model = model
        self.loss = loss
        self.opt_container = opt_container