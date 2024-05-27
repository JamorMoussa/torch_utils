import torch, torch.nn as nn
from torch.utils.data import DataLoader 

import torch_utils.data as ts_data
import torch_utils.optim as ts_optim
from torch_utils.base import TrainerBase

from typing import Tuple


__all__ = ["RegressionTrainer"]



class RegressionTrainer(TrainerBase):
    
    def compile(
            self,
            model: nn.Module,
            loss_fn: nn.modules.loss._Loss,
            opt_container: ts_optim.OptimizerContainer,
            data_container: ts_data.DataContainer
    ) -> None:
        pass 