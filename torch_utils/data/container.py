from torch.utils.data import Dataset, DataLoader

from ..base import ConfigsBase

from dataclasses import dataclass, field
from typing import Self


__all__ = ["DataContainer", "DataContainerConfigs"]

@dataclass
class DataContainerConfigs(ConfigsBase):

    train_batch: int 
    test_batch: int 
    shuffle: bool 

    types: tuple[type] = field(default= (int, int, bool), repr=False)

    @staticmethod
    def get_defaults() -> Self:
        return DataContainer(train_batch= 64, test_batch= 64, shuffle= True)


class DataContainer:

    train_set: Dataset
    test_set: Dataset

    configs: DataContainerConfigs

    def __init__(
            self,
            train_set: Dataset,
            test_set: Dataset,     
    ) -> None:
        
        self.train_set = train_set
        self.test_set = test_set

    def set_configs(self, configs: DataContainerConfigs):
        self.configs = configs

    def get_loader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle= self.configs.shuffle)