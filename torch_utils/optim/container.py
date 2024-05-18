import torch.optim as optim


__all__ = ["OptimizerContainer", ]

class OptimizerContainer:

    opt: optim.Optimizer

    def __init__(self, opt: optim.Optimizer) -> None:
        self.opt = opt