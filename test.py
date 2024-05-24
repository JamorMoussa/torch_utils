import torch_utils as ts


configs = ts.train.TrainerConfigs.get_defaults()

configs.max_iters = True

print(configs)