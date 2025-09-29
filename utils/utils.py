import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_parameter_count(count):
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return f"{count}"

def count_parameters(model,logger=None):
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        #print(f"{name}: {param.numel()} parameters (requires_grad: {param.requires_grad})")
    if logger is not None:
        logger.info(f"{'parameters'}: {format_parameter_count(total_params):>5} parameters")
    return total_params
