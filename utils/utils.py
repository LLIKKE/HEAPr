import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_parameter_count(count):
    """Format parameter counts with K/M/B suffixes.

    Args:
        count: Parameter count.

    Returns:
        Human-readable count string.
    """
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    else:
        return f"{count}"


def count_parameters(model, logger=None):
    """Count parameters in a model and optionally log it.

    Args:
        model: PyTorch model to count parameters for.
        logger: Optional logger for output.

    Returns:
        Total number of parameters.
    """
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
    if logger is not None:
        logger.info(
            f"{'parameters'}: {format_parameter_count(total_params):>5} parameters"
        )
    return total_params



def find_named_layers(module, layers=None, name: str = ""):
    """Recursively find target layers and return a name-to-layer map.

    Args:
        module: Root module to search.
        layers: Iterable of layer classes to match.
        name: Current module name prefix.

    Returns:
        Mapping from qualified layer names to layer modules.
    """
    if layers is None:
        layers = [torch.nn.Linear]
    if isinstance(module, torch.nn.Embedding) and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(
            find_named_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def find_qlayers(module, layers=None, name: str = ""):
    return find_named_layers(module, layers=layers, name=name)
