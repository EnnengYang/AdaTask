import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

def str_to_list(string):
    return [float(s) for s in string.split(",")]

def str_or_float(value):
    try:
        return float(value)
    except:
        return value

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, help="path to data")
common_parser.add_argument("--log_path", type=Path, help="path to log")
common_parser.add_argument("--n-epochs", type=int, default=200)
common_parser.add_argument("--n_task", type=int, default=2)
common_parser.add_argument("--batch-size", type=int, default=120, help="batch size")
common_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
common_parser.add_argument("--method-params-lr", type=float, default=0.025, help="lr for weight method params. If None, set to args.lr. For uncertainty weighting",)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,)

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )

