import random, os
import numpy as np
import torch
import math


def seeding(seed=0, torch_deterministic=False):
    # print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def print_error(*message):
    print("\033[91m", "ERROR ", *message, "\033[0m")
    raise RuntimeError


def print_ok(*message):
    print("\033[92m", *message, "\033[0m")


def print_warning(*message):
    print("\033[93m", *message, "\033[0m")


def print_info(*message):
    print("\033[96m", *message, "\033[0m")


def filter_dict(input_dict):
    # Create a new dictionary to store filtered items
    filtered_dict = {}
    for key, value in input_dict.items():
        # Check if the value is a tensor and if it contains only finite numbers
        if torch.is_tensor(value) and torch.isfinite(value).all():
            filtered_dict[key] = value
        elif is_number(value) and not math.isnan(value):
            filtered_dict[key] = value
    return filtered_dict


def is_number(x):
    return isinstance(x, (int, float, complex))
