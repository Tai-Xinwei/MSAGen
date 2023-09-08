# -*- coding: utf-8 -*-
import inspect
import os
from functools import wraps

from sfm.logging import logger

import wandb  # isort:skip


def check_grad():
    pass


def check_tensor():
    pass


def pipemode(forward_func):
    @wraps(forward_func)
    def wrapper(*args):
        # Get the parameter names from the original forward function
        param_names = list(inspect.signature(forward_func).parameters.keys())

        # Convert input tuple to dictionary
        input_dict = {param_name: arg for param_name, arg in zip(param_names, args)}

        # Call original forward function with input dictionary
        output_dict = forward_func(input_dict)

        # Convert output dictionary to tuple
        output_tuple = tuple(output_dict.values())

        return output_tuple

    return wrapper
