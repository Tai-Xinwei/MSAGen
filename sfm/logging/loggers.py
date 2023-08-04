# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import asdict

import torch
from loguru import logger

handlers = {}


def get_logger():
    if not handlers:
        logger.remove()  # remove default handler
        handlers["console"] = logger.add(
            sys.stdout,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>][<cyan>{level}</cyan>]: {message}",
            colorize=True,
            filter=console_log_filter,
            enqueue=True,
        )

    return logger


class MetricLogger(object):
    def __init__(self):
        import wandb

        if wandb_configed() and is_master_node(None):
            wandb_project = os.getenv("WANDB_PROJECT")
            wandb_run_name = os.getenv("WANDB_RUN_NAME")
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
            )
            self.log_to_wandb = True
        else:
            logger.warning("Wandb not configured, logging to console only")
            self.log_to_wandb = False

    def log(self, metrics, prefix=""):
        import wandb

        if not is_master_node():
            return

        if self.log_to_wandb:
            if type(metrics) is dict:
                log_data = metrics
            elif hasattr(metrics, "__dataclass_fields__"):
                log_data = asdict(metrics)

                if "extra_output" in log_data:
                    extra_output = log_data["extra_output"]
                    if extra_output is not None:
                        for k, v in extra_output.items():
                            log_data[k] = v
                    del log_data["extra_output"]

            # Add prefix
            if prefix:
                log_data = {f"{prefix}/{k}": v for k, v in log_data.items()}

            wandb.log(log_data)
        else:
            # Log to console
            logger.info(metrics)


def console_log_filter(record):
    # For message with level INFO, we only log it on master node
    # For others, we log it on all nodes
    if record["level"].name != "INFO":
        return True

    return is_master_node()


def is_master_node():
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        return True
    else:
        return False

    # if not torch.distributed.is_initialized():  # single node
    #     return True
    # else:
    #     return torch.distributed.get_rank() == 0 or deepspeed.comm.get_rank() == 0


def wandb_configed():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    return wandb_api_key and wandb_project
