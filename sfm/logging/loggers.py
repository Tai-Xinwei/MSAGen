# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import asdict

import torch
from loguru import logger

handlers = {}


def get_logger():
    import wandb

    if not handlers:
        logger.remove()  # remove default handler
        handlers["console"] = logger.add(
            sys.stdout,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>][<cyan>{level}</cyan>]: {message}",
            colorize=True,
            filter=is_master_node,
            enqueue=True,
        )

        if wandb_configed() and is_master_node(None):
            wandb_project = os.getenv("WANDB_PROJECT")
            wandb_run_name = os.getenv("WANDB_RUN_NAME")
            wandb_tags = os.getenv("WANDB_TAGS")
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                tags=wandb_tags,
            )

            handlers["wandb"] = logger.add(
                wandb_sink, filter=wandb_filter, serialize=False
            )
        else:
            logger.warning("Wandb not configured, logging to console only")

    return logger


def is_master_node(_):
    if not torch.distributed.is_initialized():  # single node
        return True
    else:
        return torch.distributed.get_rank() == 0


def wandb_configed():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    return wandb_api_key and wandb_project


def wandb_filter(record):
    return "wandb_log" in record["extra"]


def wandb_sink(msg):
    import wandb

    wandb_log = asdict(msg.record["extra"]["wandb_log"])
    data = wandb_log
    for k, v in wandb_log["extra_output"].items():
        data[k] = v
    del data["extra_output"]
    wandb.log(data)
