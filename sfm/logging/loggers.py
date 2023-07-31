# -*- coding: utf-8 -*-
import os
import sys

import torch
import wandb
from loguru import logger

from sfm.pipeline.accelerator.dataclasses import LogOutput

handlers = {}


def get_logger():
    if not handlers:
        logger.remove()  # remove default handler
        handlers["console"] = logger.add(
            sys.stdout,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>][{level}]: {message}",
            colorize=True,
            filter=is_master_node,
        )

        if wandb_configed():
            wandb_project = os.getenv("WANDB_PROJECT")
            wandb_run_name = os.getenv("WANDB_RUN_NAME")
            wandb_tags = os.getenv("WANDB_TAGS")
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                tags=wandb_tags,
            )

            handlers["wandb"] = logger.add(
                wandb_sink,
                format=wandb_format,
                filter=wandb_filter,
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
    message = record["message"]
    if isinstance(message, LogOutput):
        return True

    return False


def wandb_sink(msg):
    wandb.log(msg)


def wandb_format(msg):
    message = msg["message"]
    if isinstance(message, LogOutput):
        return message.to_dict()

    return message
