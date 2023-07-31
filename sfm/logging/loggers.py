# -*- coding: utf-8 -*-
import logging
import os
from abc import ABC, abstractmethod

import wandb

logging.basicConfig(level=logging.NOTSET)

sfm_logger = logging.getLogger()
sfm_logger.setLevel(logging.INFO)

# Create a Formatter object with a timestamp
formatter = logging.Formatter("[%(asctime)s]: %(message)s")

# Create a StreamHandler object and set its Formatter to the one we just created
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add the StreamHandler to the logger
sfm_logger.handlers.clear()
sfm_logger.addHandler(stream_handler)


class Logger(ABC):
    def log(self, *data):
        log_text = " ".join([str(d) for d in data])
        sfm_logger.info(log_text)
        stream_handler.flush()

    @abstractmethod
    def log_metrics(self, metrics):
        pass


class TextLogger(Logger):
    def log_metrics(self, metrics):
        return self.log(metrics)


class WandbLogger(Logger):
    def __init__(self, project, run_name, tags, config):
        self.project = project
        self.run_name = run_name
        self.tags = tags
        self.config = config
        self.run = None

        wandb.init(
            project=self.project, name=self.run_name, tags=self.tags, config=self.config
        )

    def log_metrics(self, metrics):
        # convert data class to dict
        if hasattr(metrics, "_asdict"):
            metrics = metrics._asdict()

        wandb.log(metrics)


def get_logger() -> Logger:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_run_name = os.getenv("WANDB_RUN_NAME")
    wandb_tags = os.getenv("WANDB_TAGS")

    if not wandb_api_key or not wandb_project:
        sfm_logger.info("Wandb not configured, using text logger")
        return TextLogger()
    else:
        sfm_logger.info("Wandb configured, using wandb logger")
        return WandbLogger(
            project=wandb_project,
            run_name=wandb_run_name,
            tags=wandb_tags,
            config={},
        )
