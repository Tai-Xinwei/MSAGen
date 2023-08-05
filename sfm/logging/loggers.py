# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import asdict

from loguru import logger

from sfm.utils.dist_utils import is_master_node

import wandb  # isort:skip

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
    def log(self, metrics, prefix=""):
        if not is_master_node():
            return

        if wandb.run is None:
            # Log to console
            logger.info(metrics)
        else:
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


def console_log_filter(record):
    # For message with level INFO, we only log it on master node
    # For others, we log it on all nodes
    if record["level"].name != "INFO":
        return True

    return is_master_node()
