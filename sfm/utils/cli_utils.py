# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from functools import wraps

from sfm.logging import logger
from sfm.pipeline.accelerator.trainer import seed_everything
from sfm.utils import arg_utils, dist_utils, env_init

import wandb  # isort:skip


def cli(*cfg_classes):
    def decorator(main):
        @wraps(main)
        def wrapper():
            parser = ArgumentParser()
            parser = arg_utils.add_dataclass_to_parser(cfg_classes, parser)
            args = parser.parse_args()

            logger.info(args)

            seed_everything(args.seed)

            env_init.set_env(args)

            if dist_utils.is_master_node():
                wandb_project = os.getenv("WANDB_PROJECT")
                wandb_run_name = os.getenv("WANDB_RUN_NAME")
                wandb_api_key = os.getenv("WANDB_API_KEY")

                if not wandb_api_key:
                    logger.warning("Wandb not configured, logging to console only")
                else:
                    wandb.init(
                        project=wandb_project,
                        name=wandb_run_name,
                        config=args,
                    )

            logger.success(
                "====================================Start!===================================="
            )
            try:
                main(args)
            except Exception as e:
                logger.exception(e)
                raise e

            logger.success(
                "====================================Done!===================================="
            )

        return wrapper

    return decorator
