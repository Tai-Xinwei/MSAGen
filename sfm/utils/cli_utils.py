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
                wandb_api_key = os.getenv("WANDB_API_KEY")

                if not wandb_api_key:
                    logger.warning("Wandb not configured, logging to console only")
                else:
                    args.wandb = True
                    wandb_project = os.getenv("WANDB_PROJECT")
                    wandb_run_name = os.getenv("WANDB_RUN_NAME")
                    wandb_team = os.getenv("WANDB_TEAM")
                    wandb_group = os.getenv("WANDB_GROUP")

                    args.wandb_team = getattr(args, "wandb_team", wandb_team)
                    args.wandb_group = getattr(args, "wandb_group", wandb_group)
                    args.wandb_project = getattr(args, "wandb_project", wandb_project)

                    wandb.init(
                        project=wandb_project,
                        group=wandb_group,
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
