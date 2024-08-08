# -*- coding: utf-8 -*-
import logging

from rich.logging import RichHandler


def setup_logging(**rich_handler_kwargs):
    log_handler = RichHandler(**rich_handler_kwargs)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[log_handler],
        force=True,
    )


setup_logging(rich_tracebacks=True, markup=True)
logger = logging.getLogger("sfm.data")
logger.setLevel(logging.INFO)
