# -*- coding: utf-8 -*-
import logging
import os
import pickle
import zlib

logger = logging.getLogger(__name__)


def add_to_env(toadd: dict):
    for k, v in toadd.items():
        if k in os.environ:
            logger.warning(
                f"Warning: environment variable {k}={os.environ[k]} already exists, will be overwritten"
            )
        os.environ[k] = str(v)


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))
