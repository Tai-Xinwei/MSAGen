# -*- coding: utf-8 -*-
import importlib
import logging
import os
import pickle
import sys
import zlib

from sfm.logging import logger

for submod in ("numeric", "multiarray"):
    sys.modules[f"numpy._core.{submod}"] = importlib.import_module(
        f"numpy.core.{submod}"
    )


def add_to_env(toadd: dict):
    for k, v in toadd.items():
        if k in os.environ:
            logger.warning(
                f"Warning: environment variable {k}={os.environ[k]} already exists, will be overwritten"
            )
        os.environ[k] = str(v)


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))


def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
