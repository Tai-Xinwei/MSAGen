# -*- coding: utf-8 -*-
import pickle
import zlib
from typing import Any


def bstr2obj(bstr: bytes) -> Any:
    """Decompress bytes and load the object"""
    return pickle.loads(zlib.decompress(bstr))


def obj2bstr(obj: Any) -> bytes:
    """Dump the object and compress the bytes"""
    return zlib.compress(pickle.dumps(obj))
