# -*- coding: utf-8 -*-
from sfm.logging.loggers import logger


class Register:
    # a register from https://applenob.github.io/python/register/
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        if key not in self._dict:
            raise ValueError("Key %s is not in registry %s." % (key, self._name))
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()
