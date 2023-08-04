# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from dataclasses import _MISSING_TYPE, fields
from enum import Enum

from sfm.logging import loggers


def make_enum_praser(enum: Enum):
    choices = [e.name for e in enum]

    def parse_enum(arg):
        try:
            return enum[arg]
        except KeyError:
            raise ValueError(
                f"Invalid choice: {arg} for type {enum.__name__}. Valid choices are: {choices}"
            )

    return parse_enum


def add_dataclass_to_parser(configs, parser: ArgumentParser):
    exist_configs = set()

    for config in configs:
        group = parser.add_argument_group(config.__name__)
        for field in fields(config):
            name = field.name.replace("-", "_")

            if name in exist_configs:
                loggers.warning(f"Duplicate config name: {name}, not added to parser")
                continue
            else:
                exist_configs.add(name)

            if field.default != _MISSING_TYPE:
                default = field.default
            elif field.default_factory != _MISSING_TYPE:
                default = field.default_factory()
            else:
                default = None

            if field.type == bool:
                action = "store_false" if default else "store_true"
                group.add_argument("--" + name, action=action, default=default)
            elif issubclass(field.type, Enum):
                parse_enum = make_enum_praser(field.type)
                group.add_argument("--" + name, type=parse_enum, default=default)
            else:
                group.add_argument("--" + name, type=field.type, default=default)

    return parser


def from_args(args, config):
    kwargs = {}
    for field in fields(config):
        name = field.name.replace("-", "_")
        kwargs[name] = getattr(args, name)
    return config(**kwargs)
