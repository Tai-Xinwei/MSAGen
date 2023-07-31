# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from dataclasses import _MISSING_TYPE, fields


def add_dataclass_to_parser(configs: list, parser: ArgumentParser):
    for config in configs:
        group = parser.add_argument_group(config.__name__)
        for field in fields(config):
            name = field.name.replace("-", "_")
            if field.default != _MISSING_TYPE:
                default = field.default
            elif field.default_factory != _MISSING_TYPE:
                default = field.default_factory()
            else:
                default = None

            if field.type == bool:
                action = "store_false" if default else "store_true"
                group.add_argument("--" + name, action=action, default=default)
            else:
                group.add_argument("--" + name, type=field.type, default=default)

    return parser


def from_args(args, config):
    kwargs = {}
    for field in fields(config):
        name = field.name.replace("-", "_")
        kwargs[name] = getattr(args, name)
    return config(**kwargs)
