# -*- coding: utf-8 -*-
import os

import mendeleev


def get_all_elements():
    return mendeleev.get_all_elements()


def get_all_elements_symbols():
    return [e.symbol for e in get_all_elements()]


def get_all_digits():
    return [str(i) for i in range(10)]


def get_all_space_groups():
    return [str(i) for i in range(1, 231)]


def get_dict():
    return get_all_elements_symbols() + get_all_space_groups()


def main(save_path):
    result = get_dict()
    with open(save_path, "w") as f:
        for tok in result:
            f.write(tok + "\n")


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    main(os.path.join(path, "dict.txt"))
