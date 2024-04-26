# -*- coding: utf-8 -*-
import os
import json
import re


fname="/blob/lihe/scigpt/data/material/all_materials.pended.train.txt"
out_fname="/blob/lihe/scigpt/data/material/all_materials_processed.pended.train.txt"

def main():
    with open(fname, "r") as fr:
        with open(out_fname, "w") as fw:
            for line in fr:
                line = line.strip()
                matches = re.findall(r'<i>(\S+)', line)
                ret = []
                for i in range(1, len(matches), 2):
                    ret.extend([matches[i-1]] * int(matches[i]))
                ret = " ".join(ret)
                match = re.search(r"<sg.+>", line)
                sg = match[0]
                ret = f"<material> {ret} {sg} </material>"
                fw.write(ret + "\n")


if __name__ == "__main__":
    main()
