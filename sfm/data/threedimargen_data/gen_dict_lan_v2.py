# -*- coding: utf-8 -*-
import os

with open("dict.txt", "r", encoding="utf8") as fr:
    all_tok = [line.strip() for line in fr]

# add number groups
for i in range(1000):
    all_tok.append(format(i, "03d"))

# add special tokens
all_tok.append("-")
all_tok.append(".")

with open("dict_lan_v2.txt", "w", encoding="utf8") as fw:
    for tok in all_tok:
        print(tok, file=fw)
