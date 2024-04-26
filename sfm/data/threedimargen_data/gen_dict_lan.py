# -*- coding: utf-8 -*-
import os

with open("dict.txt", "r", encoding="utf8") as fr:
    all_tok = [line.strip() for line in fr]

# add digits
for i in range(10):
    all_tok.append(str(i))

# add special tokens
all_tok.append("-")
all_tok.append(".")

with open("dict_lan.txt", "w", encoding="utf8") as fw:
    for tok in all_tok:
        print(tok, file=fw)
