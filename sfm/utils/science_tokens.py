# -*- coding: utf-8 -*-

import os

# These are tags to wrap science tokens
SCIENCE_TAG_TOKENS = [
    "<mol>",
    "</mol>",
    "<num>",  # 32003
    "</num>",
    "<material>",
    "</material>",
    "<protein>",
    "</protein>",
    "<dna>",
    "</dna>",
    "<rna>",
    "</rna>",
    "<ans>",
    "</ans>",
    "<xyz>",
    "</xyz>",
    "<product>",
    "</product>",
    "<reactants>",  # Use plural as it may wrap multiple reactants that separated by "."
    "</reactants>",
    "<antibody>",
    "</antibody>",
    "<fragA>",
    "</fragA>",
    "<fragB>",
    "</fragB>",
    "<reagent>",
    "</reagent>",
    "<cf1>",
    "</cf1>",
    "<cf2>",
    "</cf2>",
    "<fcf>",
    "</fcf>",
    "<dna6mer>",
    "</dna6mer>",
]

# may need to use in future
for i in range(45):
    SCIENCE_TAG_TOKENS.append(f"<dummy{i}>")
    SCIENCE_TAG_TOKENS.append(f"</dummy{i}>")


# These are science entities, such as elements in SMILES.
SCIENCE_TOKENS = []
file_path = os.path.join(os.path.dirname(__file__), "science_tokens.txt")
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        SCIENCE_TOKENS.append(line.strip())
