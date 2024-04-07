# -*- coding: utf-8 -*-
# %%
import json
from tqdm import tqdm
from pathlib import Path
# %%
data_home = Path('/blob/guoqing/pistachio_2023Q2_v2_o_smiles_preprocessed/')
# %%
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(l) for l in tqdm(f)]
    return data
# %%
train_data = load_jsonl(data_home / 'train_augmentation_40.jsonl')
# %%
print(len(train_data))
# %%
with open(data_home / 'train_augmentation_40.txt', 'w') as f:
    for d in tqdm(train_data):
        f.write(f'<product>{d["psmiles"]}</product>\t<reactants>{d["rsmiles"]}</reactants>\n')
        f.write(f'<reactants>{d["rsmiles"]}</reactants>\t<product>{d["psmiles"]}</product>\n')

# %%
valid_data = load_jsonl(data_home / 'valid.jsonl')
# %%
with open(data_home / 'valid.txt', 'w') as f:
    for d in tqdm(valid_data):
        f.write(f'<product>{d["psmiles"]}</product>\t<reactants>{d["rsmiles"]}</reactants>\n')
        f.write(f'<reactants>{d["rsmiles"]}</reactants>\t<product>{d["psmiles"]}</product>\n')
# %%
