# -*- coding: utf-8 -*-
# %%
old_file = '/blob/lihe/scigpt/data/material/all_materials.pended.shuffle.txt'

# %%
with open(old_file, 'r') as f:
    lines = f.readlines()
# %%
lines[0]
# %%

def reformat(line):
    line = line.strip().split()
    line = line[1:-1] # remove [T] [/T]

    ret = []
    for i in range(0, len(line)-1, 2):
        chem = line[i][3:]
        cnt = int(line[i+1][3:])

        ret.extend([chem] * cnt)

    ret.append(line[-1])

    return '<material>' + ' '.join(ret) + '</material>'

print(lines[0], reformat(lines[0]))

# %%
from tqdm import tqdm
output = []
for line in tqdm(lines):
    output.append(reformat(line))


# %%
output[-1]
# %%
output[-2]
# %%
with open('/blob/shufxi/data/scigpt/CrystalLLM/train.txt', 'w') as f:
    f.write('\n'.join(output))
# %%
