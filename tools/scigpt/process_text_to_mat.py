# -*- coding: utf-8 -*-
# %%
raw_data_path = '/blob/v-zequnliu/tamgent2/material_new/'

text_file = raw_data_path + 'train_text.txt'
mat_file = raw_data_path + 'train_smi.txt'
# %%
with open(text_file, 'r') as f:
    text_lines = f.readlines()
# %%
with open(mat_file, 'r') as f:
    mat_lines = f.readlines()
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

# %%
output = []
for text_line, mat_line in zip(text_lines, mat_lines):
    text_line = text_line.strip()
    mat_line = mat_line.strip()

    if text_line == '' or mat_line == '':
        continue

    output.append(f'{text_line} The material is {reformat(mat_line)}.')


# %%
len(output)
# %%
with open('/blob/shufxi/data/scigpt/text2material/train.txt', 'w') as f:
    for line in output:
        f.write(line + '\n')
# %%
