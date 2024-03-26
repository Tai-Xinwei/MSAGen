# -*- coding: utf-8 -*-
#%%
from matbench.bench import MatbenchBenchmark
# %%
tasks = [
        'matbench_expt_gap',
        'matbench_expt_is_metal',
        'matbench_glass',
        # 'matbench_steels'
    ]
mb = MatbenchBenchmark(
    autoload=False,
    subset=tasks
)
# %%
# for each dataset, show example input/output

for task in mb.tasks:
    task.load()
    print(task.dataset_name)

    print(task.df.head())
# %%
import re

re_tok = re.compile(r'([A-Z][a-z]?|\d+\.?\d*|\(|\))')
def tokenize_mat(form):
    # split the formula into elements
    # Ag(AuS)2 -> [Ag, (, Au, S, ), 2]
    # Ag0.5Ge1Pb1.75S4 -> [Ag, 0.5, Ge, 1, Pb, 1.75, S, 4]
    return re_tok.findall(form)

print(tokenize_mat('Ag(AuS)2'))
print(tokenize_mat('Ag0.5Ge1Pb1.75S4'))
print(tokenize_mat('Ag(W3Br7)2'))


# %%
from fractions import Fraction
from collections import defaultdict

def parse_formula(tokens):
    stack = []
    idx = 0

    while idx < len(tokens):
        if tokens[idx] == '(':
            stack.append('(')
        elif tokens[idx] == ')':
            group = defaultdict(Fraction)
            group_replicate = Fraction(tokens[idx+1])
            while stack[-1] != '(':
                last = stack.pop()
                for k in last:
                    group[k] += last[k]
            for k in group:
                group[k] *= group_replicate
            idx += 1
            stack.pop()
            stack.append(group)
        else:
            try:
                replicate = Fraction(tokens[idx])
                stack[-1][tokens[idx-1]] = replicate
            except:
                stack.append({tokens[idx]: Fraction(1)})
        idx += 1

    ret = defaultdict(Fraction)
    # print(stack)
    for group in stack:
        for k in group:
            ret[k] += group[k]
    return ret

print(parse_formula(tokenize_mat('Ag2')))
print(parse_formula(tokenize_mat('Ag(AuS)2')))
print(parse_formula(tokenize_mat('Ag0.5Ge1Pb1.75S4')))
# %%
import math
def make_all_frac_int(fracs: list[Fraction]):
    # find the LCM of all denominators
    lcm = fracs[0].denominator
    for f in fracs:
        lcm = lcm * f.denominator // math.gcd(lcm, f.denominator)
    # multiply all fractions by the LCM
    return [int(f * lcm) for f in fracs]

def formula_to_str(form, max_token=4000):
    tokens = tokenize_mat(form)
    parsed = parse_formula(tokens)

    int_fracs = make_all_frac_int(list(parsed.values()))
    ret = []
    for k, v in zip(parsed.keys(), int_fracs):
        if v == 1:
            ret.append(k)
        else:
            ret.extend([k] * v)

    ret.sort()
    return ' '.join(ret[:max_token])

formula_to_str('Ag(AuS)2')
formula_to_str('Ag0.5Ge1Pb1.75S4')
# %%

from pathlib import Path
output_base = Path('/blob/shufxi/data/scigpt/matbench')

for task in mb.tasks:
    task.load()
    print(task.dataset_name)

    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        with open(output_base / f'{task.dataset_name}_{fold}_train.txt', 'w') as ft, \
            open(output_base / f'{task.dataset_name}_{fold}_val.txt', 'w') as fv:
            for idx, (i, o) in enumerate(zip(train_inputs, train_outputs)):
                formula = formula_to_str(i)
                output = f'<material>{formula}</material>\t{o}\n'
                if idx % 20 > 0:
                    ft.write(output)
                else:
                    fv.write(output)
# %%
task.validation
# %%
