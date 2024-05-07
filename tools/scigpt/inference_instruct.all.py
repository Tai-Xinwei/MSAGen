# -*- coding: utf-8 -*-
from inference_module import SFMGenerator
from glob import glob
import pickle as pkl
from tqdm import tqdm
import os

ckpt_home = '/home/yinxia/blob1.v2/yinxia/scigpt/7bv3/overall_instruct_20240430/run2/global_step24128'
generator = SFMGenerator(ckpt_home)

FF = glob(r'/home/yinxia/blob1.v2/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240429/test*')
FF.append(r'/home/yinxia/workdir/SFM_framework_yinxia_scigptv3/tools/scigpt/i2s_i.txt')
FF.append(r'/home/yinxia/workdir/SFM_framework_yinxia_scigptv3/tools/scigpt/s2i_s.txt')
FF.append(r'/home/yinxia/blob1.v2/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240429/sfmdata.prot.test.tsv')

for fn in FF:
    print(fn)
    with open(fn, 'r', encoding='utf8') as fr:
        all_lines = [e.strip() for e in fr]

    buffer = []
    for idx, test_sample in tqdm(enumerate(all_lines),total=len(all_lines)):
        q = test_sample.split('\t')[0].strip()
        r0 = generator.chat(q, do_sample=False)
        r1 = generator.chat(q, do_sample=True)
        buffer.append((test_sample, r0, r1))

    basename = os.path.basename(fn)
    fn_out = basename + '.response.pkl'

    with open(fn_out, 'wb') as fw:
        pkl.dump(buffer, fw)
