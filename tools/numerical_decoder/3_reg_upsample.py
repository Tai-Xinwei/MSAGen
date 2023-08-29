# -*- coding: utf-8 -*-
filenames = ["file1.txt", "file2.txt", "file3.txt"]

with open("/sfm/ds_dataset/qizhi_numerical/json/molnet_3_reg_up_train.json", 'w') as outfile:
    with open('/sfm/ds_dataset/qizhi_numerical/json/ESOL_train.json') as infile:
        for line in infile:
            for _ in range(4):
                outfile.write(line)
    with open('/sfm/ds_dataset/qizhi_numerical/json/freesolv_train.json') as infile:
        for line in infile:
            for _ in range(7):
                outfile.write(line)
    with open('/sfm/ds_dataset/qizhi_numerical/json/lipo_train.json') as infile:
        for line in infile:
            for _ in range(1):
                outfile.write(line)
