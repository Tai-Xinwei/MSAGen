# -*- coding: utf-8 -*-
import lmdb
from tqdm import tqdm

print("step 0")
write_env = lmdb.open("/mnt/shiyu/dataset/pubchemqc-dft/merged/S0-msg", map_size=1024 ** 4)
write_txn = write_env.begin(write=True)
print("step 1")
read_env = lmdb.open("/mnt/shiyu/dataset/pubchemqc-dft/merged/S0/")
read_txn = read_env.begin(write=False)
print("step 2")
cnt = 0
for key, graph in tqdm(read_txn.cursor()):
    write_txn.put(key, ''.encode())
    cnt += 1
    if cnt % 1000000 == 0:
        write_txn.commit()
        write_txn = write_env.begin(write=True)
if cnt % 1000000 != 0:
    write_txn.commit()
