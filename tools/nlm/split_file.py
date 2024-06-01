# -*- coding: utf-8 -*-
from multiprocessing import Pool, cpu_count
import os


def worker(args):
    in_filename, out_filename, start, end = args
    with open(in_filename, "rb") as f:
        f.seek(start)
        lines = []
        while f.tell() < end:
            line = f.readline()
            lines.append(line)
    with open(out_filename, "wb") as f:
        f.writelines(lines)


def split_large_file_multi(in_filename, out_path, num_files):
    size = os.path.getsize(in_filename)
    chunk_size = size // num_files

    offsets = [0]
    with open(in_filename, "rb") as f:
        for _ in range(num_files - 1):
            f.seek(chunk_size, 1)
            f.readline()  # 跳过一个完整的行
            offsets.append(f.tell())
    offsets.append(size)

    with Pool(cpu_count()) as p:
        p.map(
            worker,
            [
                (in_filename, os.path.join(out_path, f"output_{i}.txt"), start, end)
                for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:]))
            ],
        )


split_large_file_multi(
    "/home/v-zekunguo/zekun_data/scidata/raw_data/valid.c4",
    "/home/v-zekunguo/zekun_data/scidata/raw_data/split",
    16,
)


# def count_lines(filename):
#     with open(filename, "r") as file:
#         return sum(1 for line in file)


# result = 0
# for file_name in os.listdir("/home/v-zekunguo/zekun_data/scidata/raw_data/split"):
#     result += count_lines(
#         os.path.join("/home/v-zekunguo/zekun_data/scidata/raw_data/split", file_name)
#     )
# print(result)
# exit(0)
# 使用方法
# split_large_file_multi(
#     "/home/v-zekunguo/zekun_data/scidata/raw_data/valid.c4",
#     "/home/v-zekunguo/zekun_data/scidata/raw_data/split",
#     16,
# )
