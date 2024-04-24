# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import tqdm
from sfm.logging import logger


def main():
    parser = ArgumentParser()
    parser.add_argument("input_files", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--pad_idx", type=int, default=32000)
    args = parser.parse_args()

    files = [e.strip() for e in args.input_files.split(',')]
    buffer = []
    with open(files[0], 'rb') as fbin:
        newdata = np.frombuffer(fbin.read(), dtype='<H').astype(np.uint16)
        buffer.append(newdata)
        print(files[0], newdata.shape[0])

    for fn in files[1:]:
        with open(fn, 'rb') as fbin:
            newdata = np.frombuffer(fbin.read(), dtype='<H').astype(np.uint16)
            buffer.append(newdata)
            print(fn, newdata.shape[0])
    data = np.concatenate(buffer, dtype=np.uint16)
    del buffer
    token_count = data.shape[0]
    logger.info("number of tokens: {}".format(token_count))
    logger.info("first token loaded: {}, type {}".format(data[:10], data.dtype))
    if token_count % args.seq_len != 0:
        # pad to multiple of seq_len
        data = np.append(data, np.ones(args.seq_len - token_count % args.seq_len, dtype=np.uint16)*args.pad_idx)
    data = data.reshape(-1, args.seq_len)
    logger.info("total tokens: {:,}".format(data.shape[0] * args.seq_len))
    logger.info("shape {}".format(data.shape))
    # shuffle the data by rows
    logger.info("shuffle the data")
    np.random.seed(42)
    np.random.shuffle(data)
    logger.info("save to {}".format(args.output))
    np.save(args.output, data)


if __name__ == "__main__":
    main()
