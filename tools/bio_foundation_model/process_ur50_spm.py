# -*- coding: utf-8 -*-
import sentencepiece as spm
import multiprocessing
from multiprocessing import Pool
import tqdm
import argparse
import struct
import random
import numpy as np
import os

def tokenize(line):
    if getattr(tokenize, "sp", None) is None:
        setattr(tokenize, "sp", spm.SentencePieceProcessor(model_file="/blob/shufxi/data/biofm/ur50bpe/ur50bpe.model"))

    sp = getattr(tokenize, "sp")
    tokens = sp.encode(line, out_type=int)
    return [1] + tokens + [2] # [bos] + tokens + [eos]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--pad", type=int, default=3)

    args = parser.parse_args()

    lines = []
    with open(args.input, 'r') as f:
        for line in f:
            lines.append(line.strip())

    random.seed(42)
    random.shuffle(lines)
    print('Total lines: ', len(lines))

    buffer = []
    n_tokens = 0
    too_long = 0
    with Pool(multiprocessing.cpu_count()) as p, open(args.output + '.bin', 'wb') as fbin:
        for tokens in tqdm.tqdm(p.imap(tokenize, lines), total=len(lines)):
            if len(tokens) > args.seq_len:
                too_long += 1
                tokens = tokens[:args.seq_len]

            if len(buffer) + len(tokens) >= args.seq_len:
                # if we cannot put tokens in buffer, we need to pad it
                assert len(buffer) <= args.seq_len

                buffer.extend([args.pad] * (args.seq_len - len(buffer)))
                assert len(buffer) == args.seq_len
                fbin.write(struct.pack('<' + 'H'*args.seq_len, *buffer))
                n_tokens += len(buffer)
                buffer = []

            buffer.extend(tokens)

        if len(buffer) > 0:
            buffer.extend([args.pad] * (args.seq_len - len(buffer)))
            assert len(buffer) == args.seq_len
            for i in buffer:
                assert type(i) == int, i
                assert i < 65536
            fbin.write(struct.pack('<' + 'H'*args.seq_len, *buffer))
            n_tokens += len(buffer)
            buffer = []
    print('Total tokens: ', n_tokens)
    print('Too long: ', too_long)
    with open(args.output + '.bin', 'rb') as fbin:
        data = np.frombuffer(fbin.read(), dtype='<H').astype(np.uint16)
        assert data.shape[0] % args.seq_len == 0
        data = data.reshape(-1, args.seq_len)
        print('Data shape: ', data.shape)
        np.save(args.output, data)

    os.remove(args.output + '.bin')

if __name__ == "__main__":
    main()
