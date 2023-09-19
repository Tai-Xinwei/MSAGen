# -*- coding: utf-8 -*-
import multiprocessing as mp
import random
from argparse import ArgumentParser

import numpy as np
import tqdm

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger

random.seed(42)


def file_reader(path, n_workers, line_queue):
    bsz = 1000
    lines = []
    for name in path.split(","):
        name = name.strip()
        if not name:
            continue
        with open(name, "r") as f:
            lines.extend(f.readlines())
            logger.info("Total {:,} lines after adding {}".format(len(lines), name))

    logger.info("total lines: {:,}".format(len(lines)))
    random.shuffle(lines)
    logger.info("shuffled")

    for i in range(0, len(lines), bsz):
        line_queue.put(lines[i : i + bsz])

    for _ in range(n_workers):
        line_queue.put(None)


def worker(input_queue, output_queue, tokenizer_path, seq_len):
    tokenizer = SFMDecTokenizer.from_pretrained(tokenizer_path)
    bsz = 1000
    result = []
    cur = []
    while True:
        lines = input_queue.get()
        if lines is None:
            break

        for line in lines:
            try:
                tokens = tokenizer.tokenize(line)
                tokens = (
                    [tokenizer.bos_token_id]
                    + tokenizer.convert_tokens_to_ids(tokens)
                    + [tokenizer.eos_token_id]
                )
            except:
                # some lines have weird tags that can't be tokenized
                tokens = []

            idx = 0
            while idx < len(tokens):
                to_add = min(seq_len - len(cur), len(tokens) - idx)
                cur.extend(tokens[idx : idx + to_add])
                if len(cur) == seq_len:
                    result.append(cur)
                    cur = []
                idx += to_add

            if len(result) >= bsz:
                output_queue.put(result)
                result = []


    if len(cur) > 0:
        cur.extend([tokenizer.pad_token_id] * (seq_len - len(cur)))
        result.append(cur)

    if len(result) > 0:
        output_queue.put(result)

    output_queue.put(None)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=4096)
    args = parser.parse_args()

    line_queue = mp.Queue(maxsize=1000)
    token_queue = mp.Queue(maxsize=1000)

    file_reader_process = mp.Process(
        target=file_reader, args=(args.input, args.num_workers, line_queue)
    )
    worker_processors = [
        mp.Process(
            target=worker,
            args=(line_queue, token_queue, args.tokenizer_path, args.seq_len),
        )
        for _ in range(args.num_workers)
    ]

    for p in [file_reader_process] + worker_processors:
        p.start()

    bar = tqdm.tqdm()
    # Max 65535. LLAMA has vocab 32k + entity tokens so should be fine
    output = np.empty((0, args.seq_len), dtype=np.uint16)
    token_cnt = 0

    finished_workers = 0
    while True:
        tokens = token_queue.get()
        if tokens is None:
            finished_workers += 1
            if finished_workers == args.num_workers:
                break
        else:
            tokens = np.array(tokens, dtype=np.uint16)
            output = np.append(output, tokens, axis=0)
            bar.update(tokens.shape[0])
            token_cnt += tokens.shape[0] * tokens.shape[1]
            bar.set_description(f"Token count: {token_cnt:,}")

    for p in [file_reader_process] + worker_processors:
        p.join()

    bar.close()

    logger.info("shape {}".format(output.shape))
    logger.info("save to {}".format(args.output))
    np.save(args.output, output)


if __name__ == "__main__":
    main()
