# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import tqdm

import os
import multiprocessing as mp

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger

import struct

import random

def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = SFMDecTokenizer.from_pretrained(tokenizer_path)

def tokenize(line):
    global tokenizer
    try:
        tokens = tokenizer.tokenize(line)
        tokens = (
            [tokenizer.bos_token_id]
            + tokenizer.convert_tokens_to_ids(tokens)
            + [tokenizer.eos_token_id]
        )
        return tokens
    except:
        # some lines have weird tags that can't be tokenized
        return []


# def worker(input_queue, output_queue, tokenizer_path, seq_len):
#     tokenizer = SFMDecTokenizer.from_pretrained(tokenizer_path)
#     bsz = 1000
#     result = []
#     cur = []
#     while True:
#         try:
#             lines = input_queue.get(timeout=600)
#         except Exception as e:
#             logger.exception(e)
#             break

#         if lines is None:
#             break

#         for line in lines:
#             try:
#                 tokens = tokenizer.tokenize(line)
#                 tokens = (
#                     [tokenizer.bos_token_id]
#                     + tokenizer.convert_tokens_to_ids(tokens)
#                     + [tokenizer.eos_token_id]
#                 )
#             except:
#                 # some lines have weird tags that can't be tokenized
#                 tokens = []

#             idx = 0
#             while idx < len(tokens):
#                 to_add = min(seq_len - len(cur), len(tokens) - idx)
#                 cur.extend(tokens[idx : idx + to_add])
#                 if len(cur) == seq_len:
#                     result.append(cur)
#                     cur = []
#                 idx += to_add

#             if len(result) >= bsz:
#                 output_queue.put(result)
#                 result = []


#     if len(cur) > 0:
#         cur.extend([tokenizer.pad_token_id] * (seq_len - len(cur)))
#         result.append(cur)

#     if len(result) > 0:
#         output_queue.put(result)

#     output_queue.put(None)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    lines = []
    for file in args.input.split(','):
        with open(file.strip(), 'r') as f:
            lines.extend(f.readlines())
            logger.info("Total {:,} lines after reading {}".format(len(lines), file.strip()))

    logger.info("total lines: {:,}".format(len(lines)))
    random.seed(42)
    random.shuffle(lines)
    logger.info("shuffled")

    loged = False
    with open(args.output+'.bin', 'wb') as fbin:
        with mp.Pool(args.num_workers, initializer=init_tokenizer, initargs=(args.tokenizer_path,)) as pool:
            for tokens in tqdm.tqdm(pool.imap(tokenize, lines), total=len(lines)):
                if len(tokens) > 0:
                    if not loged:
                        logger.info("first token: {}".format(tokens[:10]))
                        loged = True
                    fbin.write(struct.pack('<'+'H'*len(tokens), *tokens))

    logger.info("saved to {}".format(args.output+'.bin'))

    del lines

    with open(args.output+'.bin', 'rb') as fbin:
        data = np.frombuffer(fbin.read(), dtype='<H').astype(np.uint16)
        logger.info("first token loaded: {}, type {}".format(data[:10], data.dtype))
        token_count = data.shape[0]
        if token_count % args.seq_len != 0:
            # pad to multiple of seq_len
            data = np.append(data, np.ones(args.seq_len - token_count % args.seq_len, dtype=np.uint16)*32000)
        data = data.reshape(-1, args.seq_len)
        logger.info("total tokens: {:,}".format(data.shape[0] * args.seq_len))
        logger.info("shape {}".format(data.shape))
        logger.info("save to {}".format(args.output))
        np.save(args.output, data)
    # delete binary file

    os.remove(args.output+'.bin')


    # output = np.empty((0, args.seq_len), dtype=np.uint16)
    # cur = []
    # tokenizer = SFMDecTokenizer.from_pretrained(args.tokenizer_path)

    # for line in tqdm.tqdm(lines):
    #     tokens = tokenize(line.strip(), tokenizer)
    #     cur.extend(tokens)
    #     while len(cur) >= args.seq_len:
    #         new_row = cur[:args.seq_len]
    #         # ret.append(new_row)
    #         # ret.append(np.array(new_row, dtype=np.uint16))
    #         output = np.append(output, np.array([new_row], dtype=np.uint16), axis=0)
    #         cur = cur[args.seq_len:]
    # if len(cur) > 0:
    #     # init_tokenizer(args.tokenizer_path)
    #     # global tokenizer
    #     cur.extend([tokenizer.pad_token_id] * (args.seq_len - len(cur)))
    #     # ret.append(cur)
    #     # ret.append(np.array(cur, dtype=np.uint16))
    #     output = np.append(output, np.array([cur], dtype=np.uint16), axis=0)

    # line_queue = mp.Queue()
    # token_queue = mp.Queue()

    # file_reader_process = mp.Process(
    #     target=file_reader, args=(args.input, args.num_workers, line_queue)
    # )
    # worker_processors = [
    #     mp.Process(
    #         target=worker,
    #         args=(line_queue, token_queue, args.tokenizer_path, args.seq_len),
    #     )
    #     for _ in range(args.num_workers)
    # ]

    # for p in [file_reader_process] + worker_processors:
    #     p.start()

    # bar = tqdm.tqdm()
    # # Max 65535. LLAMA has vocab 32k + entity tokens so should be fine
    # output = np.empty((0, args.seq_len), dtype=np.uint16)
    # token_cnt = 0

    # finished_workers = 0
    # while True:
    #     tokens = token_queue.get()
    #     if tokens is None:
    #         finished_workers += 1
    #         logger.info("finished workers: {}".format(finished_workers))
    #         if finished_workers == args.num_workers:
    #             break
    #     else:
    #         tokens = np.array(tokens, dtype=np.uint16)
    #         output = np.append(output, tokens, axis=0)
    #         bar.update(tokens.shape[0])
    #         token_cnt += tokens.shape[0] * tokens.shape[1]
    #         bar.set_description(f"Token count: {token_cnt:,}")

    # for p in [file_reader_process] + worker_processors:
    #     p.join()

    # bar.close()

    # logger.info("total tokens: {:,}".format(len(ret) * args.seq_len))
    # output = np.array(ret, dtype=np.uint16)
    # output = np.stack(ret, axis=0)
    # logger.info("total tokens: {:,}".format(output.shape[0] * args.seq_len))
    # logger.info("shape {}".format(output.shape))
    # logger.info("save to {}".format(args.output))
    # np.save(args.output, output)


if __name__ == "__main__":
    main()
