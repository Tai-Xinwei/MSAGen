# -*- coding: utf-8 -*-
import lmdb
import os
import lmdb
import pickle
import zlib
from tqdm import tqdm
import multiprocessing
import sentencepiece as spm
from itertools import groupby
from multiprocessing import Pool
import functools


vocab = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<mask>': 31}


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))

def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def tokenize(line):
    if getattr(tokenize, "sp", None) is None:
        setattr(tokenize, "sp", spm.SentencePieceProcessor(model_file="/home/lijuwu/protein/lijuwu/biofm/data/Tokenizer_15k.model"))

    sp = getattr(tokenize, "sp")
    # line = ''.join(line)
    tokens = sp.encode(line, out_type=int)
    return tokens # [bos] + tokens + [eos]


def data_process(data):
    data = data.strip()
    tokens = [0] + [vocab[tok] for tok in data] + [2]
    bpe_token = tokenize(data)

    return tokens, bpe_token


def main():

    with open("/home/lijuwu/protein/lijuwu/biofm/data/Tokenizer_15k.vocab") as file:
        vocab_lines = file.readlines()
        vocab_lines = [word.strip().split(' ')[0] for word in vocab_lines]

    bpe_vocab = {}
    bpe_idx2word = {}
    for idx, word in enumerate(vocab_lines):
        bpe_idx2word[idx] = word
        bpe_vocab[word] = idx

    # process raw data
    # raw_data = pickle.load(open("/home/lijuwu/biofm_data_tmp/ur50_raw_data.pkl", "rb"))
    raw_data = open('/home/lijuwu/biofm_data_tmp/uniref50_2023_05.shorten.train.seqs', 'r').readlines()
    print("Finish loading raw data")

    buffer_len = 0
    sequence_length = 1536
    buffer = []
    bpe_buffer = []
    last_tokens = None
    metadata = {}
    names = []
    sizes = []
    token_list = []
    bpe_token_list = []

    # read raw data and do tokenization
    print('Processing raw data')
    idx = 0
    with Pool(processes=120) as pool:
        length = len(raw_data)
        iter = pool.imap(data_process, raw_data)
        for idx, results in tqdm(enumerate(iter), total=length):
            tokens, bpe_token = results
            if len(tokens) > 1536:
                continue

            token_list.append(tokens)
            bpe_token_list.append(bpe_token)

    del raw_data

    # write to lmdb
    write_file = '/home/lijuwu/biofm_data_tmp/ur50_23_msa_ppi_bpe_pack1536.lmdb'
    write_env = lmdb.open(write_file, map_size=1536 ** 4)
    write_txn = write_env.begin(write=True)

    # Add the aa and BPE data to the output LMDB file
    print('Add the aa and BPE data to the output LMDB file')
    for tokens, bpe_token in tqdm(zip(token_list, bpe_token_list), total=len(token_list)):
        # pack 1536
        buffer.extend(tokens)

        bpe_token_long = []
        for token in bpe_token:
            token_len = len(bpe_idx2word[token])
            bpe_token_long.extend([token] * token_len)

        bpe_token_long = [0] + bpe_token_long + [2]

        bpe_buffer.extend(bpe_token_long)

        buffer_len += len(tokens)
        if buffer_len >= sequence_length:

            if buffer_len == sequence_length:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer
                dict_buffer['bpe_seq'] = bpe_buffer
                write_txn.put(f"{idx}".encode(), pickle.dumps(dict_buffer))
                buffer_len = 0
                buffer = []
                bpe_buffer = []

                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
            else:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer[:sequence_length]
                dict_buffer['bpe_seq'] = bpe_buffer[:sequence_length]
                write_txn.put(f"{idx}".encode(), pickle.dumps(dict_buffer))
                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
                buffer_len = len(tokens)
                buffer = tokens
                bpe_buffer = bpe_token_long

    # Add the MSA data to the output LMDB file
    print('Adding MSA data')
    msa_path = '/home/lijuwu/biofm_data_tmp/hhba4ms_msa_seqs.txt'
    msa_data = []
    with open(msa_path, 'r') as f:
        for line in f:
            msa_sequence = line.strip()
            msa_sequence_idx = [vocab[aa] for aa in msa_sequence]
            if len(msa_sequence_idx) > 1536:
                continue
            msa_data.append(msa_sequence_idx)

    buffer = []
    bpe_buffer = []
    buffer_len = 0
    for token_list in tqdm(msa_data):
        tokens = [0] + token_list + [2]
        bpe_tokens = [0] + [1]*len(token_list) + [2]
        if len(tokens) > 1536:
            continue

        buffer.extend(tokens)
        bpe_buffer.extend(bpe_tokens)
        buffer_len += len(tokens)
        if buffer_len >= sequence_length:
            if buffer_len == sequence_length:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer
                dict_buffer['bpe_seq'] = bpe_buffer
                write_txn.put(f"{idx}".encode(), pickle.dumps(dict_buffer))
                buffer_len = 0
                buffer = []
                bpe_buffer = []
                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
            else:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer[:sequence_length]
                dict_buffer['bpe_seq'] = bpe_buffer[:sequence_length]
                write_txn.put(f"{idx}".encode(), pickle.dumps(dict_buffer))
                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
                buffer_len = len(tokens)
                buffer = tokens
                bpe_buffer = bpe_tokens

    # pbar = tqdm.tqdm(total=len(msa_data), desc='Adding MSA data')
    # for msa_sequence in msa_data:
    #     msa_key = f"{idx}".encode()
    #     # msa_value = obj2bstr(msa_sequence)
    #     msa_sequence_idx = [vocab[aa] for aa in msa_sequence]
    #     msa_bpe_value = [1] * len(msa_sequence_idx)
    #     msa_dict_value = {'aa_seq': msa_sequence_idx, 'bpe_seq': msa_bpe_value}
    #     write_txn.put(msa_key, pickle.dumps(msa_dict_value))
    #     # add name and values to the metadata
    #     names.append(msa_key.decode())
    #     sizes.append(len(msa_sequence_idx))
    #     pbar.update(1)

    # pbar.close()
    print('msa data added finished')

    # Add the PPI data to the output LMDB file
    print('Adding ppi data')
    ppi_path = '/home/lijuwu/biofm_data_tmp/ppi_all.txt'
    ppi_data = []
    with open(ppi_path, 'r') as f:
        for line in f:
            ppi_sequence = line.strip()
            ppi_sequence_idx = [vocab[aa] for aa in ppi_sequence]   # convert string to ids
            if len(ppi_sequence_idx) > 1536:
                continue
            ppi_data.append(ppi_sequence_idx)

    buffer = []
    bpe_buffer = []
    buffer_len = 0
    for token_list in tqdm(ppi_data):
        tokens = [0] + token_list + [2]
        bpe_tokens = [0] + [1]*len(token_list) + [2]
        if len(tokens) > 1536:
            continue

        buffer.extend(tokens)
        bpe_buffer.extend(bpe_tokens)
        buffer_len += len(tokens)
        if buffer_len >= sequence_length:
            if buffer_len == sequence_length:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer
                dict_buffer['bpe_seq'] = bpe_buffer
                write_txn.put(f"{idx}".encode(), pickle.dumps(dict_buffer))
                buffer_len = 0
                buffer = []
                bpe_buffer = []
                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
            else:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer[:sequence_length]
                dict_buffer['bpe_seq'] = bpe_buffer[:sequence_length]
                write_txn.put(f"{idx}".encode(), pickle.dumps(dict_buffer))
                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
                buffer_len = len(tokens)
                buffer = tokens
                bpe_buffer = bpe_tokens

    # pbar = tqdm.tqdm(total=len(ppi_data), desc='Adding PPI data')
    # for idx, ppi_sequence in enumerate(ppi_data):
    #     ppi_key = f"ppi_{idx}".encode()
    #     # ppi_value = obj2bstr(ppi_sequence)
    #     ppi_sequence_idx = [vocab[aa] for aa in ppi_sequence]   # convert string to ids
    #     ppi_bpe_value = [1] * len(ppi_sequence_idx)
    #     ppi_dict_value = {'aa_seq': ppi_sequence_idx, 'bpe_seq': ppi_bpe_value}
    #     write_txn.put(ppi_key, pickle.dumps(ppi_dict_value))
    #     # add name and values to the metadata
    #     names.append(ppi_key.decode())
    #     sizes.append(len(ppi_sequence_idx))
    #     pbar.update(1)

    # pbar.close()
    print('ppi data added finished')

    metadata['prot_accessions'] = names
    metadata['lengths'] = sizes

    write_txn.put("metadata".encode(), obj2bstr(metadata))
    write_txn.commit()
    print(f"Finish processing {write_file}")

    write_env.close()


if __name__ == "__main__":
    main()
