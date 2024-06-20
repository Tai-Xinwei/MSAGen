# -*- coding: utf-8 -*-
import itertools
from itertools import groupby


class GeneKMerTokenizer:
    def __init__(self, kmer: int = 6):
        # Generate all possible kmers
        nucleotides = ["A", "T", "C", "G"]
        all_kmers = ["".join(p) for p in itertools.product(nucleotides, repeat=kmer)]

        # Assign a unique ID to each kmer
        self.encode_dict = {
            all_kmers[i]: i + 3 for i in range(len(all_kmers))
        }  # Start from 3 to leave space for special tokens
        self.kmer = kmer
        # Special tokens
        self.encode_dict["<unk>"] = 0
        self.encode_dict["<bos>"] = 1
        self.encode_dict["<eos>"] = 2
        self.encode_dict["<pad>"] = 4099
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 0
        self.pad_token_id = 4099

        # Create a reverse dict for decoding
        self.decode_dict = {i: kmer for kmer, i in self.encode_dict.items()}
        # print(self.encode_dict)

    def compress_zeros(self, lst):
        write = 0
        for read in range(len(lst)):
            if lst[read] != self.unk_token_id or (
                write > 0 and lst[write - 1] != self.unk_token_id
            ):
                lst[write] = lst[read]
                write += 1
        return lst[:write]

    def encode(self, sequence):
        # print("########")
        # print(sequence)
        start_index = 0
        end_index = len(sequence)
        if sequence.startswith("<bos>"):
            start_index = 5
        if sequence.endswith("<eos>"):
            end_index -= 5
        sequence = sequence.upper()
        # print(sequence)
        # exit(0)
        result = (
            [self.bos_token_id]
            + [
                self.encode_dict.get(sequence[i : i + self.kmer], self.unk_token_id)
                for i in range(start_index, end_index, self.kmer)
            ]
            + [self.eos_token_id]
        )
        return self.compress_zeros(result)

    def encode_no_bos(self, sequence):
        # print("########")
        # print(sequence)
        start_index = 0
        end_index = len(sequence)
        sequence = sequence.upper()
        # print(sequence)
        # exit(0)
        result = [
            self.encode_dict.get(sequence[i : i + self.kmer], self.unk_token_id)
            for i in range(start_index, end_index, self.kmer)
        ]
        return self.compress_zeros(result)

    def decode(self, ids):
        return "".join([self.decode_dict.get(id, self.unk_token) for id in ids])

    def vocab_size(
        self,
    ):
        return len(self.encode_dict)

    def __len__(self):
        return len(self.encode_dict)

    def encode_sequences(
        self, sequences, padding=True, truncation=True, max_length=16384
    ):
        encoded_sequences = [self.encode_no_bos(sequence) for sequence in sequences]

        if padding:
            max_seq_length = max(len(seq) for seq in encoded_sequences)
            if truncation and max_seq_length > max_length:
                max_seq_length = max_length
            encoded_sequences = [
                seq + [self.pad_token_id] * (max_seq_length - len(seq))
                for seq in encoded_sequences
            ]

        if truncation:
            encoded_sequences = [seq[:max_length] for seq in encoded_sequences]

        return dict(
            input_ids=encoded_sequences,
        )


if __name__ == "__main__":
    tokenizer = GeneKMerTokenizer(6)
    sequence = "<bos>AGCTAGactgatNNNNNNNNNNNNctaacaCTAGCTNNN<eos>"
    encoded = tokenizer.encode(sequence)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
