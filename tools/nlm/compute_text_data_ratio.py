# -*- coding: utf-8 -*-
import sys

text_token_per_file = {
    'v5_processed_text_train.npy.lmdb': 14_398_813_720
}

sci_token_per_file = {
    'v5_processed_general_cmpd_train.npy.lmdb': 4_180_601_029,
    'v5_processed_textSMILES_train.npy.lmdb': 3_044_142_472,
    'v5_processed_wrap_train.npy.lmdb': 3_974_260_183,
    'v5_processed_material_text_train.npy.lmdb': 207_657_885,
    'v5_processed_material_train.npy.lmdb': 20_746_261,
    'v5_processed_protein_train.npy.lmdb': 65_232_240_143,
    'v5_processed_protein_dna_train.npy.lmdb': 4_103_662_326,
    'v5_processed_protein_smile_text_train.npy.lmdb': 28_968_845,
    'v5_processed_protein_text_train.npy.lmdb': 1_348_115_830,
    'v5_processed_dna_train.npy.lmdb': 19_826_959_218,
    'v5_processed_rna_train.npy.lmdb': 27_456_079_159,
}

def main():
    alpha = float(sys.argv[1])
    prot_raito = float(sys.argv[2])
    files = list(text_token_per_file.keys() | sci_token_per_file.keys())

    total_text_tokens = 0
    total_sci_tokens = 0

    for file in files:
        if file in text_token_per_file:
            total_text_tokens += text_token_per_file[file]
        else:
            if file == 'v5_processed_protein_train.npy.lmdb':
                total_sci_tokens += prot_raito*sci_token_per_file[file]
            else:
                total_sci_tokens += sci_token_per_file[file]

    print(f"#text_tokens:\t{total_text_tokens:,}")
    print(f"#sci_tokens:\t{total_sci_tokens:,}")
    print(f"#tokens:\t{total_text_tokens + total_sci_tokens:,}")
    print(f"Alpha: {alpha}")
    print(f"prot_raito: {prot_raito}")

    ret = []
    for file in files:
        if file in text_token_per_file:
            weight = alpha * text_token_per_file[file] / total_text_tokens
        else:
            if file == 'v5_processed_protein_train.npy.lmdb':
                weight = (1 - alpha) * prot_raito*sci_token_per_file[file] / total_sci_tokens
            else:
                weight = (1 - alpha) * sci_token_per_file[file] / total_sci_tokens
        ret.append(f"{weight:.12f}")
    for file, weight in zip(files, ret):
        print(f"{file:50}: {weight}")

    with open('/tmp/data_ratio.txt', 'w') as f:
        f.write(','.join(ret))

    with open('/tmp/train_data_path.txt', 'w') as f:
        f.write(','.join(files))


if __name__ == '__main__':
    main()
