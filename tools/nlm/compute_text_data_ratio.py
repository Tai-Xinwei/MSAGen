# -*- coding: utf-8 -*-
import sys

text_token_per_file = {
    "v5_processed_text_train.npy.lmdb": 14_398_813_720,
    "v5_processed_other_text_train.npy.lmdb": 6_248_042_664,
}

other_per_file = {
    "v5_processed_general_cmpd_train.npy.lmdb": 4_180_601_029,
    "v5_processed_textSMILES_train.npy.lmdb": 3_044_142_472,
    # "v5_processed_wrap_train.npy.lmdb": 3_974_260_183,
    "v5_processed_dna_six_protein_text_train.npy.lmdb": 3_974_260_183,
    "v5_processed_material_text_train.npy.lmdb": 207_657_885,
    "v5_processed_material_train.npy.lmdb": 20_746_261,
    "v5_processed_protein_dna_six_train.npy.lmdb": 1_246_570_706,
    "v5_processed_protein_smile_text_train.npy.lmdb": 28_968_845,
    "v5_processed_protein_text_train.npy.lmdb": 1_348_115_830,
    "v5_processed_rna_train.npy.lmdb": 27_456_079_159,
}
protein_token_per_file = {
    "v5_processed_protein_train.npy.lmdb": 65_232_240_143,
}
dna_token_per_file = {
    "v5_processed_dna_six_train.npy.lmdb": 3_507_277_123,
}


def main():
    alpha = float(sys.argv[1])
    prot_raito = float(sys.argv[2])
    dna_raito = float(sys.argv[3])
    other_raito = float(sys.argv[4])
    files = list(
        text_token_per_file.keys()
        | other_per_file.keys()
        | protein_token_per_file.keys()
        | dna_token_per_file.keys()
    )

    total_text_tokens = 0
    total_dna_tokens = 0
    total_protein_tokens = 0
    total_other_tokens = 0
    for file in files:
        if file in other_per_file:
            total_other_tokens += other_per_file[file]
        elif file in text_token_per_file:
            total_text_tokens += text_token_per_file[file]
        elif file in protein_token_per_file:
            total_protein_tokens += protein_token_per_file[file]
        elif file in dna_token_per_file:
            total_dna_tokens += dna_token_per_file[file]

    print(f"#text_tokens:\t{total_text_tokens:,}")
    print(f"#dna_tokens:\t{total_dna_tokens:,}")
    print(f"#protein_tokens:\t{total_protein_tokens:,}")
    print(f"#other_tokens:\t{total_other_tokens:,}")
    print(
        f"#tokens:\t{total_text_tokens + total_other_tokens+total_dna_tokens+total_protein_tokens:,}"
    )
    print(f"Alpha: {alpha}")
    print(f"prot_raito: {prot_raito}")
    print(f"dna_raito: {dna_raito}")
    print(f"other_raito: {other_raito}")
    ret = []
    for file in files:
        if file in other_per_file:
            weight = other_raito * other_per_file[file] / total_other_tokens
        elif file in text_token_per_file:
            weight = alpha * text_token_per_file[file] / total_text_tokens
        elif file in protein_token_per_file:
            weight = prot_raito * protein_token_per_file[file] / total_protein_tokens
        elif file in dna_token_per_file:
            weight = dna_raito * dna_token_per_file[file] / total_dna_tokens
        ret.append(f"{weight:.12f}")
    for file, weight in zip(files, ret):
        print(f"{file:50}: {weight}")

    with open("/tmp/data_ratio.txt", "w") as f:
        f.write(",".join(ret))

    with open("/tmp/train_data_path.txt", "w") as f:
        f.write(",".join(files))


if __name__ == "__main__":
    main()
