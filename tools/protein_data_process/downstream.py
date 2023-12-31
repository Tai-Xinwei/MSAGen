# -*- coding: utf-8 -*-
import csv
from typing import List, Union

from pathlib import Path
from sfm.logging import logger
import pandas as pd
import commons
from tqdm import tqdm
from joblib import Parallel, delayed
import datetime
import lmdb
import shutil
from collections import defaultdict
import pickle
import numpy as np

from torchdrug import data, utils


def load_sequence(sequences, targets, names=None, **kwargs):
    num_sample = len(sequences)
    for field, target_list in targets.items():
        if len(target_list) != num_sample:
            raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                "Expect %d but found %d" % (field, num_sample, len(target_list)))

    kwargs = kwargs
    data = []
    sequences = tqdm(sequences, "Constructing proteins from sequences")
    for i, sequence in enumerate(sequences):
        if isinstance(sequence, list) and len(sequence) == 1:
            sequence = sequence[0]
        if isinstance(sequence, list):
            protein = [commons.Protein.from_sequence(seq, name=names[i] if names else str(i), target=[targets[field][i] for field in targets], **kwargs) for seq in sequence]
        else:
            protein = commons.Protein.from_sequence(sequence, name=names[i] if names else str(i), target=[targets[field][i] for field in targets], **kwargs)
        data.append(protein)
    return data


def load_lmdbs(lmdb_files, sequence_field="primary", target_fields=None, number_field="num_examples", **kwargs):
    # if target_fields is not None:
    #     target_fields = set(target_fields)
    if isinstance(sequence_field, str):
        sequence_field = [sequence_field]
    sequences = []
    num_samples = []
    targets = defaultdict(list)
    for lmdb_file in lmdb_files:
        env = lmdb.open(str(lmdb_file), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_sample = pickle.loads(txn.get(number_field.encode()))
            for i in range(num_sample):
                item = pickle.loads(txn.get(str(i).encode()))
                # sequences.append(item[sequence_field])
                sequences.append([item[field] for field in sequence_field])
                if target_fields:
                    for field in target_fields:
                        value = item[field]
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        targets[field].append(value)
            num_samples.append(num_sample)
    data = load_sequence(sequences, targets, **kwargs)
    return data, num_samples


def init_metadata(directory: Path, comment: str=""):
    metadata = {
        "sizes": [],
        "keys": [],
        "comment": f"Created time: {str(datetime.datetime.now())}\n"
        f"Base directory: {str(directory)}\n" + comment,
    }
    return metadata

def write_lmdb_metadata(metadata: dict, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        txn.put("__metadata__".encode(), commons.obj2bstr(metadata))


def write_lmdb_datachunk(prot_lst: List[commons.Protein], env: lmdb.Environment):
    sizes, names = [], []
    with env.begin(write=True) as txn:
        for prot in prot_lst:
            if isinstance(prot, list):
                # keys: {'name', 'size', 'aa', 'pos', 'pos_mask', 'ang', 'ang_mask', 'target'}
                newdict = {'name': '--'.join([p['name'] for p in prot]),
                           'size': sum([p['size'] for p in prot]),
                           'aa': [p['aa'] for p in prot],
                           'pos': [p['pos'] for p in prot],
                           'pos_mask': [p['pos_mask'] for p in prot],
                           'ang': [p['ang'] for p in prot],
                           'ang_mask': [p['ang_mask'] for p in prot],
                           'target': prot[0]['target']}
                key = newdict['name'].encode()
                bstr = commons.obj2bstr(newdict)
                txn.put(key, bstr)
                sizes.append(newdict['size'])
                names.append(newdict['name'])
            else:
                key = prot['name'].encode()
                bstr = commons.obj2bstr(prot)
                txn.put(key, bstr)
                sizes.append(prot['size'])
                names.append(prot['name'])
    return sizes, names



def process_sequence_only_dataset(url, md5, splits, target_fields, download_path, processed_path, dataset_name, **kwargs):
    download_path = Path(download_path)
    if not download_path.exists():
        download_path.mkdir(parents=True)
    processed_path = Path(processed_path)

    zip_file = commons.Protein.download(url, download_path, md5=md5)
    data_path = commons.Protein.extract(zip_file)
    lmdb_files = [data_path / f"{dataset_name}_{split}.lmdb" for split in splits]
    data, num_samples = load_lmdbs(lmdb_files, target_fields=target_fields, **kwargs)

    # split
    offset = 0
    data_splits = []
    for num_sample in num_samples:
        data_splits.append(data[offset: offset + num_sample])
        offset += num_sample

    # save processed
    for split, data_split in zip(splits, data_splits):
        save_path = processed_path / dataset_name
        if not save_path.exists():
            save_path.mkdir(parents=True)
        add_comment = f'\n{dataset_name} dataset from torchdrug. \nThis is {split} set. \n Labels are {target_fields} stored in key = "target".\n Only "aa" and "target" are used in this dataset, no structure features.'
        metadata = init_metadata(processed_path, comment=add_comment)
        env = lmdb.open(str(save_path / f'{dataset_name}_{split}.lmdb'), map_size=1024**4)
        if isinstance(data_split[0], list):
            data_split = [[d.to_dict() for d in data_lst] for data_lst in data_split]
        else:
            data_split = [d.to_dict() for d in data_split]
        sizes, keys = write_lmdb_datachunk(data_split, env)
        metadata["sizes"].extend(sizes)
        metadata["keys"].extend(keys)
        write_lmdb_metadata(metadata, env)




def process_enzyme_commission(ec_dir, splits, processed_path):
    dataset_name = 'EnzymeCommission'
    for split in splits:
        # load data
        # every item has 'prot_id', 'protein_length', 'sequence', 'ec_labels'
        data = np.load(ec_dir / f'{split}_seq_label.npz', allow_pickle=True)['data']
        # sizes = np.asarray([int(item['protein_length']) for item in data]) + 2
        metadata = np.load(ec_dir / 'metadata.npz', allow_pickle=True)['data'].item()
        ec2label = metadata['ec_set']
        # pos_weights = metadata['ec_pos_weights']
        # process
        data_split = []
        for item in tqdm(data):
            seq = item['sequence']
            name = item['prot_id']
            targets = [ec2label[i] for i in item['ec_labels']]
            data_split.append(commons.Protein.from_sequence(seq, name, targets))

        # save
        # for split, data_split in zip(splits, data_splits):
        save_path = processed_path / dataset_name
        if not save_path.exists():
            save_path.mkdir(parents=True)
        add_comment = f'\n{dataset_name} dataset from He Zhang. \nThis is {split} set. \n Labels are Enzyme commission number stored in key = "target", see below for conversion dict.\nOnly "aa" and "target" are used in this dataset, no structure features.'
        add_comment += '\n\nConversion dict:\n' + f"{repr(ec2label)}\n"
        metadata = init_metadata(processed_path, comment=add_comment)
        env = lmdb.open(str(save_path / f'{dataset_name}_{split}.lmdb'), map_size=1024**4)
        if isinstance(data_split[0], list):
            data_split = [[d.to_dict() for d in data_lst] for data_lst in data_split]
        else:
            data_split = [d.to_dict() for d in data_split]
        sizes, keys = write_lmdb_datachunk(data_split, env)
        metadata["sizes"].extend(sizes)
        metadata["keys"].extend(keys)
        write_lmdb_metadata(metadata, env)


def process_gene_ontology(go_dir, splits, processed_path, category):
    assert category in ['mf', 'bp', 'cc']
    dataset_name = f'GeneOntology_{category}'
    for split in splits:
        # load data
        # every item has 'prot_id', 'protein_length', 'sequence', '{category}_labels'
        data = np.load(go_dir / f'{split}_seq_label.npz', allow_pickle=True)['data']
        # sizes = np.asarray([int(item['protein_length']) for item in data]) + 2
        metadata = np.load(go_dir / 'metadata.npz', allow_pickle=True)['data'].item()
        go2label = metadata[f'{category}_set']
        # pos_weights = metadata[f'{category}_pos_weights']
        # process
        data_split = []
        for item in tqdm(data):
            seq = item['sequence']
            # patch for some sequences
            if isinstance(seq, list):
                seq = ''.join(seq)
            name = item['prot_id']
            targets = [go2label[i] for i in item[f'{category}_labels']]
            data_split.append(commons.Protein.from_sequence(seq, name, targets))
        # save
        # for split, data_split in zip(splits, data_splits):
        save_path = processed_path / dataset_name
        if not save_path.exists():
            save_path.mkdir(parents=True)
        add_comment = f'\n{dataset_name} dataset from He Zhang. \nThis is {split} set. \n Labels are Enzyme commission number stored in key = "target", see below for conversion dict.\nOnly "aa" and "target" are used in this dataset, no structure features.'
        add_comment += '\n\nConversion dict:\n' + f"{repr(go2label)}\n"
        metadata = init_metadata(processed_path, comment=add_comment)
        env = lmdb.open(str(save_path / f'{dataset_name}_{split}.lmdb'), map_size=1024**4)
        if isinstance(data_split[0], list):
            data_split = [[d.to_dict() for d in data_lst] for data_lst in data_split]
        else:
            data_split = [d.to_dict() for d in data_split]
        sizes, keys = write_lmdb_datachunk(data_split, env)
        metadata["sizes"].extend(sizes)
        metadata["keys"].extend(keys)
        write_lmdb_metadata(metadata, env)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("download_path", type=str)
    parser.add_argument("processed_path", type=str)
    args = parser.parse_args()

    # sequence only datasets
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/beta_lactamase.tar.gz"
    md5 = "65766a3969cc0e94b101d4063d204ba4"
    splits = ["train", "valid", "test"]
    target_fields = ["scaled_effect1"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'beta_lactamase')

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz"
    md5 = "d63d1d51ec8c20ff0d981e4cbd67457a"
    splits = ["train", "valid", "test"]
    target_fields = ["log_fluorescence"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'fluorescence')

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz"
    md5 = "aa1e06eb5a59e0ecdae581e9ea029675"
    splits = ["train", "valid", "test"]
    target_fields = ["stability_score"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'stability')

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/solubility.tar.gz"
    md5 = "8a8612b7bfa2ed80375db6e465ccf77e"
    splits = ["train", "valid", "test"]
    target_fields = ["solubility"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'solubility')

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization_2.tar.gz"
    md5 = "5d2309bf1c0c2aed450102578e434f4e"
    splits = ["train", "valid", "test"]
    target_fields = ["localization"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'subcellular_localization_2')

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz"
    md5 = "37cb6138b8d4603512530458b7c8a77d"
    splits = ["train", "valid", "test"]
    target_fields = ["localization"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'subcellular_localization')

    # structure
    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz"
    md5 = "2f61e8e09c215c032ef5bc8b910c8e97"
    splits = ["train", "valid", "casp12", "ts115", "cb513"]
    target_fields = ["ss3", "valid_mask"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'secondary_structure')

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz"
    md5 = "1d687bdeb9e3866f77504d6079eed00a"
    splits = ["train", "valid", "test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"]
    target_fields = ["class_label", "fold_label", "superfamily_label", "family_label"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'remote_homology')

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/proteinnet.tar.gz"
    md5 = "ab44ab201b1570c0171a2bba9eb4d389"
    splits = ["train", "valid", "test"]
    target_fields = ["tertiary", "valid_mask"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'proteinnet')

    ec_dir = Path('/mnta/yaosen/data/benchmarks/EC')
    splits = ['train', 'valid', 'test']
    dataset_name = 'EnzymeCommission'
    processed_path = Path('/mnta/yaosen/data/processed_benchmarks')
    process_enzyme_commission(ec_dir, splits, processed_path)
    # labels = list(range(538))

    go_dir = Path('/mnta/yaosen/data/benchmarks/GO')
    splits = ['train', 'valid', 'test']
    processed_path = Path('/mnta/yaosen/data/processed_benchmarks')
    categories = ['mf', 'bp', 'cc']
    for category in categories:
        process_gene_ontology(go_dir, splits, processed_path, category)
    # labels: mf: list(range(489)), bp: list(range(1943)), cc: list(range(320))

    # PPI
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/ppidata/human_ppi.zip"
    md5 = "89885545ebc2c11d774c342910230e20"
    splits = ["train", "valid", "test", "cross_species_test"]
    target_fields = ["interaction"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'human_ppi', sequence_field=["primary_1", "primary_2"])

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/ppidata/yeast_ppi.zip"
    md5 = "3993b02c3080d74996cddf6fe798b1e8"
    splits = ["train", "valid", "test", "cross_species_test"]
    target_fields = ["interaction"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'yeast_ppi', sequence_field=["primary_1", "primary_2"])

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/ppidata/ppi_affinity.zip"
    md5 = "d114907fd20c75820e41881f8901e9e4"
    splits = ["train", "valid", "test"]
    target_fields = ["interaction"]
    process_sequence_only_dataset(url, md5, splits, target_fields, args.download_path, args.processed_path, 'ppi_affinity', sequence_field=["primary_1", "primary_2"])
