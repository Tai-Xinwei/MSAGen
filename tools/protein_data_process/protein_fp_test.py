# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent))
from sfm.data.prot_data.dataset import DownstreamLMDBDataset
from sfm.data.prot_data.vocalubary import Alphabet
from sfm.logging import logger
from sfm.tasks.pfm.commons import mae, mse, rmse, f1_max, accuracy, binary_accuracy, pearsonr, spearmanr, area_under_prc, variadic_mean
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import lmdb
import scipy
from tqdm import tqdm
from joblib import Parallel, delayed
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, metrics
import torch
from itertools import product


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Namespace()
args.max_length = 2048
args.data_basepath = "/blob/data/bfm_benchmark"
args.task_name = "solubility"
args.n_jobs = 12



class ProteinSequenceFingerprint:
    vocab = Alphabet()
    idx_to_tok = {v: k for k, v in vocab.tok_to_idx.items()}

    @staticmethod
    def reverse2str(tokens):
        aaseq = []
        for i in tokens:
            if i in [ProteinSequenceFingerprint.vocab.unk_idx, ProteinSequenceFingerprint.vocab.padding_idx,
                     ProteinSequenceFingerprint.vocab.cls_idx, ProteinSequenceFingerprint.vocab.mask_idx, ProteinSequenceFingerprint.vocab.eos_idx,]:
                continue
            aaseq.append(ProteinSequenceFingerprint.idx_to_tok[i])
        return "".join(aaseq)

    def batch_convert(self, seqs, n_jobs=-1):
        ret = np.array(Parallel(n_jobs=n_jobs)(delayed(self)(i) for i in tqdm(seqs, ncols=80, desc="Fingerprinting")))
        return ret

    def seq_stats(self, seqs):
        lens = [len(s) for s in seqs]
        print(f"Sequence stats: # {len(seqs)} sequences, mean #aa {np.mean(lens):.2f}, max #aa {np.max(lens)}, min #aa {np.min(lens)}")


class ProteinkmerHistogram(ProteinSequenceFingerprint):
    def __init__(self, k):
        self.k = k
        # only upper case letters, 25 tokens in total
        self.standard_toks = [i for i in self.vocab.standard_toks if i.isupper()]
        print(f"Building kmer2idx for k={k}")
        self.kmer2idx = {"".join(i): idx for idx, i in enumerate(product(self.standard_toks, repeat=k))}

    def __call__(self, tokens):
        seq = self.reverse2str(tokens)
        kmer_count = np.zeros(len(self.kmer2idx))
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if kmer in self.kmer2idx:
                kmer_count[self.kmer2idx[kmer]] += 1
        return kmer_count



class ProteinPairwiseResidueFingerprint(ProteinSequenceFingerprint):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, tokens):
        seq = self.reverse2str(tokens)
        # iteration 1
        for r in range(self.radius+1):
            for idx in range(len(seq)):
                seq[idx-r:idx+r+1]



dataset_dict = DownstreamLMDBDataset.load_dataset(args)
trainset = dataset_dict["train"]
valset = dataset_dict["valid"]
# others are test sets
testset_dict = {
    k: v for k, v in dataset_dict.items() if k not in ["train", "valid"]
}

def load_seq_target(dset):
    seq, Y = [], []
    if DownstreamLMDBDataset.TASKINFO[dset.task_name]["type"] == "multi_classification":
        n_class = len(DownstreamLMDBDataset.TASKINFO[dset.task_name]["classes"])
        for item in tqdm(dset, ncols=80, desc='Read data'):
            seq.append(item["aa"])
            y = np.zeros(n_class)
            y[item["target"].squeeze()] = 1
            Y.append(y)
    else:
        for item in tqdm(dset, ncols=80, desc='Read data'):
            seq.append(item["aa"])
            Y.append(item["target"])
    seq, Y = seq, np.array(Y).squeeze()
    return seq, Y

seq_train, Y_train = load_seq_target(trainset)
seq_val, Y_val = load_seq_target(valset)
seq_test, Y_test = load_seq_target(testset_dict["test"])


fp = ProteinkmerHistogram(4)

fp.seq_stats(seq_train)
fp.seq_stats(seq_val)
fp.seq_stats(seq_test)

X_train = fp.batch_convert(seq_train, args.n_jobs)
X_val = fp.batch_convert(seq_val, args.n_jobs)
X_test = fp.batch_convert(seq_test, args.n_jobs)

train_pool = Pool(data=X_train, label=Y_train)
val_pool = Pool(data=X_val, label=Y_val)
test_pool = Pool(data=X_test, label=Y_test)

class F1Max:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # the larger metric value the better
        return True

    def evaluate(self, approxes, target, weight):
        print(f"From {self.__class__.__name__}: {len(approxes)}, {len(target)}")
        print(f"From {self.__class__.__name__}: {approxes[0]}, {target[0]}")

        approxes, target = torch.tensor(approxes), torch.tensor(target)
        return f1_max(approxes, target), 0


params = {'loss_function': '', 'iterations': 5000, 'random_seed': 42, 'learning_rate': 0.01, 'early_stopping_rounds': 100, 'verbose': 10, 'task_type': 'GPU',
          'use_best_model': True, 'metric_period': 50, }

if DownstreamLMDBDataset.TASKINFO[args.task_name]['type'] == 'binary':
    print("binary")
    params['loss_function'] = 'Logloss'
    params['eval_metric'] = 'Accuracy'
    model = CatBoostClassifier(**params)
elif DownstreamLMDBDataset.TASKINFO[args.task_name]['type'] == 'classification':
    print("classification")
    params['loss_function'] = 'CrossEntropy'
    params['eval_metric'] = 'Accuracy'
    model = CatBoostClassifier(**params)
elif DownstreamLMDBDataset.TASKINFO[args.task_name]['type'] == 'regression':
    print("regression")
    params['loss_function'] = 'MSE'
    model = CatBoostRegressor(**params)
elif DownstreamLMDBDataset.TASKINFO[args.task_name]['type'] == 'multi_classification':
    print("multi_classification")
    params['loss_function'] = 'MultiLogloss'
    params['eval_metric'] = 'Accuracy'
    model = CatBoostClassifier(**params)


model.fit(train_pool, eval_set=val_pool, ) # plot=True,)

pred_test = model.predict(X_test)
print("Prediction shape", pred_test.shape)
print("Groundtruth shape", Y_test.shape)

if DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "regression":
    mean, std = DownstreamLMDBDataset.TASKINFO[args.task_name]["mean_std"]
    pred = torch.tensor(pred_test)
    true = torch.tensor(Y_test)
    test_fns = [pearsonr, spearmanr, mae, mse, rmse]
elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "binary":
    pred = torch.tensor(pred_test)
    true = torch.tensor(Y_test)
    test_fns = [binary_accuracy]
elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "classification":
    pred = torch.tensor(pred_test)
    true = torch.tensor(Y_test)
    test_fns = [accuracy]
elif (
    DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "multi_classification"
):
    pred = torch.tensor(pred_test)
    true = torch.tensor(Y_test)
    test_fns = [f1_max, area_under_prc]

results = dict()
for fn in test_fns:
    results[fn.__name__] = fn(pred, true).item()

print("Predicted 1s:", model.predict(X_test).sum())
print("Scores:", results)
