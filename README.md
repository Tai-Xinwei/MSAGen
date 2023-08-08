[![Build Status](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_apis/build/status%2FPython%20Unit%20Tests?branchName=main)](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_build/latest?definitionId=1&branchName=main)

# SFM Repository


## Pre-commit Hooks

To install pre-commit, run the following command:
```
pip install pre-commit
```

Run the following command from the root of this repository:
```
pre-commit install
```

To run pre-commit manually on all files in your repository, use the following command:
```
pre-commit run --all-files
```


When submit multi-node job in Hai1 Cluster, avoid using following nodes due to IB port error:
```
Severe: GCRHYP3C103, GCRHYP3C112, GCRHYP3C142, GCRHYP3C149, GCRHYP3C336, GCRHYP3C324, GCRHYP3C224 (Not use),
Medium: GCRHYP3C314, gcrhyp3c342, GCRHYP3C346, gcrhyp3c257 (OK to use with short job like 1-3 days, do not use it with pretraining),
Mild: gcrhyp3c108, GCRHYP3C218, gcrhyp3c225, gcrhyp3c253 (Good to use, port error happens rarely)
```

## Example Scripts
Pretraining Graphormer on 4 GPUs:
```
bash scripts/pretrain_graphormer.sh
```

Finetuning Graphormer on 4 GPUs:
```
bash scripts/ft_graphormer.sh
```

Finetuning Llama2 + Graphormer on 4 GPUs:
```
bash scripts/ft_graphormer_llama_smiles.sh
```
