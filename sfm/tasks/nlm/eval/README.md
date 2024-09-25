# VLLM NLM Inference Engine

## Create an Isolated Env for VLLM

### Step 1: Create the Python 3.10 Environment
```bash
conda create -n vllm python=3.10
```
This command creates a new environment called `myenv` with Python 3.10.

### Step 2: Activate the Environment
```bash
conda activate vllm
```

### Step 3: Install `vllm`
You can install `vllm` using `pip` inside the Conda environment:

```bash
pip install vllm
```

### Step 4: Verify Installation
To ensure everything is set up correctly, run the following to check the Python version and confirm `vllm` is installed:

```bash
python --version
pip show vllm
```


## Prepare data
You need to put your data in this format:

nlm/
└── eval/
    └── inputs/
        └── nlm_data/
            ├── molecules_test/
            ├── molecules_test_mini/
            └── science_test/
                ├── test.bace.instruct.tsv
                ├── test.bbbp.instruct.tsv
                ├── test.desc2mol.tsv
                ├── test.hERG.tsv
                ├── test.mol2desc.tsv
                ├── test.molinstruct.reaction.tsv
                ├── test.raw.i2s_i.txt
                ├── test.raw.i2s_s.txt
                ├── test.raw.s2i_i.txt
                ├── test.raw.s2i_s.txt
                ├── test.uspto50k.reaction.osmi.tsv
                └── test.uspto50k.retro.osmi.tsv


## Prepare HuggingFace Format Checkpoint
You will need to generate HF-format checkpoint with `moe_inference_module.py`. Store the converted checkpoint in a local path `${NLM_LOCAL_PATH}`, do not utilize `blobfuse` location.

## Inference with VLLM-NLM Engine
Execute this in NLM home, do note that `${NLM_LOCAL_PATH}` is the path to your local HF-format NLM ckpt:
```
# In vllm env
bash ./scripts/nlm/eval/run_nlm_moe_vllm_inference.sh
```
