# A4SFramework tutorials

Simple tutorials to demonstrate [A4SFramework package](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_artifacts/feed/sfm-framework) capabilities.

## Installation of the A4SFramework Package using Pip

Initial step is to generate a Personal Access Token using the user's credentials to be able to access the feed. To do that, get these packages (a virtual environment should be considered):
```bash
pip install keyring artifacts-keyring
```

A4SFramework package installation will ignore the dependencies completely, this way, installation is as compact as possible. Depending on the use cases, these dependencies should also be preinstalled:
```bash
pip install --no-cache-dir setuptools==69.5.1 cython deepspeed==0.14.0 peft wandb torch-geometric torch-tb-profiler
```

Then, installing A4SFramework Package can be done with:
```bash
pip install a4sframework -i https://pkgs.dev.azure.com/AI4ScienceSFM/SFM_framework/_packaging/sfm-framework/pypi/simple/ --no-deps
```

## Use-case
Currently, these tutorials are built on top of official GPT2 training on only few tokens for simplicity. More information on the actual training can be found [here](https://huggingface.co/docs/transformers/en/model_doc/gpt2).

1. In the folder [ZeRO](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?version=GBei/tut&_a=contents&path=/tutorials/ZeRO), A4SFramework package's [ZeRO Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/) capability is demonstrated.

1. In the folder [DynamicLoader](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?version=GBei/tut&_a=contents&path=/tutorials/DynamicLoader), A4SFramework package's [dynamic dataloader strategy](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?path=/sfm/data/dynamics_loader.py) is demonstrated.

## Run
Simply use the `start.sh` scripts on respective folder.

## Notes:
Note that NumPy recently released v.2, which is not backward compatible. Hence, reverting back to v.1 with `pip install numpy==1.26.4` should be considered.
