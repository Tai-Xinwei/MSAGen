.. A4SFramework documentation master file, created by
   sphinx-quickstart on Mon Sep 25 05:01:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _distributed:

Large Distributed Training
==========================

Zero Optimization
~~~~~~~~~~~~~~~~~

`Zero Optimization/FSDP <https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/>`__ is a novel optimization technique that allows us to train models with more than 1 billion parameters on a single GPU.

To use zero optimization in A4SFramework, just need to add strategy flag in the training script. For example, to use Zero1 strategy, just add ``--strategy Zero1`` in the training script.

.. code::

    torchrun --use_env main.py --strategy Zero1


Model parallelism
~~~~~~~~~~~~~~~~~
A4SFramework leverage `Megatron <https://github.com/NVIDIA/Megatron-LM>`__ to support model parallelism. To use model parallelism in A4SFramework, firstly strategy flag is needed in the training script. For example, to use model parallelism with 2 pipeline parallelism and 2 tensor parallelism, just add ``--strategy ThreeD --pipeline_model_parallel_size 2 --tensor_model_parallel_size 2`` in the training script:

.. code::

    torchrun --nproc_per_node=8 --use_env main.py --strategy ThreeD --pipeline_model_parallel_size 2 --tensor_model_parallel_size 2

Secondly, model needs to be written in a way that is compatible with model parallelism. For example, to use model parallelism with Chemical Generalist (MFM + Llama2), the model is written in :mod:`models.generalist.graphormer_llama`.


Pretrained Layer Spec
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfm.utils.pretrained_layer_spec.PretrainedLayerSpec
    :members:
    :undoc-members:
    :show-inheritance:


The :mod:`sfm.utils.pretrained_layer_spec`.PretrainedLayerSpec class is a useful tool for loading pretrained checkpoints and initializing the parameters of large models to prevent out-of-memory (OOM) errors on the CPU.

For example, to load a pretrained checkpoint for the Llama2 decoder layer, we can use the following code:

.. code::

    from sfm.utils.pretrained_layer_spec import PretrainedLayerSpec
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    pipe_layers.append(
        PretrainedLayerSpec(
            LlamaDecoderLayerPP,
            config,
            load_ckpt=load_ckpt,
            pretrained_ckpt_path=os.path.join(
                args.llm_model_name_or_path, "model.layers.{}.pt".format(i)
            ),
        )
    )

The detailed usage example of :mod:sfm.utils.pretrained_layer_spec.PretrainedLayerSpec can be find in :mod:`models.generalist.graphormer_llama` as well.


Examples
~~~~~~~~
Here are some examples for DDP, Zero, and model parallelism.

Pretraining Graphormer with DDP or Zero:

.. code::

    bash scripts/graphormer/pretrain_graphormer.sh

Finetuning Graphormer with DDP or Zero:

.. code::

    bash scripts/graphormer/ft_graphormer.sh

Finetuning Llama2 + Graphormer with model parallelism:

.. code::

    bash scripts/generalist/ft_graphormer_llama_smiles.sh

Finetuning Llama2 + Graphormer with PP + TP + Zero1:

.. code::

    bash scripts/generalist/ftmp_graphormer_llama_smiles.sh
