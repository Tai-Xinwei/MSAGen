# -*- coding: utf-8 -*-
import math
import os
from argparse import ArgumentParser
from typing import Dict

import deepspeed
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.runtime.utils import see_memory_usage
from transformers import AutoTokenizer

# from apex.optimizers import FusedAdam as Adam
from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.initialize import initialize_megatron
from megatron.model.utils import init_method_normal
from megatron.tokenizer.tokenizer import _vocab_size_with_padding
from sfm.criterions.copilotloss import CopilotCriterionsPP
from sfm.criterions.copilotloss3d import CopilotCriterionsMP, CopilotCriterionsNumMP
from sfm.data.mol_data.moltext_dataset import SupervisedProcessedDataWithSmiles
from sfm.logging import logger
from sfm.models.generalist.generalist_config import GeneralistConfig
from sfm.models.generalist.graphormer_llama import GraphormerLlamaModel
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.pipeline.accelerator.dataclasses import TrainerConfig
from sfm.utils import PPEngine
from sfm.utils.arg_utils import ExtraArgsProvider
from sfm.utils.chemical_tokens import CHEMICAL_TOKENS
from sfm.utils.get_paranum import count_paranum
from sfm.utils.mypp_module import PipelineModule
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import groupWarmupDecayLR


class Trainer3D:
    def __init__(self, freeze_list=[], unfreeze_list=[]):
        super().__init__()
        see_memory_usage("Before Building Model", force=True)

        # Initialization.
        extra_args_provider = ExtraArgsProvider(
            [TrainerConfig, GraphormerConfig, GeneralistConfig]
        )
        initialize_megatron(extra_args_provider)
        args = get_args()
        self.args = args

        logger.info(
            {
                "add-3d": args.add_3d,
                "no-2d": args.no_2d,
                "mfm_lora": args.mfm_lora,
                "pool_mode": args.pool_mode,
                "embedding_length": args.embedding_length,
                "btn_adaptor": args.btn_adaptor,
            }
        )

        # Data stuff
        data_module = make_supervised_data_module(args, mode="train")
        train_data = data_module["train_dataset"]
        args.padded_vocab_size = max(args.padded_vocab_size, data_module["vocab_size"])
        vocab_size = args.padded_vocab_size
        logger.info(
            "length of dataset {}, vocab_size {}".format(
                len(data_module["train_dataset"]), vocab_size
            )
        )
        ckp_list = [
            "layer_{}-model_00-model_states.pt".format(str(i).zfill(2))
            for i in range(61)
        ]

        # Model, optimizer, and learning rate.
        net = GraphormerLlamaModel(args, vocab_size, ckp_list=ckp_list)
        topo = PipeModelDataParallelTopology(
            num_pp=mpu.get_pipeline_model_parallel_world_size(),
            num_mp=mpu.get_tensor_model_parallel_world_size(),
            num_dp=mpu.get_data_parallel_world_size(),
        )

        criterion = (
            CopilotCriterionsPP(args, vocab_size)
            if args.tensor_model_parallel_size == 1
            else CopilotCriterionsNumMP(args, vocab_size)
        )

        net = PipelineModule(
            layers=net.to_layers(),
            topology=topo,
            loss_fn=criterion,
            device=args.local_rank,
            partition_method=args.pp_partition_layer_name,
            part_list=args.pp_part_list,
            loss_log_dict={"lm_loss": 0.0, "num_loss": 0.0, "bce_loss": 0.0},
        )

        optimizer, param_groups = myAdam(
            net,
            # impl=Adam,
            freeze_list=freeze_list,
            unfreeze_list=unfreeze_list,
            lr=args.max_lr,
            betas=[0.9, 0.999],
            weight_decay=0.0,
            eps=1e-8,
        )
        scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=args.total_num_steps,
            warmup_max_lr=args.max_lr,
            warmup_num_steps=args.warmup_num_steps,
        )
        see_memory_usage("Model built", force=True)

        self.model_engine, _, _, _ = PPEngine.initialize(
            args=args,
            model=net,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            model_parameters=param_groups,  # [p for p in net.parameters() if p.requires_grad],
            training_data=train_data,
            collate_fn=train_data.collater,
            repeat_dataloader=True,
        )

    def resume(self, resume_path, ckpt_id=None):
        if os.path.isdir(resume_path) and os.paht.exists(
            os.path.join(resume_path, "latest")
        ):
            logger.info("resume from %s" % resume_path)
            if ckpt_id is None:
                self.model_engine.load_checkpoint(resume_path)
            else:
                self.model_engine.load_checkpoint(resume_path, tag=ckpt_id)

    def train_tensor_pipeline(self):
        logger.info("start 3D parallelism training")

        for global_step in range(1, self.args.total_num_steps + 1):
            self.model_engine.train_batch()
            if global_step % 500 == 0:
                self.save_ckp(global_step)

            torch.cuda.empty_cache()

    def save_ckp(self, global_step):
        self.model_engine.save_checkpoint(
            save_dir=self.args.save_dir,
            client_state={"checkpoint_step": global_step},
        )


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def make_supervised_data_module(args, mode="train") -> Dict:
    assert mode in [
        "train",
        "eval",
        "test",
    ], f"Invalid mode: {mode}, must be train, eval, or test."

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # special_tokens_dict["additional_special_tokens"] = CHEMICAL_TOKENS
    tokenizer.add_special_tokens(special_tokens_dict)

    """ Make dataset and collator for supervised fine-tuning. """
    dataset = SupervisedProcessedDataWithSmiles(
        data_path=args.data_path,
        dataset_names=args.dataset_names,
        dataset_splits=args.dataset_splits,
        in_memory=False,
        model_max_length=args.model_max_length,
        mol_embed_type="atoms",
        molecule_max_size=512,
        pad_token_id=tokenizer.pad_token_id,
        # dataset_ratios=args.dataset_ratios,
        pool_mode=args.pool_mode,
        embedding_length=args.embedding_length,
    )
    vocab_size = _vocab_size_with_padding(len(tokenizer), args)

    return dict(train_dataset=dataset, eval_dataset=None, vocab_size=vocab_size)
