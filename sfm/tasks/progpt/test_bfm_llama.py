# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import torch
from transformers import AutoTokenizer

# from megatron.arguments import parse_megatron_args
# from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.initialize import initialize_megatron
from sfm.data.prot_data.prottext_dataset import ProteinTextDataset
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.progpt.progpt import ProGPTModel
from sfm.models.progpt.progpt_config import ProGPTConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, TrainStrategy
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS

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

    if args.use_llama_tokenizer:
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

        special_tokens_dict["additional_special_tokens"] = SCIENCE_TAG_TOKENS
        tokenizer.add_special_tokens(special_tokens_dict)

    else:
        tokenizer = SFMDecTokenizer.from_pretrained(
            args.llm_model_name_or_path,
            prot_spm_path=os.path.join(args.tokenizer_path, "ur50bpe/bpe"),
            dna_spm_path=os.path.join(args.tokenizer_path, "dnabpe/bpe"),
            rna_spm_path=os.path.join(args.tokenizer_path, "rnabpe/bpe"),
        )

        args.vocab_size = len(tokenizer)  # now we have new tokens
        args.pad_token_id = tokenizer.pad_token_id

    """ Make dataset and collator for supervised fine-tuning. """
    dataset = ProteinTextDataset(
        data_path=args.valid_data_path,
        model_max_length=args.model_max_length,
        protein_max_size=args.protein_max_size,
        pad_token_id=tokenizer.pad_token_id,
        pool_mode=args.pool_mode,
        embedding_length=args.embedding_length,
        num_token_id=tokenizer.encode("<num>", add_special_tokens=False)[0],
        protein_pad_id=1,
        pp_mode=(
            args.strategy == TrainStrategy.Pipeline
            or args.strategy == TrainStrategy.ThreeD
        ),
        local_rank=args.local_rank,
        use_llama_tokenizer=args.use_llama_tokenizer,
    )

    return dict(
        train_dataset=None,
        eval_dataset=dataset,
        vocab_size=len(tokenizer),
        tokenizer=tokenizer,
    )


@cli(
    DistributedTrainConfig,
    PFMConfig,
    ProGPTConfig,
    # TransformerConfig,
    # parse_megatron_args,
)
def main(args) -> None:
    data_module = make_supervised_data_module(args, mode="train")

    # if args.strategy == TrainStrategy.ThreeD:
    # initialize_megatron(args, tokenizer=data_module["tokenizer"])

    logger.info(f"length of dataset: {len(data_module['eval_dataset'])}")
    logger.info(f"vocab size: {data_module['vocab_size']}")

    args.val_batch_log_interval = 100
    model = ProGPTModel(args, data_module["vocab_size"])

    trainer = Trainer(
        args,
        train_data=data_module["eval_dataset"],
        valid_data=data_module["eval_dataset"],
        model=model,
        loss_log_dict={"lm_loss": 0.0, "lm_loss_text": 0.0, "lm_loss_special": 0.0},
    )

    trainer.finetune_from_checkpoint()

    trainer.validate()


if __name__ == "__main__":
    main()
