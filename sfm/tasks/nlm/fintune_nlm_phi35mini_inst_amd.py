# -*- coding: utf-8 -*-
import os
import sys

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from transformers import AutoTokenizer

from sfm.data.sci_data.dataset import LMDBInstDataset, LMDBInstDFDataset
from sfm.logging import logger
from sfm.models.nlm.moe_config import MoeModelConfig, sfm_nlm_phi35_mini_config
from sfm.models.nlm.nlm_mi300 import NLMBaseAMDModel
from sfm.pipeline.accelerator.dataclasses import TrainStrategy
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli


@cli(MoeModelConfig)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    assert (
        args.valid_data_path is not None and len(args.valid_data_path) > 0
    ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    # tokenizer = NlmLlama3Tokenizer.from_pretrained(args.dict_path)

    # llama2 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.dict_path)
    # args.vocab_size = len(tokenizer)  # use original tokenizer from llama2
    # tokenizer.pad_token = tokenizer.eos_token  # pad_token is eos_token
    # args.pad_token_id = tokenizer.pad_token_id

    # phi35 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.dict_path)
    print("len(tokenizer)", len(tokenizer))
    print("tokenizer.pad_token_id", tokenizer.pad_token_id)
    print("tokenizer.pad_token", tokenizer.pad_token)
    assert args.pad_token_id == tokenizer.pad_token_id

    args.use_dali_pipeline = False
    # args = sfm_nlm_1b_base_config(args)
    args = sfm_nlm_phi35_mini_config(args)
    print("args.vocab_size", args.vocab_size)

    if args.strategy == TrainStrategy.ThreeD:
        raise NotImplementedError("3D training is not supported for this task on AMD.")
    # model = NLMBaseAMDModel(args, len(tokenizer))
    model = NLMBaseAMDModel(args, args.vocab_size)

    if args.train_hf_data_path != "":
        train_dataset = LMDBInstDFDataset(
            args.train_data_path,
            args.train_hf_data_path,
            args.pad_token_id,
            args.max_position_embeddings,
            args.hf_sample_count,
        )
    else:
        train_dataset = LMDBInstDataset(
            args.train_data_path, args.pad_token_id, args.max_position_embeddings
        )

    valid_dataset = LMDBInstDataset(
        args.valid_data_path, args.pad_token_id, args.max_position_embeddings
    )
    logger.info("datasets loaded")

    trainer = Trainer(
        args,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
        loss_log_dict={"lm_loss": 0.0},
    )
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
