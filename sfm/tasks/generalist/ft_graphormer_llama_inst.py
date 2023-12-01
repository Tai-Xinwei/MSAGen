# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from transformers import AutoTokenizer

from megatron.arguments import parse_megatron_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.initialize import initialize_megatron
from sfm.data.mol_data.moltext_dataset import SupervisedProcessedDataWithSmiles
from sfm.logging import logger
from sfm.models.generalist import GraphormerLlamaModel
from sfm.models.generalist.generalist_config import GeneralistConfig
from sfm.models.graphormer.graphormer_config import GraphormerConfig
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
        dataset_ratios=args.dataset_ratios,
        pool_mode=args.pool_mode,
        embedding_length=args.embedding_length,
        num_token_id=tokenizer.encode("<num>", add_special_tokens=False)[0],
        use_pp=(
            args.strategy == TrainStrategy.Pipeline
            or args.strategy == TrainStrategy.ThreeD
        ),
        use_pbc=args.use_pbc,
        local_rank=args.local_rank,
        num_data_loading_workers=args.num_data_loading_workers,
        skip_num_datasets=args.skip_num_datasets,
    )

    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        vocab_size=len(tokenizer),
        tokenizer=tokenizer,
    )


@cli(
    DistributedTrainConfig,
    GraphormerConfig,
    GeneralistConfig,
    TransformerConfig,
    parse_megatron_args,
)
def main(args) -> None:
    data_module = make_supervised_data_module(args, mode="train")

    if args.strategy == TrainStrategy.ThreeD:
        initialize_megatron(args, tokenizer=data_module["tokenizer"])

    logger.info(f"length of dataset: {len(data_module['train_dataset'])}")

    model = GraphormerLlamaModel(args, data_module["vocab_size"])

    trainer = Trainer(
        args,
        train_data=data_module["train_dataset"],
        valid_data=data_module["eval_dataset"],
        model=model,
        loss_log_dict={"lm_loss": 0.0, "num_loss": 0.0, "bce_loss": 0.0},
    )
    trainer.train()


if __name__ == "__main__":
    main()
