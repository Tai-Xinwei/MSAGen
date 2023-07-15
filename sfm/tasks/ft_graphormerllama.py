# -*- coding: utf-8 -*-
import logging
import os
import sys
from typing import Dict

import deepspeed
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".")

from sfm.data.mol_data.moltext_dataset import SupervisedProcessedData
from sfm.models.transformers import AutoTokenizer, PreTrainedTokenizer
from sfm.models.transformers.models.llama.configuration_llama import LlamaConfig
from sfm.pipeline.graphormerllama_trainer import Trainer
from sfm.utils.add_argument import add_argument
from sfm.utils.chemical_tokens import CHEMICAL_TOKENS

logging.getLogger().setLevel(logging.ERROR)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


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

    LlamaConfig.from_pretrained(args.llm_model_name_or_path)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    special_tokens_dict["additional_special_tokens"] = CHEMICAL_TOKENS
    tokenizer.add_special_tokens(special_tokens_dict)

    """Make dataset and collator for supervised fine-tuning."""
    dataset = SupervisedProcessedData(
        data_path=args.data_path,
        dataset_names=args.dataset_names,
        dataset_splits=args.dataset_splits,
        mol2idx_dict_path=args.smiles_dict_path,
        embedding_length=args.embedding_length,
        in_memory=False,
        mol_size_path=args.mol_size_path,
        pad_token_id=tokenizer.pad_token_id,
        pool_mode=args.pool_mode,
    )
    return dict(train_dataset=dataset, eval_dataset=None, vocab_size=len(tokenizer))


def main() -> None:
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    args = add_argument()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    # os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']
    os.environ["NCCL_BLOCKING_WAIT"] = "0"

    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()

    print(
        "Print os.environ:--- RANK: {}, WORLD_SIZE: {}, LOCAL_RANK: {}".format(
            os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]
        )
    )

    data_module = make_supervised_data_module(args, mode="train")
    print("length of dataset", len(data_module["train_dataset"]))

    args.add_3d = False
    args.infer = True
    if args.rank == 0:
        print(
            {
                "add-3d": args.add_3d,
                "no-2d": args.no_2d,
                "mfm_lora": args.mfm_lora,
                "pool_mode": args.pool_mode,
                "embedding_length": args.embedding_length,
            }
        )

    trainer = Trainer(
        args, data_module["train_dataset"], vocab_size=data_module["vocab_size"]
    )
    trainer()


if __name__ == "__main__":
    main()
