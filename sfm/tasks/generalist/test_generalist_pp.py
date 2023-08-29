# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import dataclass
from itertools import islice
from typing import Dict

import torch
from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC
from tqdm import tqdm
from transformers import AutoTokenizer

from sfm.criterions.copilotloss import CopilotCriterionsPP
from sfm.data.mol_data.moltext_dataset import SupervisedProcessedDataWithSmiles
from sfm.logging import logger
from sfm.models.generalist.generalist_config import GeneralistConfig
from sfm.models.generalist.graphormer_llama import GraphormerLlamaModel
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.utils.chemical_tokens import CHEMICAL_TOKENS
from sfm.utils.cli_utils import cli
from sfm.utils.get_paranum import count_paranum
from sfm.utils.move_to_device import move_to_device
from sfm.utils.mypp_module import PipelineModule

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class TestGeneralistConfig:
    test_checkpoint_path: str
    batch_size: int
    output_fname: str
    test_task: str


class Evaler:
    def __init__(self, args, valid_data, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer)
        net = GraphormerLlamaModel(args, vocab_size)

        dist.init_distributed(
            dist_backend=args.dist_backend,
        )
        net = PipelineModule(
            layers=net.to_layers(),
            num_stages=args.pipeline_model_parallel_size,
            loss_fn=CopilotCriterionsPP(args, vocab_size),
            partition_method="parameters",
            device=args.local_rank,
        )

        see_memory_usage("Model built", force=True)

        self.model = net

        self.load_ckpt()

        logger.info(f"number of parameters: {count_paranum(self.model)}")

        self.valid_data = valid_data
        self.valid_dataloader = DataLoader(
            self.valid_data,
            batch_size=args.batch_size,
            collate_fn=self.valid_data.collate,
            drop_last=False,
            shuffle=False,
        )

    def load_ckpt(self):
        if os.path.isdir(self.args.test_checkpoint_path) and os.path.exists(
            os.path.join(self.args.test_checkpoint_path)
        ):
            logger.info(
                "Load pipeline parallel checkpoint from %s"
                % self.args.test_checkpoint_path
            )
            for i in range(self.model._num_layers):
                logger.info(f"load layer {i}")
                self.model._modules[str(i)].load_state_dict(
                    torch.load(
                        f"{self.args.test_checkpoint_path}/layer_{i:02}-model_states.pt"
                    ),
                    strict=True,
                )
            logger.info("Load pipeline parallel checkpoint done")

    @torch.no_grad()
    def validate_reg(self):
        logger.info("Start...")
        self.model.eval()
        num_preds_res = []
        num_labels_res = []
        for i, batch_data in tqdm(enumerate(self.valid_dataloader)):
            batch_data = move_to_device(
                batch_data, device=self.args.local_rank, non_blocking=True
            )
            model_input, labels = batch_data

            logits = self.model(model_input)
            num_logits = logits[1].squeeze(-1)
            # lm_labels = labels[0][..., 0].to(torch.int64)
            num_labels = labels[2]
            # import ipdb; ipdb.set_trace()
            num_idx = num_labels != -100
            num_labels = num_labels[num_idx]
            num_logits = num_logits[num_idx].view(-1)

            num_preds_res.append(num_logits.detach())
            num_labels_res.append(num_labels)

        # calculate RMSE for num_preds_res and num_labels_res
        num_preds_res = torch.cat(num_preds_res, dim=0)
        num_labels_res = torch.cat(num_labels_res, dim=0)
        rmse = torch.sqrt(torch.mean((num_preds_res - num_labels_res) ** 2))
        logger.info(f"Data size: {len(num_preds_res)}")
        logger.info(f"RMSE: {rmse}")

    @torch.no_grad()
    def validate_reg_multitask(self):
        logger.info("Start...")
        self.model.eval()
        num_preds_res = []
        num_labels_res = []
        for i, batch_data in tqdm(enumerate(self.valid_dataloader)):
            batch_data = move_to_device(
                batch_data, device=self.args.local_rank, non_blocking=True
            )
            model_input, labels = batch_data

            logits = self.model(model_input)
            num_logits = logits[1].squeeze(-1)
            # lm_labels = labels[0][..., 0].to(torch.int64)
            num_labels = labels[2]
            # import ipdb; ipdb.set_trace()
            num_idx = num_labels != -100
            num_labels = num_labels[num_idx]
            num_logits = num_logits[num_idx].view(-1)

            num_preds_res.append(num_logits.detach())
            num_labels_res.append(num_labels)

        # calculate RMSE for num_preds_res and num_labels_res
        num_preds_res = torch.cat(num_preds_res, dim=0)
        num_labels_res = torch.cat(num_labels_res, dim=0)
        rmse = torch.sqrt(torch.mean((num_preds_res - num_labels_res) ** 2))
        logger.info(f"Data size: {len(num_preds_res)}")
        logger.info(f"RMSE: {rmse}")

    @torch.no_grad()
    def validate_cls(self):
        logger.info("Start...")
        self.model.eval()
        metric_auroc = BinaryAUROC()
        test_size = 0
        for i, batch_data in tqdm(enumerate(self.valid_dataloader)):
            batch_data = move_to_device(
                batch_data, device=self.args.local_rank, non_blocking=True
            )
            model_input, labels = batch_data

            logits = self.model(model_input)
            lm_logits = logits[0]
            # Yes: 8241, No: 3782
            lm_binary_label_mask = (model_input[0] == 8241) | (model_input[0] == 3782)
            assert lm_binary_label_mask.sum() == len(labels[0])
            lm_binary_label_mask_shift = torch.cat(
                [
                    lm_binary_label_mask[:, 1:],
                    torch.zeros_like(lm_binary_label_mask[:, 0]).unsqueeze(-1),
                ],
                dim=-1,
            )
            assert lm_binary_label_mask_shift.sum() == len(labels[0])
            lm_binary_logits = lm_logits[lm_binary_label_mask_shift]
            lm_binary_true = lm_binary_logits[:, 8241]
            lm_binary_false = lm_binary_logits[:, 3782]
            lm_binary_pred = torch.softmax(
                torch.stack([lm_binary_true, lm_binary_false], dim=-1), dim=-1
            )[:, 0]
            lm_binary_label_dict_idx = labels[0][lm_binary_label_mask]
            lm_binary_label = torch.zeros_like(lm_binary_pred)
            lm_binary_label[lm_binary_label_dict_idx == 8241] = 1
            lm_binary_label[lm_binary_label_dict_idx == 3782] = 0

            metric_auroc.update(lm_binary_pred, lm_binary_label)
            test_size += len(labels[0])

        logger.info(f"Data size: {test_size}")
        logger.info(f"AUROC {metric_auroc.compute()}")

    @torch.no_grad()
    def validate_cls_mlp(self):
        logger.info("Start...")
        self.model.eval()
        metric_auroc = BinaryAUROC()
        test_size = 0
        for i, batch_data in tqdm(enumerate(self.valid_dataloader)):
            batch_data = move_to_device(
                batch_data, device=self.args.local_rank, non_blocking=True
            )
            model_input, labels = batch_data

            logits = self.model(model_input)
            mlp_logits = logits[1]
            # lm_logits = logits[0]
            # Yes: 8241, No: 3782
            lm_binary_label_mask = (model_input[0] == 8241) | (model_input[0] == 3782)
            assert lm_binary_label_mask.sum() == len(labels[0])
            lm_binary_label_mask_shift = torch.cat(
                [
                    lm_binary_label_mask[:, 1:],
                    torch.zeros_like(lm_binary_label_mask[:, 0]).unsqueeze(-1),
                ],
                dim=-1,
            )
            assert lm_binary_label_mask_shift.sum() == len(labels[0])
            lm_binary_logits = mlp_logits[lm_binary_label_mask_shift].squeeze(-1)
            lm_binary_logits = torch.sigmoid(lm_binary_logits)
            # lm_binary_true = lm_binary_logits[:, 8241]
            # lm_binary_false = lm_binary_logits[:, 3782]
            # lm_binary_pred = torch.softmax(torch.stack([lm_binary_true, lm_binary_false], dim=-1), dim=-1)[:, 0]
            lm_binary_label_dict_idx = labels[0][lm_binary_label_mask]
            lm_binary_label = torch.zeros_like(lm_binary_logits)
            lm_binary_label[lm_binary_label_dict_idx == 8241] = 1
            lm_binary_label[lm_binary_label_dict_idx == 3782] = 0

            metric_auroc.update(lm_binary_logits, lm_binary_label)
            test_size += len(labels[0])

        logger.info(f"Data size: {test_size}")
        logger.info(f"AUROC {metric_auroc.compute()}")


def make_supervised_data_module(args) -> Dict:
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

    special_tokens_dict["additional_special_tokens"] = CHEMICAL_TOKENS
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
        num_token_id=tokenizer.encode("<num>", add_special_tokens=False)[0],
    )

    return dict(valid_dataset=dataset, tokenizer=tokenizer)


@cli(DistributedTrainConfig, GraphormerConfig, GeneralistConfig, TestGeneralistConfig)
def main(args) -> None:
    data_module = make_supervised_data_module(args)
    logger.info(f"length of dataset: {len(data_module['valid_dataset'])}")
    if args.tensor_model_parallel_size == 1:
        evaler = Evaler(
            args,
            valid_data=data_module["valid_dataset"],
            # vocab_size=data_module["vocab_size"],
            tokenizer=data_module["tokenizer"],
        )
    else:
        raise Exception("Not implemented yet")
    if args.test_task == "reg":
        evaler.validate_reg()
    elif args.test_task == "reg_multitask":
        evaler.validate_reg_multitask()
    elif args.test_task == "cls":
        evaler.validate_cls()
    elif args.test_task == "cls_mlp":
        evaler.validate_cls_mlp()
    else:
        raise Exception("Unknown test task")


if __name__ == "__main__":
    main()
