# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import asdict, dataclass

import numpy as np
import sentencepiece as spm
from scipy import stats

from sfm.criterions.mae3d import ProteinPMLM
from sfm.data.prot_data.dataset import BatchedDataDataset, DownstreamLMDBDataset
from sfm.logging import logger, metric_logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_mlm_config import (
    PfmMlmConfig,
    pfm_mlm_tiny_config,
    pfm_mlm_tiny_h24_config,
)
from sfm.models.pfm.pfm_mlm_model import PfmMlmBpeModel
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.pfm.pfmmodel import PFMModel
from sfm.pipeline.accelerator.dataclasses import (
    DistributedTrainConfig,
    ModelOutput,
    ValidLogOutput,
)
from sfm.pipeline.accelerator.trainer import Model, Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device

# for custimize training steps, cosine lr decay
TRAINLENTH = 0


@dataclass
class DownstreamConfig:
    task_name: str
    data_basepath: str
    head_dropout: float = 0.1
    base_model: str = "pfm"


class SingleSequenceModel(Model):
    def __init__(self, args, model, n_classes):
        super().__init__()
        self.args = args
        self.model = model
        self.n_sequence = (
            2 if args.task_name in ["yeast_ppi", "human_ppi", "ppi_affinity"] else 1
        )
        self.n_classes = n_classes
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(args.head_dropout),
            torch.nn.Linear(
                args.encoder_embed_dim * self.n_sequence, args.encoder_embed_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.head_dropout),
            torch.nn.Linear(args.encoder_embed_dim, n_classes),
        )

    def forward(self, batch_data):
        if self.n_sequence == 1:
            # x: (B, L, C)
            x = self.model.ft_forward(batch_data)
            x = x[:, 0, :].squeeze(1)
            # logger.debug(f"batch_data keys: {batch_data.keys()}")
        else:
            xs = []
            for i in range(self.n_sequence):
                batch_data["x"] = batch_data[f"x_{i}"]
                x = self.model.ft_forward(batch_data)
                x = x[:, 0, :].squeeze(1)
                xs.append(x)
            x = torch.cat(xs, dim=1)
        logits = self.head(x)
        return logits

    def load_pretrained_weights(self, args, pretrained_model_path):
        self.model.load_pretrained_weights(args, pretrained_model_path)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        if self.n_sequence == 1:
            bs = batch_data["x"].shape[0]
        else:
            bs = batch_data["x_0"].shape[0]

        if DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"] == "regression":
            mean, std = DownstreamLMDBDataset.TASKINFO[self.args.task_name]["mean_std"]
            target = (batch_data["target"].unsqueeze(1) - mean) / std
            # (B x n_classes)
            lossfn = torch.nn.MSELoss()
        elif DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"] == "binary":
            target = batch_data["target"].unsqueeze(1).to(torch.float32)
            # (B x n_classes)
            lossfn = torch.nn.BCEWithLogitsLoss()
        elif (
            DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"]
            == "classification"
        ):
            target = batch_data["target"]
            # (B, )
            lossfn = torch.nn.CrossEntropyLoss()
        elif (
            DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"]
            == "multi_classification"
        ):
            target = batch_data["target"]
            target_offset = batch_data["target_offset"]
            # multi_hot_target: (B, n_classes)
            multi_hot_target = torch.zeros(
                (bs, self.n_classes), dtype=torch.float32, device=target.device
            )
            curr = 0
            for idx, n in enumerate(target_offset):
                label_idx = target[curr : curr + n]  # (n, 1)
                multi_hot_target[idx, label_idx] = 1
                curr += n
            target = multi_hot_target
            lossfn = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # (B x n_classes)
        logits = model_output
        loss = lossfn(logits.to(torch.float32), target)
        return ModelOutput(loss=loss, num_examples=bs)

    def config_optimizer(self):
        optimizer, _ = myAdam(
            self,
            lr=self.args.max_lr,
            betas=[0.9, 0.999],
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )

        # may be here we should use a grouped lr scheduler
        # eg: set the lr of the head to be 10x of the pretrained model
        total_num_steps = (
            self.args.total_num_epochs * TRAINLENTH / self.args.train_batch_size + 1
        )
        logger.info(f"Manually set total num steps: {total_num_steps}")
        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=total_num_steps,  # self.args.total_num_steps,
            warmup_max_lr=self.args.max_lr,
            warmup_num_steps=int(0.1 * total_num_steps),  # self.args.warmup_num_steps,
            d_tilde=0.1,  # this is the ratio of the lr of the encoder to the head
            decay_type=DECAY_COSINE_RATE,
        )
        return optimizer, lr_scheduler


class BPEPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        vocab,
        args=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.args = args
        self.vocab = vocab

        self.idx_to_tok = {v: k for k, v in self.vocab.tok_to_idx.items()}
        # self.collate_fn = dataset.collate
        self.sequence_length = args.max_length

    def reverse2str(self, tokens):
        vocab = self.vocab
        aaseq = []
        for i in tokens:
            if i in [
                vocab.unk_idx,
                vocab.padding_idx,
                vocab.cls_idx,
                vocab.mask_idx,
                vocab.eos_idx,
            ]:
                continue
            aaseq.append(self.idx_to_tok[i])
        return "".join(aaseq)

    def patch(self, aastr):
        # TODO: should return an object that PfmMlmBpeModel accepts
        pass

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        target, target_offset = item["target"], item["target_offset"]

        if self.args.task_name in ["yeast_ppi", "human_ppi", "ppi_affinity"]:
            aaseq = [self.reverse2str(item[f"x_{i}"]) for i in range(2)]
            item0, item1 = self.patch(aaseq[0]), self.patch(aaseq[1])
            return {
                "x_0": item0,
                "x_1": item1,
                "target": target,
                "target_offset": target_offset,
            }
        else:
            aaseq = self.reverse2str(item["x"])
            item = self.patch(aaseq)
            return {"x": item, "target": target, "target_offset": target_offset}

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        # TODO: Data + target + target_offset
        pass

    def num_tokens(self, index: int) -> int:
        # TODO: return the length (?) of the sequence
        pass


def load_batched_dataset(args):
    global TRAINLENTH
    # required args:
    # args.task_name, args.data_basepath
    dataset_dict = DownstreamLMDBDataset.load_dataset(args)
    trainset = dataset_dict["train"]
    TRAINLENTH = len(trainset)
    valset = dataset_dict["valid"]
    # others are test sets
    testset_dict = {
        k: v for k, v in dataset_dict.items() if k not in ["train", "valid"]
    }
    if args.base_model == "pfm_bpe":
        # make the conversion
        logger.info("Loading BPE dataset")
        train_data = BPEPatchDataset(
            trainset,
            vocab=trainset.vocab,
            args=args,
        )
        val_data = BPEPatchDataset(
            valset,
            vocab=trainset.vocab,
            args=args,
        )
        testset_dict = {
            k: BPEPatchDataset(
                v,
                vocab=trainset.vocab,
                args=args,
            )
            for k, v in testset_dict.items()
        }
    else:
        logger.info("Loading sequence dataset")
        train_data = BatchedDataDataset(
            trainset,
            args=args,
            vocab=trainset.vocab,
        )
        val_data = BatchedDataDataset(
            valset,
            args=args,
            vocab=trainset.vocab,
        )
        testset_dict = {
            k: BatchedDataDataset(
                v,
                args=args,
                vocab=trainset.vocab,
            )
            for k, v in testset_dict.items()
        }
    logger.info(f"Got test dataset: {testset_dict.keys()}")
    return train_data, val_data, testset_dict


def build_base_model(args):
    if args.base_model == "pfm":
        return PFMModel(args, loss_fn=ProteinPMLM)
    elif args.base_model == "pfm_bpe":
        return PfmMlmBpeModel(args)


def init_model(args):
    # seems model loading require this parameter
    args.ft = True
    basemodel = build_base_model(args)

    if DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "regression":
        model = SingleSequenceModel(args, basemodel, n_classes=1)
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "binary":
        model = SingleSequenceModel(args, basemodel, n_classes=1)
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "classification":
        n_classes = len(DownstreamLMDBDataset.TASKINFO[args.task_name]["classes"])
        model = SingleSequenceModel(args, basemodel, n_classes=n_classes)
    elif (
        DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "multi_classification"
    ):
        n_classes = len(DownstreamLMDBDataset.TASKINFO[args.task_name]["classes"])
        model = SingleSequenceModel(args, basemodel, n_classes=n_classes)
    else:
        raise NotImplementedError()
    return model


@cli(DistributedTrainConfig, PFMConfig, PfmMlmConfig, DownstreamConfig)
def finetune(args) -> None:
    train_data, val_data, testset_dict = load_batched_dataset(args)

    model = init_model(args)
    logger.info(f"Finetuning on task {args.task_name}")

    # any important settings to keep in mind?
    trainer = Trainer(
        args,
        model,
        train_data=train_data,
        valid_data=val_data,
    )
    trainer.train()


if __name__ == "__main__":
    finetune()
