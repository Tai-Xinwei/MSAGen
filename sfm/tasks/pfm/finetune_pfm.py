# -*- coding: utf-8 -*-
import os
import sys

import torch

from sfm.logging import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import dataclass

import sentencepiece as spm

from sfm.criterions.mae3d import ProteinPMLM
from sfm.data.prot_data.dataset import BatchedDataDataset, DownstreamLMDBDataset
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_mlm_config import PfmMlmConfig
from sfm.models.pfm.pfm_mlm_model import PfmMlmBpeModel
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.pfm.pfmmodel import PFMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer
from sfm.utils.cli_utils import cli

# for custimize training steps, cosine lr decay
TRAINLENTH = 0


@dataclass
class DownstreamConfig:
    task_name: str
    data_basepath: str
    head_dropout: float = 0.1
    base_model: str = "pfm"
    spm_model_path: str = ""


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
                batch_data["pad_mask"] = batch_data[f"pad_mask_{i}"]
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

        self.spm = spm.SentencePieceProcessor(model_file=args.spm_model_path)
        self.multi_seq_tasks = ["yeast_ppi", "human_ppi", "ppi_affinity"]

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
        tokens = self.spm.encode(aastr, out_type=int)
        return [self.args.bos_token_id] + tokens + [self.args.eos_token_id]

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        target = item["target"]

        if self.args.task_name in self.multi_seq_tasks:
            aaseq = [self.reverse2str(item[f"aa_{i}"]) for i in range(2)]
            item0, item1 = self.patch(aaseq[0]), self.patch(aaseq[1])
            return {
                "x_0": item0,
                "x_1": item1,
                "target": target,
            }
        else:
            aaseq = self.reverse2str(item["aa"])
            item = self.patch(aaseq)
            return {"x": item, "target": target}

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        def make_x_batch(x_list):
            max_len = max([len(x) for x in x_list])
            for i in range(len(x_list)):
                x_list[i] = x_list[i] + [self.args.pad_token_id] * (
                    max_len - len(x_list[i])
                )
            return torch.tensor(x_list, dtype=torch.long)

        batch = {}
        if self.args.task_name in self.multi_seq_tasks:
            batch["x_0"] = make_x_batch([s["x_0"] for s in samples])
            batch["x_1"] = make_x_batch([s["x_1"] for s in samples])
        else:
            batch["x"] = make_x_batch([s["x"] for s in samples])

        batch["target"] = torch.cat([torch.from_numpy(s["target"]) for s in samples])
        batch["target_offset"] = torch.tensor(
            [len(s["target"]) for s in samples], dtype=torch.long
        )

        if "x" in batch:
            batch["pad_mask"] = batch["x"] != self.args.pad_token_id
        else:
            batch["pad_mask_0"] = batch["x_0"] != self.args.pad_token_id
            batch["pad_mask_1"] = batch["x_1"] != self.args.pad_token_id
        return batch

    def num_tokens(self, index: int) -> int:
        if self.args.task_name in self.multi_seq_tasks:
            item = self.dataset[index]
            return len(item["x_0"]) + len(item["x_1"]) + 4
        return len(self.spm.encode(self.dataset[index]["x"], out_type=int)) + 2


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


def build_base_model(args, load_ckpt=True):
    if args.base_model == "pfm":
        return PFMModel(args, loss_fn=ProteinPMLM, load_ckpt=load_ckpt)
    elif args.base_model == "pfm_bpe":
        return PfmMlmBpeModel(args, load_ckpt=load_ckpt)


def init_model(args, load_ckpt=True):
    # seems model loading require this parameter
    args.ft = True
    basemodel = build_base_model(args, load_ckpt=load_ckpt)

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
