# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import asdict, dataclass

import sentencepiece as spm
from commons import (
    accuracy,
    area_under_prc,
    binary_accuracy,
    f1_max,
    mae,
    mse,
    rmse,
    spearmanr,
)

from sfm.data.prot_data.dataset import DownstreamLMDBDataset
from sfm.logging import logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_mlm_config import PfmMlmConfig
from sfm.models.pfm.pfm_mlm_model import PfmMlmBpeModel
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer
from sfm.utils.cli_utils import cli

# for custimize training steps, cosine lr decay
TRAINLENTH = 0


def multi_label_transform(target, target_offset, bs, n_classes):
    # multi_hot_target: (B, n_classes)
    multi_hot_target = torch.zeros(
        (bs, n_classes), dtype=torch.float32, device=target.device
    )
    curr = 0
    for idx, n in enumerate(target_offset):
        label_idx = target[curr : curr + n]  # (n, 1)
        multi_hot_target[idx, label_idx] = 1
        curr += n
    return multi_hot_target


@dataclass
class DownstreamConfig:
    task_name: str
    data_basepath: str
    head_dropout: float = 0.1
    base_model: str = "pfm_bpe"
    label_normalize: bool = False
    checkpoint_dir: str = ""
    spm_model_path: str = ""
    which_set: str = "valid"


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

    def forward(self, batch_data, residue_emb=False):
        if self.n_sequence == 1:
            # x: (B, L, C)
            x = self.model.ft_forward(batch_data)
            if not residue_emb:
                x = x[:, 0, :].squeeze(1)
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
            if self.args.label_normalize:
                target = (
                    batch_data["target"].unsqueeze(1).to(torch.float32) - mean
                ) / std
            else:
                target = batch_data["target"].unsqueeze(1).to(torch.float32)
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
            target = multi_label_transform(target, target_offset, bs, self.n_classes)
            lossfn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        elif (
            DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"]
            == "residue_classification"
        ):
            target = batch_data["target"]
            batch_data["target_mask"]
            lossfn = torch.nn.CrossEntropyLoss()

        # (B x n_classes)
        logits = model_output
        loss = lossfn(logits.to(torch.float32), target)
        return ModelOutput(loss=loss, num_examples=bs, logits=logits, label=target)

    def config_optimizer(self):
        optimizer, _ = myAdam(
            self,
            lr=self.args.max_lr,
            betas=[0.9, 0.999],
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )

        total_num_steps = (
            self.args.total_num_epochs * TRAINLENTH / self.args.train_batch_size + 1
        )
        logger.info(f"Manually set total num steps: {total_num_steps}")
        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=total_num_steps,
            warmup_max_lr=self.args.max_lr,
            warmup_num_steps=int(0.1 * total_num_steps),
            d_tilde=0.1,  # this is the ratio of the lr of the encoder to the head
            decay_type=DECAY_COSINE_RATE,
        )
        return optimizer, lr_scheduler

    def calculate_metric(self, label, logits) -> dict:
        pred, true = logits.cpu().to(torch.float32), label.cpu().to(torch.float32)
        if DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"] == "regression":
            mean, std = DownstreamLMDBDataset.TASKINFO[self.args.task_name]["mean_std"]
            if self.args.label_normalize:
                pred = pred * std + mean
            test_fn = [spearmanr, mae, mse, rmse]
        elif DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"] == "binary":
            test_fn = [binary_accuracy]
        elif (
            DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"]
            == "classification"
        ):
            test_fn = [accuracy]
        elif (
            DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"]
            == "multi_classification"
        ):
            test_fn = [f1_max, area_under_prc]
        else:
            raise NotImplementedError()
        metric_result = {fn.__name__: fn(pred, true).item() for fn in test_fn}
        logger.info(f"Metric result on valid set: {metric_result}")
        return metric_result


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
    # args.task_name, args.data_basepath
    dataset_dict = DownstreamLMDBDataset.load_dataset(args)
    trainset = dataset_dict["train"]
    TRAINLENTH = len(trainset)
    valset = dataset_dict["valid"]
    # others are test sets
    testset_dict = {
        k: v for k, v in dataset_dict.items() if k not in ["train", "valid"]
    }

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
    logger.info(f"Got test dataset: {testset_dict.keys()}")
    return train_data, val_data, testset_dict


def build_base_model(args, load_ckpt=True):
    model = PfmMlmBpeModel(args, load_ckpt=load_ckpt)
    args.encoder_embed_dim = model.config.hidden_size
    return model


def init_model(args, load_ckpt: bool):
    # seems model loading require this parameter
    args.ft = True
    basemodel = build_base_model(args, load_ckpt=load_ckpt)

    if DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "regression":
        n_classes = 1
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "binary":
        n_classes = 1
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "classification":
        n_classes = len(DownstreamLMDBDataset.TASKINFO[args.task_name]["classes"])
    elif (
        DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "multi_classification"
    ):
        n_classes = len(DownstreamLMDBDataset.TASKINFO[args.task_name]["classes"])
    elif (
        DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "secondary_structure"
    ):
        n_classes = len(DownstreamLMDBDataset.TASKINFO[args.task_name]["classes"])
    else:
        raise NotImplementedError()
    model = SingleSequenceModel(args, basemodel, n_classes=n_classes)
    return model


@cli(DistributedTrainConfig, PFMConfig, PfmMlmConfig, DownstreamConfig)
def finetune(args) -> None:
    train_data, val_data, testset_dict = load_batched_dataset(args)
    model = init_model(args, load_ckpt=True)
    logger.info(f"Finetuning on task {args.task_name}")

    trainer = Trainer(
        args,
        model,
        train_data=train_data,
        valid_data=val_data,
    )
    trainer.train()


if __name__ == "__main__":
    finetune()
