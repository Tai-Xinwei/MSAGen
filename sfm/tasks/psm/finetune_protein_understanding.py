# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from sfm.data.prot_data.dataset import BatchedDataDataset
from sfm.data.psm_data.ft_prot_dataset import (
    ProteinDownstreamDataset,
    accuracy,
    area_under_prc,
    binary_accuracy,
    f1_max,
    mae,
    mse,
    rmse,
    spearmanr,
)
from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import (
    DECAY_COSINE_RATE,
    AdamFP16,
    groupWarmupDecayLR,
    myAdam,
)
from sfm.models.psm.psmmodel import PSMModel
from sfm.modules.layer_norm import LayerNorm
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer, seed_everything
from sfm.tasks.psm.ft_modules import PSM_FT_REGISTER
from sfm.utils import env_init
from sfm.utils.cli_utils import cli, wandb_init

import wandb  # isort:skip


# for custimize training steps, cosine lr decay
TRAINLENTH = 0


@dataclass
class ProteinUnderstandingConfig:
    task_name: str
    data_basepath: str
    head_dropout: float = 0.1
    # base_model: str = "pfm"
    label_normalize: bool = False
    checkpoint_dir: str = ""
    which_set: str = "valid"


@dataclass
class Config(DistributedTrainConfig, PSMConfig, ProteinUnderstandingConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    ode_mode: bool = False
    finetune_module: Optional[str] = None


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


def load_batched_dataset(args):
    global TRAINLENTH
    dataset_dict = ProteinDownstreamDataset.load_dataset(args)
    trainset = dataset_dict["train"]
    TRAINLENTH = len(trainset)
    valset = dataset_dict["valid"]
    # others are test sets
    testset_dict = {
        k: v for k, v in dataset_dict.items() if k not in ["train", "valid"]
    }

    logger.info("Loading sequence dataset")
    train_data = BatchedDataDataset(
        trainset,
        args=args,
        # vocab=trainset.vocab,
    )
    val_data = BatchedDataDataset(
        valset,
        args=args,
        # vocab=trainset.vocab,
    )
    testset_dict = {
        k: BatchedDataDataset(
            v,
            args=args,
            # vocab=trainset.vocab,
        )
        for k, v in testset_dict.items()
    }
    logger.info(f"Got test dataset: {testset_dict.keys()}")
    return train_data, val_data, testset_dict


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
            torch.nn.GELU(),
            LayerNorm(args.encoder_embed_dim),
            torch.nn.Linear(args.encoder_embed_dim, n_classes),
        )
        self.return_residue_emb = (
            True if args.task_name == "secondary_structure" else False
        )

    def forward(self, batch_data):
        if self.n_sequence == 1:
            # x: (B, L, C)
            result_dict = self.model(batch_data)
            x = result_dict["encoder_output"]

            if not self.return_residue_emb:
                x = x[:, 0, :].squeeze(1)
            else:
                # (B, L, C)
                x = x
        else:
            raise NotImplementedError()
            # xs = []
            # for i in range(self.n_sequence):
            #     batch_data["x"] = batch_data[f"x_{i}"]
            #     x = self.model.ft_forward(batch_data)
            #     x = x[:, 0, :].squeeze(1)
            #     xs.append(x)
            # x = torch.cat(xs, dim=1)
        logits = self.head(x)
        return logits

    def load_pretrained_weights(self, args, pretrained_model_path):
        self.model.load_pretrained_weights(args, pretrained_model_path)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        # token_id = batched_data["token_id"]
        # padding_mask = token_id.eq(0)  # B x T x 1
        if self.n_sequence == 1:
            bs = batch_data["token_id"].shape[0]
        else:
            raise NotImplementedError()
            # bs = batch_data["x_0"].shape[0]

        if (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "regression"
        ):
            mean, std = ProteinDownstreamDataset.TASKINFO[self.args.task_name][
                "mean_std"
            ]
            if self.args.label_normalize:
                target = (
                    batch_data["target"].unsqueeze(1).to(torch.float32) - mean
                ) / std
            else:
                target = batch_data["target"].unsqueeze(1).to(torch.float32)
            # (B x n_classes)
            lossfn = torch.nn.MSELoss()
        elif ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"] == "binary":
            target = batch_data["target"].unsqueeze(1).to(torch.float32)
            # (B x n_classes)
            lossfn = torch.nn.BCEWithLogitsLoss()
        elif (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "classification"
        ):
            target = batch_data["target"]
            # (B, )
            lossfn = torch.nn.CrossEntropyLoss()
        elif (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "multi_classification"
        ):
            target = batch_data["target"]
            target_offset = batch_data["target_offset"]
            target = multi_label_transform(target, target_offset, bs, self.n_classes)
            lossfn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        elif (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "residue_classification"
        ):
            raise NotImplementedError()
            # (B, L, C)
            model_output = model_output[:, 1:-1, :]
            # they are padded to the same length with cls and bos
            # (B, L)
            target = batch_data["target"][:, :-2].to(torch.long)
            # (B, L)
            target_mask = batch_data["target_mask"][:, :-2].to(torch.bool)
            lossfn = torch.nn.CrossEntropyLoss()
            loss = lossfn(
                model_output[target_mask],  # (Nvalid, C)
                target[target_mask],  # (Nvalid, )
            )
            logits = model_output
            return ModelOutput(loss=loss, num_examples=bs, logits=logits, label=target)
        else:
            raise NotImplementedError()

        # (B x n_classes)
        logits = model_output
        loss = lossfn(logits.to(torch.float32), target)
        return ModelOutput(
            loss=loss,
            num_examples=bs,
            logits=logits,
            label=target,
            log_output={"loss": loss.item()},
        )

    def config_optimizer(self, model=None):
        optimizer = myAdam(
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
            d_tilde=0.5,  # this is the ratio of the lr of the encoder to the head
            decay_type=DECAY_COSINE_RATE,
        )
        return optimizer, lr_scheduler

    def calculate_metric(self, label, logits) -> dict:
        pred = logits.cpu().squeeze().to(torch.float32)
        true = label.cpu().squeeze().to(torch.float32)
        if (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "regression"
        ):
            mean, std = ProteinDownstreamDataset.TASKINFO[self.args.task_name][
                "mean_std"
            ]
            if self.args.label_normalize:
                pred = pred * std + mean
            test_fn = [spearmanr, mae, mse, rmse]
        elif ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"] == "binary":
            test_fn = [binary_accuracy]
        elif (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "classification"
        ):
            test_fn = [accuracy]
        elif (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "multi_classification"
        ):
            test_fn = [f1_max, area_under_prc]
        elif (
            ProteinDownstreamDataset.TASKINFO[self.args.task_name]["type"]
            == "residue_classification"
        ):

            def ssp_accuracy(pred, true):
                mask = true < 0
                pred = pred[mask]
                true = true[mask]
                return accuracy(pred, true)

            test_fn = [ssp_accuracy]
        else:
            raise NotImplementedError()
        metric_result = {fn.__name__: fn(pred, true).item() for fn in test_fn}
        logger.info(f"Metric result on valid set: {metric_result}")
        return metric_result


# NOTE: set self.psm_config.psm_finetune_mode to load ckpt from args.loadcheck_path
def init_model(args):
    # seems model loading require this parameter
    args.ft = True
    basemodel = PSMModel(args, loss_fn=DiffMAE3dCriterions)

    if ProteinDownstreamDataset.TASKINFO[args.task_name]["type"] == "regression":
        n_classes = 1
    elif ProteinDownstreamDataset.TASKINFO[args.task_name]["type"] == "binary":
        n_classes = 1
    elif ProteinDownstreamDataset.TASKINFO[args.task_name]["type"] == "classification":
        n_classes = len(ProteinDownstreamDataset.TASKINFO[args.task_name]["classes"])
    elif (
        ProteinDownstreamDataset.TASKINFO[args.task_name]["type"]
        == "multi_classification"
    ):
        n_classes = len(ProteinDownstreamDataset.TASKINFO[args.task_name]["classes"])
    elif (
        ProteinDownstreamDataset.TASKINFO[args.task_name]["type"]
        == "residue_classification"
    ):
        n_classes = len(ProteinDownstreamDataset.TASKINFO[args.task_name]["classes"])
    else:
        raise NotImplementedError()
    model = SingleSequenceModel(args, basemodel, n_classes=n_classes)
    return model


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


@hydra.main(
    version_base=None, config_path="../../../config_file", config_name="config_psm"
)
def main(args: DictConfig) -> None:
    args = OmegaConf.to_object(args)
    assert isinstance(
        args, Config
    ), f"args must be an instance of Config! But it is {type(args)}"

    wandb_init(args)
    seed_everything(args.seed)
    env_init.set_env(args)

    ### define psm dataset here
    train_data, valid_data, testset_dict = load_batched_dataset(args)

    # define psm models here, define the diff loss in DiffMAE3dCriterions
    model = init_model(args)

    logger.info(f"Finetuning on task {args.task_name}")

    trainer = Trainer(
        args,
        model,
        train_data=train_data,
        valid_data=valid_data,
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


# TODO:
# 1. Done 参考Unified Dataset中的AFDBLMDBDataset，对一下数据字段
# 2. Done 搬运几个collater fn并根据这些字段修改一下，例如collate_downstream_fn
# 3. Done 对齐模型的接口，用哪个embedding, decoder_x_output & self.psm_config.psm_finetune_mode
# 5. Done validation时为什么collator会报错？item不是dict? 没有blocking，报错位置不对， 解决
# 6. Done Validation的时候计算metrics，为什么不算了
# 7. Done 根据BFM的finetune代码，修改一下训练参数
# 8. 任务怎么提交不会error
