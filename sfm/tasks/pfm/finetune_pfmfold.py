# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import dataclass

from sfm.criterions.mae3d import ProteinPMLM
from sfm.data.prot_data.dataset import BatchedDataDataset, ProteinLMDBDataset
from sfm.logging import logger
from sfm.models.pfm.openfold.loss import compute_predicted_aligned_error, compute_tm
from sfm.models.pfm.openfold.structure_module import StructureModule
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.pfm.pfmfold_config import StructureModuleConfig
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


class StructureModel(Model):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

        self.structure_module = StructureModule(
            c_s=args.encoder_embed_dim,
            c_z=args.encoder_embed_dim,
            c_ipa=args.c_ipa,
            c_resnet=args.c_resnet,
            no_heads_ipa=args.no_heads_ipa,
            no_qk_points=args.no_qk_points,
            no_v_points=args.no_v_points,
            dropout_rate=args.dropout,
            no_blocks=args.no_blocks,
            no_transition_layers=args.no_transition_layers,
            no_resnet_blocks=args.no_resnet_blocks,
            no_angles=args.no_angles,
            trans_scale_factor=args.trans_scale_factor,
            epsilon=args.epsilon,
            inf=args.inf,
        )

        self.loss_fn = torch.nn.L1Loss(reduction="mean")

    def forward(self, batch_data):
        with torch.no_grad():
            x_0 = batch_data["x"].clone()

        x = self.model.ft_forward(batch_data)
        x = self.model.net.layer_norm(x)

        single_rep = x
        q = self.model.net.fc_pmlm_q(x)
        k = self.model.net.fc_pmlm_k(x)
        pair_rep = torch.einsum("bih,bjh->bijh", q, k)

        evoformer_output_dict = {
            "single": single_rep,
            "pair": pair_rep,
        }

        padding_mask = (x_0[:, :]).eq(1)  # B x T x 1

        mask = (~padding_mask).long()
        structure: dict = self.structure_module(evoformer_output_dict, x_0, mask=mask)

        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }

        return structure

    def load_pretrained_weights(self, args, pretrained_model_path):
        self.model.load_pretrained_weights(args, pretrained_model_path)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        with torch.no_grad():
            pos = batch_data["pos"]
            bs = pos.shape[0]

            aa_seq = batch_data["x"]
            padding_mask = (aa_seq).eq(1)  # B x T x 1

        model_output = model_output["structure"][padding_mask]
        pos = pos[padding_mask]

        # compute loss
        loss = self.loss_fn(model_output["positions"], pos)

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


def load_batched_dataset(args):
    dataset = ProteinLMDBDataset(args)
    trainset, valset = dataset.split_dataset(sort=False)

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

    return train_data, val_data


@cli(DistributedTrainConfig, PFMConfig, StructureModuleConfig, DownstreamConfig)
def finetune(args) -> None:
    train_data, val_data = load_batched_dataset(args)

    basemodel = PFMModel(args, loss_fn=ProteinPMLM, load_ckpt=True)
    model = StructureModel(args, basemodel)

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
