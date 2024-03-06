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
from sfm.models.pfm.openfold import data_transform as AFDT
from sfm.models.pfm.openfold.feats import atom14_to_atom37
from sfm.models.pfm.openfold.heads import AuxiliaryHeads
from sfm.models.pfm.openfold.loss import (
    AlphaFoldLoss,
    compute_predicted_aligned_error,
    compute_tm,
)
from sfm.models.pfm.openfold.openfold_config import (
    HeadsConfig,
    StructureModuleConfig,
    loss_config,
)
from sfm.models.pfm.openfold.structure_module import StructureModule
from sfm.models.tox.tox_config import TOXConfig
from sfm.models.tox.tox_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.tox.toxmodel import TOXModel
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
            c_s=args.c_s,
            c_z=args.c_z,
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

        heads_config = HeadsConfig(c_s=args.c_s, c_z=args.c_z)
        self.aux_heads = AuxiliaryHeads(
            heads_config,
        )

        self.loss_fn = AlphaFoldLoss(loss_config)
        self.loss_angle = torch.nn.MSELoss(reduction="mean")
        self.loss_dist = torch.nn.L1Loss(reduction="mean")

    def data_transform(self, batch_data):
        batch_data = AFDT.esm_to_alphafold_aatype(batch_data)
        # batch_data = AFDT.cast_to_64bit_ints(batch_data)
        # batch_data = AFDT.squeeze_features(batch_data)
        # batch_data = AFDT.make_seq_mask(batch_data)
        batch_data = AFDT.make_atom14_masks(batch_data)
        batch_data = AFDT.make_atom14_positions(batch_data)
        batch_data = AFDT.atom37_to_frames(batch_data)
        batch_data = AFDT.atom37_to_torsion_angles(batch_data)
        batch_data = AFDT.make_pseudo_beta(batch_data)
        batch_data = AFDT.get_backbone_frames(batch_data)
        batch_data = AFDT.get_chi_angles(batch_data)

        return batch_data

    def forward(self, batch_data):
        batch_data = self.data_transform(batch_data)

        with torch.no_grad():
            x_0 = batch_data["x"].clone()  # aa type from esm dict
            aatype = batch_data["aatype"].clone()  # aa type from alphafold dict
            padding_mask = (
                (x_0[:, :]).eq(1)
                | (x_0[:, :]).eq(0)
                | (x_0[:, :]).eq(2)
                | (x_0[:, :]).eq(3)
            )  # B x T x 1

        # choose mode from ["ori_angle", "T_noise", "Diff_noise"]
        x = self.model.ft_forward(batch_data, mode="T_noise")
        x = self.model.net.layer_norm(x)

        angle_output = self.model.net.angle_decoder(x)
        angle_output = angle_output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        single_rep = x
        q = self.model.net.fc_pmlm_q(x)
        k = self.model.net.fc_pmlm_k(x)
        pair_rep = torch.einsum("bih,bjh->bijh", q, k)
        # x_pair = self.model.net.pair_head(pair_rep)
        x_pair = None

        outputs = {
            "single": single_rep,
            "pair": pair_rep,
        }

        mask = (~padding_mask).long()
        structure: dict = self.structure_module(outputs, aatype, mask=mask)

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
                "single",
            ]
        }
        outputs["sm"] = structure
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], batch_data
        )
        outputs["final_atom_mask"] = batch_data["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        outputs.update(self.aux_heads(outputs))

        return outputs, angle_output, x_pair

    def load_pretrained_weights(self, args, pretrained_model_path):
        self.model.load_pretrained_weights(args, pretrained_model_path)

    # def compute_loss(self, model_output, batch_data) -> ModelOutput:
    #     structure = model_output[0]
    #     angle_output = model_output[1]
    #     x_pair = model_output[2]

    #     with torch.no_grad():
    #         ori_pos = batch_data["pos"]
    #         bs = ori_pos.shape[0]

    #         ori_angle = batch_data["ang"]
    #         angle_mask = batch_data["ang_mask"]

    #         pos_mask = batch_data["pos_mask"]

    #         aa_seq = batch_data["x"]
    #         padding_mask = (aa_seq).eq(1)  # B x T x 1

    #         # delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
    #         # ori_dist = delta_pos0.norm(dim=-1)

    #     pos_mask = pos_mask & (~padding_mask.unsqueeze(-1))
    #     Ca_pos_m1 = structure["positions"][-1, :, :, 1, :]
    #     Ca_pos_m2 = structure["positions"][-2, :, :, 1, :]
    #     Ca_pos_m3 = structure["positions"][-3, :, :, 1, :]
    #     Ca_pos_m4 = structure["positions"][-4, :, :, 1, :]

    #     # pair_pos = (Ca_pos_m1.unsqueeze(1) - Ca_pos_m1.unsqueeze(2)).norm(dim=-1)
    #     Ca_pos_m1 = Ca_pos_m1[pos_mask]
    #     Ca_pos_m2 = Ca_pos_m2[pos_mask]
    #     Ca_pos_m3 = Ca_pos_m3[pos_mask]
    #     Ca_pos_m4 = Ca_pos_m4[pos_mask]
    #     ori_pos = ori_pos[pos_mask]

    #     angle_mask = angle_mask & (~padding_mask.unsqueeze(-1))
    #     ori_angle = ori_angle[angle_mask]
    #     angle_output = angle_output[angle_mask]

    #     # dist_mask = ~(
    #     #     padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
    #     # )
    #     # dist_filter = ori_dist < 2.0
    #     # dist_mask = dist_mask & dist_filter

    #     # ori_dist = ori_dist[dist_mask]
    #     # dist = x_pair[dist_mask].squeeze(-1)
    #     # # dist = pair_pos[dist_mask].squeeze(-1)

    #     # dist_loss = self.loss_dist(ori_dist.to(torch.float32), dist.to(torch.float32))

    #     # compute loss
    #     angle_loss = self.loss_angle(
    #         angle_output.to(torch.float32), ori_angle.to(torch.float32)
    #     )
    #     pos_loss_m1 = self.loss_fn(Ca_pos_m1, ori_pos)
    #     pos_loss_m2 = self.loss_fn(Ca_pos_m2, ori_pos)
    #     pos_loss_m3 = self.loss_fn(Ca_pos_m3, ori_pos)
    #     pos_loss_m4 = self.loss_fn(Ca_pos_m4, ori_pos)
    #     loss = (
    #         pos_loss_m1
    #         + pos_loss_m2
    #         + pos_loss_m3
    #         + pos_loss_m4
    #         + angle_loss
    #         # + 10 * dist_loss
    #     )

    #     log_output = {
    #         "total_loss": loss,
    #         "loss_pos": pos_loss_m1,
    #         "loss_pos_m2": pos_loss_m2,
    #         "loss_pos_m3": pos_loss_m3,
    #         "loss_pos_m4": pos_loss_m4,
    #         "loss_angle": angle_loss,
    #         # "loss_dist": dist_loss,
    #     }

    #     return ModelOutput(loss=loss, num_examples=bs, log_output=log_output)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        structure = model_output[0]
        model_output[1]
        model_output[2]

        with torch.no_grad():
            ori_pos = batch_data["pos"]
            bs = ori_pos.shape[0]

            batch_data["ang"]
            batch_data["ang_mask"]

            batch_data["pos_mask"]

            aa_seq = batch_data["x"]
            (aa_seq).eq(1)  # B x T x 1

            # delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
            # ori_dist = delta_pos0.norm(dim=-1)

        loss, loss_breakdown = self.loss_fn(
            structure, batch_data, _return_breakdown=True
        )

        return ModelOutput(
            loss=loss, num_examples=bs, log_output=loss_breakdown
        )  # , log_output=log_output)

    def config_optimizer(self):
        optimizer, _ = myAdam(
            self,
            lr=self.args.max_lr,
            betas=[0.9, 0.999],
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )

        logger.info(f"Manually set total num steps: {self.args.total_num_steps}")
        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.args.total_num_steps,  # self.args.total_num_steps,
            warmup_max_lr=self.args.max_lr,
            warmup_num_steps=self.args.warmup_num_steps,  # self.args.warmup_num_steps,
            d_tilde=0.5,  # this is the ratio of the lr of the encoder to the head
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


@cli(DistributedTrainConfig, TOXConfig, StructureModuleConfig, DownstreamConfig)
def finetune(args) -> None:
    train_data, val_data = load_batched_dataset(args)

    basemodel = TOXModel(args, loss_fn=ProteinPMLM, load_ckpt=True)
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
