# -*- coding: utf-8 -*-
from typing import Optional

import torch
from torch import nn

from sfm.logging import logger
from sfm.models.tox.modules.tox_internal_encoder import ToxInternalEncoder
from sfm.modules.layer_norm import LayerNorm
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model


def load_pretrained_weights(model, checkpoint_path):
    """
    Load pretrained weights from a given state_dict.

    Args:
        args: Command line arguments.
        checkpoint_path: Path to the pretrained weights.
    """
    # if args.ft or args.infer:
    checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
    checkpoints_state = checkpoints_state["model"]

    IncompatibleKeys = model.load_state_dict(checkpoints_state, strict=False)
    IncompatibleKeys = IncompatibleKeys._asdict()

    missing_keys = [
        keys for keys in IncompatibleKeys["missing_keys"] if keys.find("dummy") == -1
    ]
    unexpected_keys = [
        keys for keys in IncompatibleKeys["unexpected_keys"] if keys.find("dummy") == -1
    ]

    if len(missing_keys) > 0:
        logger.info(f"Missing keys in {checkpoint_path}: {missing_keys}")

    if len(unexpected_keys) > 0:
        logger.info("Unexpected keys {checkpoint_path}: {unexpected_keys}")

    logger.info(f"checkpoint: {checkpoint_path} is loaded")


class TaskHeads(nn.Module):
    def __init__(self, args):
        super().__init__()
        # sequence type decoder
        self.seq_type_head = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(args.embedding_dim),
            nn.Linear(args.embedding_dim, args.num_residues),
        )

        # internal coordinate decoder
        self.bl_head = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(args.embedding_dim),
            nn.Linear(args.embedding_dim, 3),
        )
        self.ba_head = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(args.embedding_dim),
            nn.Linear(args.embedding_dim, 2 * 3),
        )
        self.da_head = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.GELU(),
            nn.Linear(args.embedding_dim, 2 * 3),
        )

    def forward(self, batched_data):
        logits = batched_data["output"]["final_state"]
        seq_logits = self.seq_type_head(logits)
        batched_data["output"]["seq_logits"] = seq_logits

        bl_logits = self.bl_head(logits)
        ba_logits = self.ba_head(logits)
        da_logits = self.da_head(logits)
        batched_data["output"]["bl_logits"] = bl_logits
        batched_data["output"]["ba_logits"] = ba_logits
        batched_data["output"]["da_logits"] = da_logits
        return batched_data


class ToxInternalModel(Model):
    def __init__(self, args, loss_fn=None):
        super().__init__()
        self.args = self.check_args(args)
        self.encoder = ToxInternalEncoder(args)
        self.norm_after = LayerNorm(args.embedding_dim)
        self.task_head = TaskHeads(args)

        self.loss = loss_fn(args)
        if args.rank == 0:
            logger.info(args)

        if args.load_ckpt_from:
            load_pretrained_weights(self, args.load_ckpt_from)
        else:
            logger.info("No checkpoint is loaded")

    def check_args(self, args):
        required_lst = [
            "load_ckpt_from",
            "embedding_dim",
            "num_residues",
        ]
        for k in required_lst:
            assert hasattr(
                args, k
            ), f"args should have {k} attribute in {self.__class__.__name__}"
        return args

    def compute_loss(self, batched_data, _) -> ModelOutput:
        loss_dict = self.loss(batched_data)
        tot_loss = loss_dict.pop("tot_loss")
        return ModelOutput(
            loss=tot_loss,
            num_examples=batched_data["output"]["final_state"].shape[0],
            log_output=loss_dict,
        )

    def forward(
        self,
        batched_data: dict,
        # padding_mask: Optional[torch.Tensor] = None,
        # last_state_only: bool = True, # TODO: mabe we can use it
    ):
        # batched_data
        # input is legacy key, we now convert it to crab for input, may be we can delete it.
        # input:
        #   aa int64 [B, L] pos, pos_mask, ang, ang_mask, name, size
        # crab: do not have cls and eos token
        #   C int64 [B, L], chain identifier
        #   R int64 [B, L], amino acid token from Alphabet()
        #   A float32 [B, L, 4, 3], coordinates of protein backbone, in order of [N, CA, C, O]
        # internal:
        #   bl_N_CA float32 [B, L], bond length between N and CA, first N_res are valid
        #   bl_CA_C float32 [B, L], bond length between CA and C, first N_res are valid
        #   bl_C_N float32 [B, L], bond length between C and N, first N_res-1 are valid
        #   ba_C_N_CA float32 [B, L], bond angle between C, N, CA, first N_res-1 are valid
        #   ba_N_CA_C float32 [B, L], bond angle between N, CA, C, first N_res are valid
        #   ba_CA_C_N float32 [B, L], bond angle between CA, C, N, first N_res-1 are valid
        #   da_CA_C_N_CA float32 [B, L], dihedral angle between CA, C, N, CA, first N_res-1 are valid
        #   da_C_N_CA_C float32 [B, L], dihedral angle between C, N, CA, C, first N_res-1 are valid
        #   da_N_CA_C_N float32 [B, L], dihedral angle between N, CA, C, N, first N_res-1 are valid
        # mask:
        #   mask_seq bool [B, L]
        #   mask_str bool [B, L]
        # NOTE: L = max_num_residues + 2

        padding_mask = batched_data["crab"]["padding_mask"]
        x, inner_states = self.encoder(
            batched_data,
            padding_mask=padding_mask,
            attn_mask=None,
            last_state_only=True,
        )

        batched_data["output"] = dict()
        x = self.norm_after(x)
        batched_data["output"]["final_state"] = x
        batched_data["output"]["inner_states"] = inner_states
        batched_data = self.task_head(batched_data)
        return batched_data

    def config_optimizer(self):
        pass
