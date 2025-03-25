# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.data.psm_data.utils import VOCAB
from sfm.logging import logger
from sfm.models.psm.equivariant.e2former import E2former
from sfm.models.psm.equivariant.equiformer.graph_attention_transformer import Equiformer
from sfm.models.psm.equivariant.equiformer_series import Equiformerv2SO2
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.equivariant.nodetaskhead import (
    AADiffusionModule,
    AAVectorProjOutput,
    ConditionVectorGatedOutput,
    DiffusionModule,
    DiffusionModule2,
    DiffusionModule3,
    ForceGatedOutput,
    ForceVecOutput,
    InvariantDiffusionModule,
    MSADiffusionModule,
    NodeTaskHead,
    ScalarGatedOutput,
    VectorGatedOutput,
    VectorOutput,
    VectorProjOutput,
)
from sfm.models.psm.equivariant.vectorVT import VectorVanillaTransformer
from sfm.models.psm.invariant.dit_encoder import PSMDiTEncoder
from sfm.models.psm.invariant.ditp_encoder import PSMPDiTPairEncoder
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.invariant.plain_encoder import (
    MSAGenEncoder,
    PSMPairPlainEncoder,
    PSMPlainEncoder,
)
from sfm.models.psm.modules.diffusion import Diffsuion_LM
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.modules.mixembedding import (
    MSAGenSeqEmbedding,
    ProteaEmbedding,
    PSMLightEmbedding,
    PSMLightPEmbedding,
    PSMMix3dDitEmbedding,
    PSMMix3dEmbedding,
    PSMMixSeqEmbedding,
    PSMSeqEmbedding,
)
from sfm.models.psm.modules.mixembedding_equiv import PSMMix3DEquivEmbedding
from sfm.models.psm.modules.pbc import CellExpander
from sfm.models.psm.psm_config import ForceHeadType, GaussianFeatureNodeType, PSMConfig
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.autograd import GradientHead
from .modules.confidence_model import lddt
from .modules.dataaug import uniform_random_rotation
from .modules.diffusion import DIFFUSION_PROCESS_REGISTER
from .modules.sampled_structure_converter import SampledStructureConverter
from .modules.timestep_encoder import (
    DiffNoise,
    DiffNoiseEDM,
    NoiseStepSamplerEDM,
    TimeStepSampler,
)


class MSAGenModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        loss_fn=None,
        not_init=False,
        psm_finetune_head: nn.Module = None,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
        periodic_stress_mean=0.0,
        periodic_stress_std=1.0,
        reload_checkpoint=True,
    ):
        """
        Initialize the PSMModel class.

        Args:
            args: Command line arguments.
            loss_fn: The loss function to use.
            data_mean: The mean of the label. For label normalization.
            data_std: The standard deviation of the label. For label normalization.
            not_init: If True, the model will not be initialized. Default is False.
            psm_finetune_head: head used to finetune psm
        """

        super().__init__()
        if not_init:
            return
        self.cut_off = 64
        self.psm_config = PSMConfig(args)
        self.args = self.psm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.net = MSAGen(
            args,
            self.psm_config,
        )
        self.T = 1000
        self.diffusion = Diffsuion_LM(self.T)
        self.psm_finetune_head = psm_finetune_head

        if reload_checkpoint:
            self.checkpoint_loaded = self.reload_checkpoint()
        self.aa_mlm_loss = nn.CrossEntropyLoss(reduction="mean")
        # self.loss_fn = loss_fn(args)

    def reload_checkpoint(self):
        if self.psm_config.psm_finetune_mode or self.psm_config.psm_validation_mode:
            if os.path.exists(self.args.loadcheck_path):
                self.load_pretrained_weights(
                    self.args, checkpoint_path=self.args.loadcheck_path
                )
                loaded = True
                logger.info(f"checkpoint: {self.args.loadcheck_path} is loaded")
            else:
                logger.warning(
                    "Finetune or validation mode, but no checkpoint is loaded"
                )
                loaded = False
        else:
            logger.info("No checkpoint is loaded")
            loaded = False
        if self.psm_config.psm_finetune_mode:
            settings = dict(
                psm_finetune_reset_head=self.psm_config.psm_finetune_reset_head,
                psm_finetune_head=(
                    self.psm_finetune_head.__class__ if self.psm_finetune_head else None
                ),
                psm_finetune_noise_mode=self.psm_config.psm_finetune_noise_mode,
            )
            logger.info(f"Finetune settings: {settings}")
            if self.psm_config.psm_finetune_reset_head:
                self.net.reset_head_for_finetune()
        else:
            assert not self.psm_finetune_head
            self.psm_finetune_head = None

        return loaded

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.

        Args:
            args: Command line arguments.
            checkpoint_path: Path to the pretrained weights.
        """
        checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        for key in list(checkpoints_state.keys()):
            if key.startswith("base."):
                checkpoints_state[key[5:]] = checkpoints_state.pop(key)

        IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
        IncompatibleKeys = IncompatibleKeys._asdict()

        missing_keys = []
        for keys in IncompatibleKeys["missing_keys"]:
            if keys.find("dummy") == -1:
                missing_keys.append(keys)

        unexpected_keys = []
        for keys in IncompatibleKeys["unexpected_keys"]:
            if keys.find("dummy") == -1:
                unexpected_keys.append(keys)

        if len(missing_keys) > 0:
            logger.info(
                "Missing keys in {}: {}".format(
                    checkpoint_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            logger.info(
                "Unexpected keys {}: {}".format(
                    checkpoint_path,
                    unexpected_keys,
                )
            )

        logger.info(f"checkpoint: {checkpoint_path} is loaded")

    def max_positions(self):
        """
        Returns the maximum positions of the net.
        """
        return self.net.max_positions

    @torch.no_grad()
    def sample(self, batched_data):
        """
        Sapmle mathod for diffusion model
        """
        self.net.eval()
        B, L = batched_data["token_type"].size()
        device = batched_data["token_type"].device
        token_id = batched_data["token_type"]
        padding_mask = token_id.eq(0)
        batched_data["padding_mask"] = padding_mask
        batched_data["aa_mask"] = torch.zeros_like(
            token_id, dtype=torch.bool, device=device
        )
        samples = []
        for sample_time_index in range(self.psm_config.num_sampling_time):
            batched_data["128_msa_one_hot"] = torch.zeros(B, self.cut_off, L, 27)
            padding_mask_2D = (
                batched_data["token_type"].eq(0).repeat(1, self.cut_off, 1, 1)
            )
            clean_mask = torch.zeros(
                B, self.cut_off, L, dtype=torch.bool, device=device
            )
            clean_mask = clean_mask.masked_fill(padding_mask_2D, True)
            batched_data["clean_mask"] = clean_mask
            T = torch.full((B,), self.T, device=device)
            x_T = self.diffusion.q_sample(
                batched_data["128_msa_one_hot"], T, clean_mask, device
            )
            batched_data["128_msa_one_hot"] = x_T
            batched_data["time_step"] = T
            for t in reversed(range(1, self.T)):
                net_result = self.net(batched_data)
                t = torch.full((B,), t, device=device)
                x_t = self.diffusion.q_sample(
                    net_result["x0_pred"], t, clean_mask, device
                )
                batched_data["128_msa_ont_hot"] = x_t
                batched_data["time_step"] = T
            samples.append(x_t)
        return torch.stack(samples, dim=0)

    def _set_noise(self, batched_data):
        B, D, L = batched_data["msa_token_type"].shape
        min_D = min(D, batched_data["cut_off"])
        device = batched_data["msa_token_type"].device
        t = torch.randint(0, 1000, (B,), device=device)
        # set aa_mask and padding to clean
        clean_mask = (batched_data["aa_mask"]).repeat(1, min_D, 1) | batched_data[
            "128_2D_padding_mask"
        ]
        x_t = self.diffusion.q_sample(
            batched_data["128_msa_one_hot"],
            t,
            clean_mask,
            device,
        )
        batched_data["ori_128_msa_one_hot"] = batched_data["128_msa_one_hot"].clone()
        batched_data["128_msa_one_hot"] = x_t
        batched_data["time_step"] = t
        batched_data["clean_mask"] = clean_mask
        # return x_t,t

    def _pre_forward_operation(
        self,
        batched_data,
    ):
        """
        pre forward operation
        """
        # set padding_mask
        cut_off = 64
        batched_data["cut_off"] = 64
        token_id = batched_data["token_type"]
        padding_mask = token_id.eq(0)  # B x T x 1
        B, D, L = batched_data["msa_token_type"].shape
        msa_token_type = batched_data["msa_token_type"]
        batched_data["padding_mask"] = padding_mask
        batched_data["row_padding_mask"] = (msa_token_type == 0).all(dim=-1)
        batched_data["col_padding_mask"] = (msa_token_type == 0).all(dim=1)
        batched_data["2D_padding_mask"] = msa_token_type == 0
        batched_data["128_msa_token_type"] = batched_data["msa_token_type"][
            :, :cut_off, :
        ]
        batched_data["128_msa_one_hot"] = F.one_hot(
            batched_data["128_msa_token_type"].long(), num_classes=27
        ).float()  # 26 plus <pad>
        batched_data["128_row_padding_mask"] = batched_data["row_padding_mask"][
            :, :cut_off
        ]
        batched_data["128_col_padding_mask"] = batched_data["col_padding_mask"][
            :, :cut_off
        ]
        batched_data["128_2D_padding_mask"] = batched_data["2D_padding_mask"][
            :, :cut_off, :
        ]
        # set aa_mask
        mask_ratio = 0.15
        aa_mask = torch.rand_like(token_id, dtype=torch.float) < mask_ratio
        aa_mask = aa_mask & ~padding_mask
        batched_data["aa_mask"] = aa_mask

        # calculate true prob
        msa_token_type_t = batched_data["msa_token_type"].transpose(1, 2)  # B L D

        counts = torch.zeros(
            B, L, 26, device=batched_data["msa_token_type"].device, dtype=torch.int32
        )
        indices = (msa_token_type_t - 1).clamp(
            min=0
        )  # B L D minus 1 so that 0 means indicates=0 which indicates the first aa
        valid_mask = msa_token_type_t.ne(0)  # B L D
        # count num of valid according indices
        counts.scatter_add_(2, indices.long(), valid_mask.int())
        true_prob = counts / valid_mask.int().sum(dim=-1, keepdim=True).clamp(min=1)
        batched_data["true_prob"] = true_prob
        self._set_noise(batched_data)

    def _KL_reconstruction_loss(
        x0, x0_pred, x_t, t, beta_t, beta_t_pre, alpha_t, alpha_t_bar, alpha_t_pre_bar
    ):
        one_minus_alpha_t_bar = 1 - alpha_t_bar
        one_minus_alpha_t_pre_bar = 1 - alpha_t_pre_bar
        sqrt_alpha_t_pre_bar = torch.sqrt(alpha_t_pre_bar)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        x_t_pre_pred = (
            sqrt_alpha_t_pre_bar * beta_t * x0_pred / one_minus_alpha_t_bar
            + sqrt_alpha_t * one_minus_alpha_t_pre_bar * x_t / one_minus_alpha_t_bar
        )

        x_t_pre = (
            sqrt_alpha_t_pre_bar * beta_t * x0 / one_minus_alpha_t_bar
            + sqrt_alpha_t * one_minus_alpha_t_pre_bar * x_t / one_minus_alpha_t_bar
        )

        theta_t_square = one_minus_alpha_t_pre_bar * beta_t / one_minus_alpha_t_bar

        return F.kl_div(x_t_pre_pred, x_t_pre, reduction="batchmean") / (
            2 * theta_t_square
        )

    def _forward_net(self, batched_data, skip_sample=False, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            skip_sample: Skip the sampling step.
            **kwargs: Additional keyword
        """

        if self.psm_config.psm_sample_structure_in_finetune:
            self.net.eval()

        # context = torch.no_grad() if self.psm_config.freeze_backbone else nullcontext()
        # with context:
        result_dict = self.net(
            batched_data,
            **kwargs,
        )

        result_dict["data_name"] = (
            batched_data["data_name"] if "data_name" in batched_data else None
        )

        result_dict["position_ids"] = batched_data["position_ids"]

        return result_dict

    def forward(self, batched_data, skip_sample=False, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """

        self._pre_forward_operation(batched_data)

        if (
            self.psm_config.sample_in_validation
            and not self.training
            and not skip_sample
        ):
            match_results = self.sample_and_calc_match_metric(batched_data)

        result_dict = self._forward_net(batched_data, skip_sample, **kwargs)

        if (
            self.psm_config.sample_in_validation
            and not self.training
            and not skip_sample
        ):
            result_dict.update(match_results)

        if self.psm_finetune_head and not self.psm_config.sample_in_validation:
            if self.psm_config.psm_sample_structure_in_finetune:
                self.eval()
                if (
                    self.psm_config.diffusion_mode == "edm"
                    and self.psm_config.diffusion_sampling == "edm"
                ):
                    random.random()
                    # if select < 0.6:
                    #     self.psm_config.edm_sample_num_steps = 12
                    # elif select < 0.7:
                    #     self.psm_config.edm_sample_num_steps = 11
                    # else:
                    # self.psm_config.edm_sample_num_steps = 20

                    sampled_output = self.sample_AF3(batched_data)
                else:
                    sampled_output = self.sample(batched_data)

                for k, v in sampled_output.items():
                    result_dict[k + "_sample"] = v
                self.train()

            result_dict = self.psm_finetune_head(result_dict)

        return result_dict

    def compute_loss(self, model_output, batched_data) -> ModelOutput:
        """
        Compute loss for the model.

        Args:
            model_output: The output from the model.
            batched_data: The batch data.

        Returns:
            ModelOutput: The model output which includes loss, log_output, num_examples.
        """

        kl_loss = F.kl_div(
            model_output["model_log_prob"], model_output["true_prob"], reduction="none"
        )
        kl_loss = kl_loss.sum(dim=-1)
        mask = ~model_output["padding_mask"]
        kl_loss = (kl_loss * mask).sum() / mask.sum()
        if batched_data["aa_mask"].any():
            aa_mask = batched_data["aa_mask"]
        logits = model_output["aa_logits"][aa_mask]
        aa_mlm_loss = self.aa_mlm_loss(
            logits,
            batched_data["token_type"][aa_mask].long(),
        )

        filter_mask = ~batched_data["clean_mask"]  # B D L
        B, D, L = batched_data["128_2D_padding_mask"].size()
        cross_entropy_loss = self.compute_cross_entropy_loss(
            model_output["x0_pred"][filter_mask],
            batched_data["ori_128_msa_one_hot"][filter_mask].argmax(dim=-1),
            # batched_data["ori_128_msa_one_hot"].argmax(dim=-1).unsqueeze(-1).view(B,D*L,-1),
            filter_mask,
        )
        # l1_loss = self.compute_l1_loss(
        #     model_output["x0_pred"], batched_data["ori_128_msa_one_hot"]
        # )
        loss = aa_mlm_loss + cross_entropy_loss + kl_loss
        # loss += cross_entropy_loss
        logging_output = {
            "total_loss": float(loss.detach()),
            "cross_entropy_loss": float(cross_entropy_loss.detach()),
            "KL_loss": float(kl_loss.detach()),
            "aa_mlm_loss": float(aa_mlm_loss.detach()),
        }

        return ModelOutput(
            loss=loss,
            num_examples=model_output["model_prob"].shape[0],
            log_output=logging_output,
        )

    def compute_cross_entropy_loss(self, logits, target, filter_mask):
        """
        compute cross entropy loss
        """

        # log_prob = F.log_softmax(logits.float(), dim=-1)  # B,D,L,num_classes
        # loss = -(target * log_prob).sum(dim=-1)
        # loss = loss[filter_mask]
        # B, D, L, C = logits.size()
        # logits = logits.view(B, D * L, C).float().permute(0, 2, 1)
        # target = target.view(B, D * L)
        loss = self.aa_mlm_loss(logits, target)
        return loss

    # def compute_l1_loss(self, logits, target):
    #     """
    #     compute L1 loss
    #     """
    #     return F.l1_loss(logits, target, reduction="mean")

    def config_optimizer(self, model: Optional[nn.Module]):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        return (None, None)


class MSAGen(nn.Module):
    """
    Class for training Physics science module
    """

    def __init__(
        self,
        args,
        psm_config: PSMConfig,
    ):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args
        self.backbone = args.backbone

        self.psm_config = psm_config
        self.diffusion_num_steps = 1000
        self.embedding = MSAGenSeqEmbedding(psm_config)

        self.encoder = MSAGenEncoder(args, psm_config)

        self.x_proj = nn.Sequential(
            nn.Linear(
                psm_config.embedding_dim, psm_config.embedding_dim // 2, bias=False
            ),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim // 2, 26, bias=False),
        )
        self.aa_mask_head = nn.Sequential(
            nn.Linear(
                psm_config.embedding_dim, psm_config.embedding_dim // 2, bias=False
            ),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim // 2, 30, bias=False),
        )

        self.decoder = MSADiffusionModule(args, psm_config, self.diffusion_num_steps)

    def forward(
        self,
        batched_data,
        **unused,
    ):
        """
        Forward pass for PSM. This first computes the token

        Args:
            - batched_data: keys need to be defined in the data module
        Returns:
            - need to be defined
        """
        token_embedding = self.embedding(
            batched_data["token_type"],
            batched_data["aa_mask"],
            batched_data["padding_mask"],
        )

        encoder_x = self.encoder(
            token_embedding.transpose(0, 1),
            batched_data["padding_mask"],
            batched_data,
        )

        # msa_embedding = self.embedding(batched_data["128_msa_token_type"])
        decoder_x = self.x_proj(encoder_x)
        msa_embedding = batched_data["128_msa_one_hot"]

        x0_pred = self.decoder(
            batched_data,
            msa_embedding,
            encoder_x.transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, msa_embedding.shape[1], 1, 1),
            batched_data["128_2D_padding_mask"],
        )
        # print(decoder_x.shape)
        # x0_pred = (
        #     decoder_x.transpose(0, 1)
        #     .unsqueeze(1)
        #     .repeat(1, msa_embedding.shape[1], 1, 1)
        # )

        model_prob = F.softmax(decoder_x.transpose(0, 1), dim=-1)
        model_log_prob = F.log_softmax(
            decoder_x.transpose(0, 1), dim=-1
        )  # calculate kl loss which needs log softmax first
        aa_logits = self.aa_mask_head(encoder_x)
        result_dict = {
            "x0_pred": x0_pred,
            "aa_logits": aa_logits.transpose(0, 1),
            "true_prob": batched_data["true_prob"],
            "model_prob": model_prob,
            "model_log_prob": model_log_prob,
            "decoder_x": decoder_x.transpose(0, 1),
            "padding_mask": batched_data["padding_mask"],
        }
        return result_dict

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
