# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from contextlib import nullcontext
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

        self.psm_config = PSMConfig(args)
        self.args = self.psm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.net = MSAGen(
            args,
            self.psm_config,
            molecule_energy_per_atom_std=molecule_energy_per_atom_std,
            periodic_energy_per_atom_std=periodic_energy_per_atom_std,
            molecule_force_std=molecule_force_std,
            periodic_force_std=periodic_force_std,
            periodic_stress_mean=periodic_stress_mean,
            periodic_stress_std=periodic_stress_std,
        )

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

        context = torch.no_grad() if self.psm_config.freeze_backbone else nullcontext()
        with context:
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
        loss = kl_loss + aa_mlm_loss
        logging_output = {
            "total_loss": float(loss.detach()),
            "KL_loss": float(kl_loss.detach()),
            "aa_mlm_loss": float(aa_mlm_loss.detach()),
        }
        # loss, logging_output = self.loss_fn(model_output, batched_data)
        if (
            self.psm_finetune_head
            and hasattr(self.psm_finetune_head, "update_loss")
            and self.training
        ):
            loss, logging_output = self.psm_finetune_head.update_loss(
                loss, logging_output, model_output, batched_data
            )

        return ModelOutput(
            loss=loss,
            num_examples=model_output["model_prob"].shape[0],
            log_output=logging_output,
        )

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
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
        periodic_stress_mean=0.0,
        periodic_stress_std=1.0,
    ):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args
        self.backbone = args.backbone

        self.psm_config = psm_config

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

    def plot_probability_heatmaps(
        self, true_prob, pred_prob, padding_mask, batched_data
    ):
        """
        绘制真实概率分布、模型预测概率分布和它们的差异热图，并保存到磁盘。

        参数：
        true_prob: numpy 数组或 torch.Tensor，形状 (B, L, 26)，真实概率分布，经过 softmax 后的概率。
        pred_prob: numpy 数组或 torch.Tensor，形状 (B, L, 26)，模型预测概率分布（经过 softmax）。
        padding_mask: numpy 数组或 torch.Tensor，形状 (B, L)，布尔类型，其中 True 表示该位置为 padding（无效）。
        batched_data: 包含其他信息，如 "unique_ids"，用于标识每个样本。
        """
        save_dir = "./output/vis"
        os.makedirs(save_dir, exist_ok=True)

        # 如果输入是 torch.Tensor，则转换为 numpy 数组
        if hasattr(true_prob, "detach"):
            true_prob = true_prob.detach().cpu().numpy()
        if hasattr(pred_prob, "detach"):
            pred_prob = pred_prob.detach().cpu().numpy()
        if hasattr(padding_mask, "detach"):
            padding_mask = padding_mask.detach().cpu().numpy()

        B, L, num_classes = true_prob.shape
        # 计算差异热图：预测 - 真实
        diff = pred_prob - true_prob

        # 类别标签：26个类别
        class_labels = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
            "X",
            "B",
            "U",
            "Z",
            "O",
            "-",
        ]

        # 遍历每个样本
        for i in range(B):
            unique_id = batched_data["unique_ids"][i]
            # 获取当前样本对应的概率分布和差异
            sample_true = true_prob[i]  # (L, 26)
            sample_pred = pred_prob[i]  # (L, 26)
            sample_diff = diff[i]  # (L, 26)
            # 扩展 padding_mask: 原始 padding_mask[i] 形状为 (L,)
            # 这里 pad_mask 中 True 表示无效，将其扩展到 (L,26)
            sample_mask = np.broadcast_to(padding_mask[i][:, None], sample_true.shape)

            # 绘制真实概率分布热图
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                sample_true, mask=sample_mask, cmap="viridis", xticklabels=class_labels
            )
            ax.set_title(f"True Probability Distribution for Sample {unique_id}")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Sequence Position")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"heatmap_true_{unique_id}.png")
            plt.savefig(save_path)
            plt.close()

            # 绘制预测概率分布热图
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                sample_pred, mask=sample_mask, cmap="viridis", xticklabels=class_labels
            )
            ax.set_title(f"Predicted Probability Distribution for Sample {unique_id}")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Sequence Position")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"heatmap_pred_{unique_id}.png")
            plt.savefig(save_path)
            plt.close()

            # 绘制差异热图（预测 - 真实）
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                sample_diff,
                mask=sample_mask,
                cmap="coolwarm",
                center=0,
                xticklabels=class_labels,
            )
            ax.set_title(f"Difference Heatmap for Sample {unique_id}")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Sequence Position")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"heatmap_diff_{unique_id}.png")
            plt.savefig(save_path)
            plt.close()

            print(f"Saved heatmaps for sample {unique_id} to {save_dir}")

    def _pre_forward_operation(
        self,
        batched_data,
    ):
        """
        pre forward operation
        """
        # set padding_mask
        token_id = batched_data["token_type"]
        padding_mask = token_id.eq(0)  # B x T x 1
        B, D, L = batched_data["msa_token_type"].shape
        msa_token_type = batched_data["msa_token_type"]
        batched_data["padding_mask"] = padding_mask
        batched_data["row_padding_mask"] = (msa_token_type == 0).all(dim=-1)
        batched_data["col_padding_mask"] = (msa_token_type == 0).all(dim=1)
        batched_data["2D_padding_mask"] = msa_token_type == 0
        batched_data["128_msa_token_type"] = batched_data["msa_token_type"][:, :128, :]
        batched_data["128_row_padding_mask"] = batched_data["row_padding_mask"][:, :128]
        batched_data["128_col_padding_mask"] = batched_data["col_padding_mask"][:, :128]
        batched_data["128_2D_padding_mask"] = batched_data["2D_padding_mask"][
            :, :128, :
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
        self._pre_forward_operation(batched_data)

        token_embedding = self.embedding(
            batched_data, batched_data["aa_mask"], batched_data["padding_mask"]
        )

        encoder_x = self.encoder(
            token_embedding.transpose(0, 1), batched_data["padding_mask"], batched_data
        )
        decoder_x = self.x_proj(encoder_x)

        model_prob = F.softmax(decoder_x.transpose(0, 1), dim=-1)
        model_log_prob = F.log_softmax(
            decoder_x.transpose(0, 1), dim=-1
        )  # calculate kl loss which needs log softmax first
        aa_logits = self.aa_mask_head(encoder_x)
        result_dict = {
            "aa_logits": aa_logits.transpose(0, 1),
            "true_prob": batched_data["true_prob"],
            "model_prob": model_prob,
            "model_log_prob": model_log_prob,
            "decoder_x": decoder_x.transpose(0, 1),
            "padding_mask": batched_data["padding_mask"],
        }
        # self.plot_probability_heatmaps(
        #     true_prob.detach().cpu().numpy(),
        #     model_prob.detach().cpu().numpy(),
        #     padding_mask.detach().cpu().numpy(),
        #     batched_data,
        # )
        return result_dict

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
