# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.multihead_attention import MultiheadAttention
from sfm.modules.quant_noise import quant_noise as apply_quant_noise_
from sfm.utils import LayerDropModuleList

from .pfm_embedding import PFMEmbedding
from .pfm_encoder_layer import PFMEncoderLayer


def init_params(module):
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class PFMEncoder(nn.Module):
    def __init__(
        self,
        pfm_config,
        init_bias: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        # num_pred_attn_layer: int = 4,
    ) -> None:
        super().__init__()
        args = pfm_config.args
        self.pfm_config = pfm_config
        self.dropout_module = FairseqDropout(
            pfm_config.dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = pfm_config.layerdrop
        self.max_seq_len = pfm_config.max_seq_len
        self.embedding_dim = pfm_config.embedding_dim
        self.ffn_embedding_dim = pfm_config.ffn_embedding_dim
        self.num_attention_heads = pfm_config.num_attention_heads
        self.num_segments = pfm_config.num_segments
        self.use_position_embeddings = pfm_config.use_position_embeddings
        self.apply_bert_init = pfm_config.apply_bert_init
        self.learned_pos_embedding = pfm_config.learned_pos_embedding

        if init_bias:
            self.pfm_emb = PFMEmbedding(pfm_config)

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if pfm_config.encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        droppath_probs = [
            x.item()
            for x in torch.linspace(
                0, pfm_config.droppath_prob, pfm_config.num_encoder_layers
            )
        ]

        for nl in range(pfm_config.num_encoder_layers):
            self.layers.extend(
                [
                    self.build_transformer_sentence_encoder_layer(
                        embedding_dim=pfm_config.embedding_dim,
                        ffn_embedding_dim=pfm_config.ffn_embedding_dim,
                        num_attention_heads=pfm_config.num_attention_heads,
                        dropout=self.dropout_module.p,
                        attention_dropout=pfm_config.attention_dropout,
                        activation_dropout=pfm_config.activation_dropout,
                        activation_fn=pfm_config.activation_fn,
                        export=export,
                        q_noise=q_noise,
                        qn_block_size=qn_block_size,
                        sandwich_ln=pfm_config.sandwich_ln,
                        droppath_prob=droppath_probs[nl],
                        nl=nl,
                        args=args,
                        pfm_config=pfm_config,
                    )
                ]
            )

        init_scale = math.sqrt(math.log(pfm_config.num_encoder_layers))
        for name, p in self.named_parameters():
            if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                p.data.mul_(init_scale)

        self.args = args
        try:
            mode_prob = [float(item) for item in pfm_config.mode_prob.split(",")]
            assert len(mode_prob) == 3
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.4, 0.3, 0.3]
        self.mode_prob = mode_prob

        self.num_timesteps = args.num_timesteps
        assert args.ddpm_schedule in ["linear", "quadratic", "sigmoid", "cosine"]
        (
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        ) = self._beta_schedule(
            args.num_timesteps,
            args.ddpm_beta_start,
            args.ddpm_beta_end,
            args.ddpm_schedule,
        )

    def _beta_schedule(
        self, num_timesteps, beta_start, beta_end, schedule_type="sigmoid"
    ):
        if schedule_type == "linear":
            beta_list = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "quadratic":
            beta_list = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
            )
        elif schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, num_timesteps)
            beta_list = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif schedule_type == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta_list = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError

        alphas = 1 - beta_list
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _noise_sample(self, x_start, t):
        noise = torch.randn_like(x_start) * 1.0

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _set_noise(self, ori_pos, mask_pos):
        if self.pfm_config.noise_mode == "mae":
            return ori_pos, None
        elif self.pfm_config.noise_mode == "const":
            noise = (
                torch.randn(ori_pos.shape, device=ori_pos.device)
                * self.args.noise_scale
            )
            noise = noise.masked_fill_(~mask_pos.bool(), 0.0)
            pos = ori_pos + noise
            return pos, None
        elif self.pfm_config.noise_mode == "diff":
            time = torch.randint(
                0, self.t_timesteps, (ori_pos.shape[0],), device=ori_pos.device
            ).long()

            noisy_pos = (
                self._noise_sample(ori_pos, time)
                .masked_fill(~mask_pos.bool(), 0.0)
                .to(ori_pos.dtype)
            )
            vis_pos = ori_pos.masked_fill(mask_pos.bool(), 0.0).to(ori_pos.dtype)
            pos = noisy_pos + vis_pos
            return pos, time
        else:
            raise Exception(
                f"noise mode {self.pfm_config.noise_mode} not implemented, please choose from ['const', 'diff']"
            )

    def _set_mask(self, mask_aa, mask_pos, residue_seq):
        n_graph, n_node = residue_seq.size()[:2]

        # 1 is pad token, 2 is eos token
        padding_mask = (residue_seq[:, :]).eq(1)  # B x T x 1
        eos_mask = (residue_seq[:, :]).eq(2)

        # # 0:  mask_aa and mask_pos are the same
        # # 1:  mask_aa is full and no mask_pos
        # # 2:  mask_pos is full and no mask_aa
        mask_choice = np.random.choice(np.arange(3), n_graph, p=self.mode_prob)
        mask = torch.tensor([i for i in mask_choice]).to(residue_seq.device)
        mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1, n_node, 1)  # [ngraph, nnode+1]
        mask_pos = torch.where(mask == 0, mask_aa, mask_pos)
        mask_pos = torch.where(mask == 1, False, mask_pos)
        mask_pos = torch.where(mask == 2, True, mask_pos)
        mask_aa = torch.where(mask == 1, True, mask_aa)
        mask_aa = torch.where(mask == 2, False, mask_aa)

        # # cls token should not be masked
        mask_aa[:, 0, :] = False
        mask_pos[:, 0, :] = False
        mask_aa = mask_aa.masked_fill_(padding_mask.bool().unsqueeze(-1), False)
        mask_aa = mask_aa.masked_fill_(eos_mask.bool().unsqueeze(-1), False)
        mask_pos = mask_pos.masked_fill_(padding_mask.bool().unsqueeze(-1), False)
        mask_pos = mask_pos.masked_fill_(eos_mask.bool().unsqueeze(-1), False)

        return mask_aa, mask_pos, padding_mask, eos_mask

    def _set_bert_mask_strategy(self, mask_aa, mask_pos, residue_seq, x_0):
        n_graph, n_node = residue_seq.size()[:2]

        cls_mask = (x_0[:, :]).eq(0)
        padding_mask = (x_0[:, :]).eq(1)  # B x T x 1
        eos_mask = (x_0[:, :]).eq(2)
        comma_mask = (x_0[:, :]).eq(29)
        # line_mask = (x_0[:, :]).eq(30)

        mask_aa = mask_aa.masked_fill(cls_mask.bool().unsqueeze(-1), False)
        mask_aa = mask_aa.masked_fill(padding_mask.bool().unsqueeze(-1), False)
        mask_aa = mask_aa.masked_fill(eos_mask.bool().unsqueeze(-1), False)
        mask_aa = mask_aa.masked_fill(comma_mask.bool().unsqueeze(-1), False)
        # mask_aa = mask_aa.masked_fill(line_mask.bool().unsqueeze(-1), False)

        # enable pair mask in msa data
        for i in range(n_graph):
            mask_start_idx = (x_0[i] == 0).nonzero(as_tuple=True)[0]
            mask_end_idx = (x_0[i] == 2).nonzero(as_tuple=True)[0]
            assert len(mask_start_idx) >= len(
                mask_end_idx
            ), f"length of mask_start_idx: {mask_start_idx} should be larger than mask_end_idx: {mask_start_idx}"
            for j in range(len(mask_end_idx)):
                s_idx = mask_start_idx[j]
                e_idx = mask_end_idx[j]

                mask_comma_idx = (x_0[i][s_idx:e_idx] == 29).nonzero(as_tuple=True)[0]

                if len(mask_comma_idx) == 1:
                    c_idx = mask_comma_idx[0]
                    if e_idx - s_idx == 2 * c_idx:
                        mask_aa[i, s_idx + c_idx + 1 : e_idx] = mask_aa[
                            i, s_idx + 1 : s_idx + c_idx
                        ]
                        temp = x_0[i, s_idx + c_idx + 1 : e_idx]
                        temp_mask = mask_aa[i, s_idx + c_idx + 1 : e_idx].squeeze(-1)
                        temp[temp_mask] = 31
                        residue_seq[i, s_idx + c_idx + 1 : e_idx] = temp
                elif len(mask_comma_idx) > 1:
                    torch.set_printoptions(profile="full")
                    print(residue_seq[i][s_idx : e_idx + 1])
                    raise ValueError(
                        f"comma index error, number of comma is {len(mask_comma_idx)}, check the data"
                    )

        residue_seq = torch.where(cls_mask | eos_mask | comma_mask, x_0, residue_seq)

        return mask_aa, cls_mask, padding_mask, eos_mask, residue_seq

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        sandwich_ln,
        droppath_prob,
        nl,
        args,
        pfm_config=None,
    ):
        return PFMEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            sandwich_ln=sandwich_ln,
            droppath_prob=droppath_prob,
            nl=nl,
            args=args,
            pfm_config=pfm_config,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention

        with torch.no_grad():
            if "x_new" in batched_data.keys():
                residue_seq = batched_data["x_new"].clone()
            else:
                residue_seq = batched_data["x"].clone()

            x_0 = batched_data["x"].clone()
            mask_aa = batched_data["masked_aa"]
            mask_pos = batched_data["mask_pos"]

        # ori_pos = batched_data["pos"]

        n_graph, n_node = residue_seq.size()[:2]

        # mask_aa, mask_pos, padding_mask, _ = self._set_mask(
        #     mask_aa, mask_pos, residue_seq
        # )

        # # pos, time = self._set_noise(ori_pos, mask_pos)
        pos, time = None, None
        # 1 is pad token, 2 is eos token

        (
            mask_aa,
            cls_mask,
            padding_mask,
            eos_mask,
            residue_seq,
        ) = self._set_bert_mask_strategy(mask_aa, mask_pos, residue_seq, x_0)

        mask_seq = residue_seq[mask_aa.squeeze(-1)]
        assert (
            torch.sum((mask_seq == 0) | (mask_seq == 2)) == 0
        ), "residue_seq =- 0 | 2 should not be masked"

        x, edge_feature, delta_pos = self.pfm_emb(
            batched_data,
            padding_mask,
            pos=pos,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
            time=time,
        )

        if perturb is not None:
            x[:, :, :] = x[:, :, :] + perturb

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for _, layer in enumerate(self.layers):
            x, attn_bias = layer(
                x,
                edge_feature=edge_feature,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                mask_pos=mask_pos,
                position_ids=None,
            )
            if not last_state_only:
                inner_states.append(x)

        if attn_bias is not None:
            attn_bias = (
                attn_bias.contiguous()
                .view(n_graph, self.num_attention_heads, n_node, n_node)
                .contiguous()
            )

        return (
            x,
            attn_bias,
            delta_pos,
            pos,
            inner_states,
            padding_mask,
            mask_pos,
            mask_aa,
        )

    def finetune(
        self,
        batched_data,
        perturb=None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention

        with torch.no_grad():
            if "x_new" in batched_data.keys():
                residue_seq = batched_data["x_new"].clone()
            else:
                residue_seq = batched_data["x"].clone()
            x_0 = batched_data["x"].clone()

        # ori_pos = batched_data["pos"]

        n_graph, n_node = residue_seq.size()[:2]

        # mask_aa, mask_pos, padding_mask, _ = self._set_mask(
        #     mask_aa, mask_pos, residue_seq
        # )

        # # pos, time = self._set_noise(ori_pos, mask_pos)
        pos, time = None, None
        # 1 is pad token, 2 is eos token

        # cls_mask = (x_0[:, :]).eq(0)
        padding_mask = (x_0[:, :]).eq(1)  # B x T x 1
        (x_0[:, :]).eq(2)

        x, edge_feature, delta_pos = self.pfm_emb(
            batched_data,
            padding_mask,
            pos=pos,
            mask_aa=None,
            mask_pos=None,
            time=time,
        )

        if perturb is not None:
            x[:, :, :] = x[:, :, :] + perturb

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for _, layer in enumerate(self.layers):
            x, attn_bias = layer(
                x,
                edge_feature=edge_feature,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                mask_pos=None,
                position_ids=None,
            )
            if not last_state_only:
                inner_states.append(x)

        if attn_bias is not None:
            attn_bias = (
                attn_bias.contiguous()
                .view(n_graph, self.num_attention_heads, n_node, n_node)
                .contiguous()
            )

        return (
            x,
            attn_bias,
            delta_pos,
            pos,
            inner_states,
            padding_mask,
            None,
            None,
        )
