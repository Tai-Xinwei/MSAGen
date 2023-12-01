# -*- coding: utf-8 -*-
import os
import pickle as pkl
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import CrossEntropyLoss, Embedding, Linear, ModuleList
from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    _expand_mask,
    _make_causal_mask,
)

from sfm.models.llama2.llama_modules import (
    LlamaDecoderLayerPP,
    LlamaEmbeddingsPP,
    LlamaHead,
    LlamaModelPP,
    LlamaNorm,
)
from sfm.utils import PretrainedLayerSpec

from .modules.graphormer_encoder import GraphormerSentenceEncoder
from .modules.hybrid_emb import HybridEmbeddingsPP


def convert_graph_attention_bias_to_generalist_attention_bias(
    input_ids: LongTensor = None,
    attention_mask: Tensor = None,
    graphormer_attn_bias: FloatTensor = None,
    past_key_values: List[FloatTensor] = None,
    add_mol_attn_bias_in_llama: bool = False,
    mol_attn_bias_in_llama_layerwise: bool = False,
    llama_num_hidden_layers: int = 0,
    llama_num_attention_heads: int = 0,
    dtype=None,
):
    input_ids_shape = input_ids.size()
    if past_key_values is not None:
        # using cache in decoding
        assert input_ids.size()[-1] == 1
        past_len = past_key_values[0][0].shape[2]
        tgt_len = input_ids.size()[-1]
        src_len = past_len + tgt_len

        # assuming no molecule in decoding (for encoding molecules in generalist only)
        return torch.zeros(
            [input_ids.size()[0], 1, tgt_len, src_len],
            dtype=graphormer_attn_bias.dtype,
            device=graphormer_attn_bias.device,
        )
    else:
        # in training or first token for decoding
        assert attention_mask is not None
        zero_pad = torch.zeros(
            [input_ids_shape[0], 1], dtype=input_ids.dtype, device=input_ids.device
        )
        input_ids_left_pad = torch.cat([zero_pad, input_ids], dim=-1)
        input_ids_right_pad = torch.cat([input_ids, zero_pad], dim=-1)
        start_index_mask = (input_ids_left_pad[:, :-1] != input_ids) & (input_ids < 0)
        end_index_mask = (input_ids_right_pad[:, 1:] != input_ids) & (input_ids < 0)

        start_indexs = torch.nonzero(start_index_mask)
        end_indexs = torch.nonzero(end_index_mask)

        assert torch.all(start_indexs[:, 0] == end_indexs[:, 0])

        if add_mol_attn_bias_in_llama:
            if mol_attn_bias_in_llama_layerwise:
                expanded_graph_attn_bias = torch.full(
                    [
                        llama_num_hidden_layers,
                        input_ids_shape[0],
                        llama_num_attention_heads,
                        input_ids_shape[1],
                        input_ids_shape[1],
                    ],
                    torch.finfo(graphormer_attn_bias.dtype).min,
                    dtype=graphormer_attn_bias.dtype,
                    device=graphormer_attn_bias.device,
                )
            else:
                expanded_graph_attn_bias = torch.full(
                    [
                        input_ids_shape[0],
                        llama_num_attention_heads,
                        input_ids_shape[1],
                        input_ids_shape[1],
                    ],
                    torch.finfo(graphormer_attn_bias.dtype).min,
                    dtype=graphormer_attn_bias.dtype,
                    device=graphormer_attn_bias.device,
                )
        else:
            expanded_graph_attn_bias = torch.full(
                [input_ids_shape[0], 1, input_ids_shape[1], input_ids_shape[1]],
                torch.finfo(dtype).min,
                dtype=dtype,
                device=input_ids.device,
            )

        num_molecule_by_batch = -torch.min(
            input_ids.masked_fill(~(attention_mask.bool()), 0), dim=-1
        ).values
        num_molecule_offset = torch.cat(
            [
                torch.tensor(
                    [0], dtype=torch.long, device=num_molecule_by_batch.device
                ),
                torch.cumsum(num_molecule_by_batch, dim=-1),
            ],
            dim=-1,
        )

        for batch_index, start_index, end_index in zip(
            start_indexs[:, 0], start_indexs[:, 1], end_indexs[:, 1]
        ):
            molecule_index = (
                -input_ids[batch_index, start_index] - 1
            ) + num_molecule_offset[batch_index]
            num_atoms = end_index - start_index + 1
            if add_mol_attn_bias_in_llama:
                if mol_attn_bias_in_llama_layerwise:
                    expanded_graph_attn_bias[
                        :,
                        batch_index,
                        :,
                        start_index : end_index + 1,
                        start_index : end_index + 1,
                    ] = graphormer_attn_bias[
                        :, molecule_index, :, :num_atoms, :num_atoms
                    ]
                else:
                    expanded_graph_attn_bias[
                        batch_index,
                        :,
                        start_index : end_index + 1,
                        start_index : end_index + 1,
                    ] = graphormer_attn_bias[molecule_index, :, :num_atoms, :num_atoms]
            else:
                expanded_graph_attn_bias[
                    batch_index,
                    0,
                    start_index : end_index + 1,
                    start_index : end_index + 1,
                ] = 0.0

        # encode llm padding

        if add_mol_attn_bias_in_llama and mol_attn_bias_in_llama_layerwise:
            return expanded_graph_attn_bias.masked_fill(
                ~(attention_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).bool()),
                torch.finfo(expanded_graph_attn_bias.dtype).max,
            )
        else:
            return expanded_graph_attn_bias.masked_fill(
                ~(attention_mask.unsqueeze(1).unsqueeze(1).bool()),
                torch.finfo(expanded_graph_attn_bias.dtype).max,
            )


def _prepare_decoder_attention_mask_from_generalist_attention_bias(
    generalist_attention_bias,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    add_mol_attn_bias_in_llama,
    mol_attn_bias_in_llama_layerwise,
):
    combined_attention_mask = None
    recovered_llm_padding_mask = torch.all(
        generalist_attention_bias != torch.finfo(generalist_attention_bias.dtype).max,
        dim=-3,
    )
    assert torch.all(
        recovered_llm_padding_mask
        == torch.any(
            generalist_attention_bias
            != torch.finfo(generalist_attention_bias.dtype).max,
            dim=-3,
        )
    )

    assert torch.all(
        torch.all(recovered_llm_padding_mask, dim=-2)
        == torch.any(recovered_llm_padding_mask, dim=-2)
    )
    recovered_llm_padding_mask = torch.all(recovered_llm_padding_mask, dim=-2)

    if add_mol_attn_bias_in_llama and mol_attn_bias_in_llama_layerwise:
        assert torch.all(
            torch.all(recovered_llm_padding_mask, dim=0)
            == torch.any(recovered_llm_padding_mask, dim=0)
        )
        recovered_llm_padding_mask = torch.all(
            recovered_llm_padding_mask,
            dim=0,
        )

    if input_shape[-1] > 1:
        intra_graph_mask = (
            generalist_attention_bias
            != torch.finfo(generalist_attention_bias.dtype).min
        ) & (
            generalist_attention_bias
            != torch.finfo(generalist_attention_bias.dtype).max
        )
        assert torch.all(
            torch.all(intra_graph_mask, dim=-3) == torch.any(intra_graph_mask, dim=-3)
        )
        intra_graph_mask = torch.all(intra_graph_mask, dim=-3)
        if add_mol_attn_bias_in_llama and mol_attn_bias_in_llama_layerwise:
            assert torch.all(
                torch.all(intra_graph_mask, dim=0) == torch.any(intra_graph_mask, dim=0)
            )
            intra_graph_mask = torch.all(intra_graph_mask, dim=0)

        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

        combined_attention_mask = combined_attention_mask.masked_fill(
            intra_graph_mask.unsqueeze(1), 0.0
        )

    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_attn_mask = _expand_mask(
        recovered_llm_padding_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
    ).to(inputs_embeds.device)
    if combined_attention_mask is None:
        return expanded_attn_mask
    else:
        text_mask = ~torch.any(intra_graph_mask, dim=-1)
        if add_mol_attn_bias_in_llama and mol_attn_bias_in_llama_layerwise:
            return (
                expanded_attn_mask.unsqueeze(0)
                + combined_attention_mask.unsqueeze(0)
                + generalist_attention_bias.masked_fill(
                    text_mask.unsqueeze(-1).unsqueeze(0).unsqueeze(2), 0.0
                )
            )
        else:
            mask = (
                expanded_attn_mask
                + combined_attention_mask
                + generalist_attention_bias.masked_fill(
                    text_mask.unsqueeze(-1).unsqueeze(1), 0.0
                )
            )
            if not add_mol_attn_bias_in_llama:
                mask = mask.eq(0.0)
            return mask


class LlamaDecoderLayerFused(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_index: int,
        mol_attn_bias_in_llama_layerwise: bool = False,
    ):
        super().__init__(config)
        self.layer_index = layer_index
        self.mol_attn_bias_in_llama_layerwise = mol_attn_bias_in_llama_layerwise

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        position_ids: LongTensor = None,
        past_key_value: Tuple[Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[FloatTensor, Tuple[FloatTensor, FloatTensor]]:
        if self.mol_attn_bias_in_llama_layerwise:
            return super().forward(
                hidden_states,
                attention_mask[self.layer_index],
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            return super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )


class LlamaModelFusedGraphormer(LlamaModel):
    def __init__(
        self,
        config: LlamaConfig,
        use_lora: bool,
        add_mol_attn_bias_in_llama: bool = False,
        mol_attn_bias_in_llama_layerwise: bool = False,
    ):
        super().__init__(config)

        if add_mol_attn_bias_in_llama:
            self.layers = ModuleList(
                [
                    LlamaDecoderLayerFused(config, layer_index)
                    for layer_index in range(config.num_hidden_layers)
                ]
            )

        if use_lora:
            LORA_R = 8
            LORA_ALPHA = 16
            LORA_DROPOUT = 0.1
            TARGET_MODULES = [
                "q_proj",
                "k_proj",
                "v_proj",
            ]

            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
            )

            self.layers = ModuleList(
                [get_peft_model(layer, lora_config) for layer in self.layers]
            )

        self.add_mol_attn_bias_in_llama = add_mol_attn_bias_in_llama
        self.mol_attn_bias_in_llama_layerwise = mol_attn_bias_in_llama_layerwise

    def forward(
        self,
        input_ids: LongTensor = None,
        attention_mask: Tensor = None,
        position_ids: LongTensor = None,
        past_key_values: List[FloatTensor] = None,
        inputs_embeds: FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        graphormer_attn_bias: FloatTensor = None,
    ):
        if graphormer_attn_bias is None or past_key_values is not None:
            if past_key_values is not None:
                inputs_embeds = None
            return super().forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        # encode llm padding
        expanded_graph_attn_bias = (
            convert_graph_attention_bias_to_generalist_attention_bias(
                input_ids,
                attention_mask,
                graphormer_attn_bias,
                past_key_values,
                self.add_mol_attn_bias_in_llama,
                self.mol_attn_bias_in_llama_layerwise,
                self.config.num_hidden_layers,
                self.config.num_attention_heads,
            )
        )

        if inputs_embeds is not None:
            input_ids = None
        return super().forward(
            input_ids,
            expanded_graph_attn_bias,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def _prepare_decoder_attention_mask(
        self,
        generalist_attention_bias,
        input_shape,
        inputs_embeds,
        past_key_values_length,
    ):
        if generalist_attention_bias.dtype == inputs_embeds.dtype:
            return _prepare_decoder_attention_mask_from_generalist_attention_bias(
                generalist_attention_bias,
                input_shape,
                inputs_embeds,
                past_key_values_length,
                self.add_mol_attn_bias_in_llama,
                self.mol_attn_bias_in_llama_layerwise,
            )
        else:
            super()._prepare_decoder_attention_mask(
                generalist_attention_bias,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )


class LlamaForCausalLMFusedGraphormer(LlamaForCausalLM):
    def __init__(
        self,
        config,
        use_lora,
        add_mol_attn_bias_in_llama: bool = False,
        mol_attn_bias_in_llama_layerwise: bool = False,
    ):
        super().__init__(config)
        self.model = LlamaModelFusedGraphormer(
            config,
            use_lora,
            add_mol_attn_bias_in_llama,
            mol_attn_bias_in_llama_layerwise,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graphormer_attn_bias: FloatTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graphormer_attn_bias=graphormer_attn_bias,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.size()

        seq_length_with_past = seq_length

        device = input_ids.device
        position_ids = torch.arange(
            0,
            seq_length_with_past,
            dtype=torch.long,
            device=device,
        )
        position_ids = (
            position_ids.unsqueeze(0)
            .view(-1, seq_length_with_past)
            .repeat([batch_size, 1])
        )

        input_ids_left_pad = torch.cat(
            [
                torch.zeros(
                    [input_ids.size()[0], 1],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                input_ids[:, 1:],
            ],
            dim=-1,
        )
        is_first_atom_in_mol = (input_ids < 0) & (input_ids != input_ids_left_pad)
        mol_atom_pos_offset = torch.cumsum(
            (input_ids < 0 & (~is_first_atom_in_mol)).long(), dim=-1
        )

        position_ids -= mol_atom_pos_offset
        if past_key_values is not None:
            position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            position_ids=position_ids,
        )
        model_inputs["graphormer_attn_bias"] = kwargs.get("graphormer_attn_bias", None)
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
        return model_inputs


class GraphormerEncoderPPFused(GraphormerSentenceEncoder):
    def __init__(
        self,
        llama_config,
        graphormer_config,
        init_bias: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        q_noise: float = 0,
        qn_block_size: int = 8,
        add_mol_attn_bias_in_llama: bool = False,
        mol_attn_bias_in_llama_layerwise: bool = False,
        path_edge_cutoff: int = 0,
        pp_mode: bool = True,
    ) -> None:
        super().__init__(
            graphormer_config,
            init_bias,
            embed_scale,
            freeze_embeddings,
            n_trans_layers_to_freeze,
            export,
            q_noise,
            qn_block_size,
        )

        self.add_mol_attn_bias_in_llama = add_mol_attn_bias_in_llama
        self.mol_attn_bias_in_llama_layerwise = mol_attn_bias_in_llama_layerwise
        self.num_edge_feature_values = 14
        self.edge_embed_dim = 32
        self.path_embed_dim = 32
        self.pp_mode = pp_mode

        if self.add_mol_attn_bias_in_llama:
            if self.mol_attn_bias_in_llama_layerwise:
                num_total_heads = (
                    llama_config.num_attention_heads * llama_config.num_hidden_layers
                )
            else:
                num_total_heads = llama_config.num_attention_heads
            self.num_total_heads = num_total_heads
            # +1 in num_spatial for disconnected nodes
            self.llama_node_distance_embed = Embedding(
                graphormer_config.num_spatial + 1, num_total_heads
            )
            self.llama_edge_embed = Embedding(
                self.num_edge_feature_values, self.edge_embed_dim
            )

            if path_edge_cutoff > 0:
                num_edge_feature_weights = path_edge_cutoff
            else:
                num_edge_feature_weights = graphormer_config.num_spatial + 1
            self.llama_edge_feature_weights = Embedding(
                (self.num_edge_feature_values * num_edge_feature_weights),
                self.edge_embed_dim * self.path_embed_dim,
            )
            self.path_attn_bias_proj = Linear(self.path_embed_dim, self.num_total_heads)

            self.llama_edge_embed.weight.data[0] = 0.0
            self.llama_edge_feature_weights.weight.data[0] = 0.0

        self.llama_num_attention_heads = llama_config.num_attention_heads
        self.path_edge_cutoff = path_edge_cutoff
        self.max_distance_in_path = graphormer_config.num_spatial

    def _compress_edge_input(self, edge_input: LongTensor):
        compressed_edge_input = edge_input.clone()
        assert torch.all(edge_input[edge_input > 0] >= 3)
        compressed_edge_input -= 3
        compressed_edge_input[compressed_edge_input >= 512] -= 512
        compressed_edge_input[compressed_edge_input >= 512] -= 512
        compressed_edge_input[:, :, :, :, 1][
            compressed_edge_input[:, :, :, :, 1] >= 0
        ] += 5
        compressed_edge_input[:, :, :, :, 2][
            compressed_edge_input[:, :, :, :, 2] >= 0
        ] += 11
        compressed_edge_input += 1
        assert torch.all(compressed_edge_input[compressed_edge_input < 0] == -2)
        compressed_edge_input[compressed_edge_input < 0] = 0
        return compressed_edge_input

    def forward(self, batched_data):
        if self.pp_mode:
            (
                input_ids,
                llm_mask,
                _,
                x,
                in_degree,
                out_degree,
                attn_bias,
                spatial_pos,
                edge_input,
                _,
                pos,
                mask3d,
                node_type_edge,
            ) = batched_data

            # create dict for batched data
            batched_data = {}
            batched_data["attn_bias"] = attn_bias
            batched_data["spatial_pos"] = spatial_pos
            batched_data["in_degree"] = in_degree
            batched_data["out_degree"] = out_degree
            batched_data["x"] = x
            batched_data["edge_input"] = edge_input
            batched_data["attn_edge_type"] = None
            batched_data["pos"] = pos
            batched_data["mask3d"] = mask3d
            batched_data["node_type_edge"] = node_type_edge

        else:
            attn_bias = batched_data["attn_bias"]
            spatial_pos = batched_data["spatial_pos"]
            edge_input = batched_data["edge_input"]

        x, _, _, _, _, padding_mask, _ = super().forward(batched_data)

        max_num_nodes, num_molecules = x.size()[:2]

        edge_input = self._compress_edge_input(edge_input)

        assert torch.all(edge_input < self.num_edge_feature_values)

        # remove graphormer virtual token
        max_num_nodes -= 1
        padding_mask = padding_mask[:, 1:]

        if self.add_mol_attn_bias_in_llama:
            llama_graph_attn_bias = torch.zeros(
                [num_molecules, self.num_total_heads, max_num_nodes, max_num_nodes],
                dtype=attn_bias.dtype,
                device=attn_bias.device,
            )
            llama_graph_attn_bias = llama_graph_attn_bias.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), torch.finfo(attn_bias.dtype).min
            )
            llama_graph_attn_bias = llama_graph_attn_bias.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(-1),
                torch.finfo(attn_bias.dtype).min,
            )

            llama_graph_attn_bias += self.llama_node_distance_embed(
                spatial_pos
            ).permute(0, 3, 1, 2)
            if self.path_edge_cutoff > 0:
                edge_input = edge_input[:, :, :, : self.path_edge_cutoff]
            path_edge_feature_embeddings = self.llama_edge_embed(
                edge_input
            )  # N x T x T x D x EF x ED1
            edge_input_with_dist_offset = (
                edge_input
                + torch.arange(
                    edge_input.size()[-2],
                    dtype=edge_input.dtype,
                    device=edge_input.device,
                ).unsqueeze(-1)
                * self.num_edge_feature_values
            )
            path_edge_feature_weights = self.llama_edge_feature_weights(
                edge_input_with_dist_offset
            ).view(
                num_molecules,
                max_num_nodes,
                max_num_nodes,
                edge_input.size()[-2],
                edge_input.size()[-1],
                self.edge_embed_dim,
                self.path_embed_dim,
            )  # N x T x T x D x EF x ED1 x ED2

            if self.path_edge_cutoff > 0:
                path_edge_feature_weights = path_edge_feature_weights[
                    :, :, :, : self.path_edge_cutoff, :, :, :
                ]
            path_attn_bias = torch.bmm(
                path_edge_feature_embeddings.view(-1, 1, self.edge_embed_dim),
                path_edge_feature_weights.view(
                    -1, self.edge_embed_dim, self.path_embed_dim
                ),
            ).view(
                num_molecules,
                max_num_nodes,
                max_num_nodes,
                edge_input.size()[-2],
                edge_input.size()[-1],
                self.path_embed_dim,
            )  # N x T x T x D x EF x ED2
            path_attn_bias = path_attn_bias.mean(
                dim=-2
            )  # average over edge features, N x T x T x D x ED2

            pairwise_node_distance = spatial_pos.clone()
            pairwise_node_distance[pairwise_node_distance == 0] = 1
            pairwise_node_distance[pairwise_node_distance > 1] -= 1
            path_attn_bias = path_attn_bias.sum(
                dim=-2
            ) / pairwise_node_distance.unsqueeze(
                -1
            )  # average over path, N x T x T x ED2
            path_attn_bias = self.path_attn_bias_proj(path_attn_bias)
            path_attn_bias = path_attn_bias.permute(0, 3, 1, 2)
            llama_graph_attn_bias += path_attn_bias

            if self.mol_attn_bias_in_llama_layerwise:
                llama_graph_attn_bias = llama_graph_attn_bias.view(
                    num_molecules,
                    -1,
                    self.llama_num_attention_heads,
                    max_num_nodes,
                    max_num_nodes,
                ).transpose(0, 1)
            if self.pp_mode:
                return (
                    x,
                    llama_graph_attn_bias,
                    padding_mask,
                    llm_mask,
                    input_ids,
                )
            else:
                return (x, llama_graph_attn_bias, None, None, None, padding_mask, None)
        else:
            if self.pp_mode:
                return (
                    x,
                    padding_mask,
                    llm_mask,
                    input_ids,
                )
            else:
                return (x, llama_graph_attn_bias, None, None, None, padding_mask, None)


class LlamaEmbeddingsPPFused(LlamaEmbeddingsPP):
    def __init__(
        self,
        config: LlamaConfig,
        learnable_cutoff: int = 32001,
        add_mol_attn_bias_in_llama: bool = False,
    ):
        super().__init__(config, learnable_cutoff=learnable_cutoff)
        self.add_mol_attn_bias_in_llama = add_mol_attn_bias_in_llama

    def forward(
        self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.add_mol_attn_bias_in_llama:
            (
                mol_emb,
                graph_attn_bias,
                mol_padding_mask,
                llm_mask,
                input_ids,
            ) = input_tuple
        else:
            mol_emb, mol_padding_mask, llm_mask, input_ids = input_tuple

        mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids = super().forward(
            [mol_emb, mol_padding_mask, llm_mask, input_ids]
        )

        if self.add_mol_attn_bias_in_llama:
            return (
                mol_emb,
                graph_attn_bias,
                mol_padding_mask,
                text_embeds,
                llm_mask,
                input_ids,
            )
        else:
            return (
                mol_emb,
                mol_padding_mask,
                text_embeds,
                llm_mask,
                input_ids,
            )


class HybridEmbeddingsPPFused(HybridEmbeddingsPP):
    def __init__(
        self,
        config: PretrainedConfig,
        add_mol_attn_bias_in_llama: bool = False,
        mol_attn_bias_in_llama_layerwise: bool = False,
        num_llama_hidden_layers: int = 0,
        num_llama_attention_heads: int = 0,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.add_mol_attn_bias_in_llama = add_mol_attn_bias_in_llama
        self.mol_attn_bias_in_llama_layerwise = mol_attn_bias_in_llama_layerwise
        self.num_llama_hidden_layers = num_llama_hidden_layers
        self.num_llama_attention_heads = num_llama_attention_heads

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        input_ids,
    ):
        return _prepare_decoder_attention_mask_from_generalist_attention_bias(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            self.add_mol_attn_bias_in_llama,
            self.mol_attn_bias_in_llama_layerwise,
        )

    def forward(self, input_tuple: Tuple):
        if self.add_mol_attn_bias_in_llama:
            (
                mol_emb,
                graphormer_attn_bias,
                mol_padding_mask,
                text_embeds,
                llm_mask,
                input_ids,
            ) = input_tuple
            graphormer_attn_bias = graphormer_attn_bias.to(dtype=text_embeds.dtype)
            expanded_graph_attn_bias = convert_graph_attention_bias_to_generalist_attention_bias(
                input_ids,
                llm_mask,
                graphormer_attn_bias,
                add_mol_attn_bias_in_llama=self.add_mol_attn_bias_in_llama,
                mol_attn_bias_in_llama_layerwise=self.mol_attn_bias_in_llama_layerwise,
                llama_num_hidden_layers=self.num_llama_hidden_layers,
                llama_num_attention_heads=self.num_llama_attention_heads,
            )
        else:
            (
                mol_emb,
                mol_padding_mask,
                text_embeds,
                llm_mask,
                input_ids,
            ) = input_tuple
            expanded_graph_attn_bias = convert_graph_attention_bias_to_generalist_attention_bias(
                input_ids,
                llm_mask,
                None,
                add_mol_attn_bias_in_llama=self.add_mol_attn_bias_in_llama,
                mol_attn_bias_in_llama_layerwise=self.mol_attn_bias_in_llama_layerwise,
                llama_num_hidden_layers=self.num_llama_hidden_layers,
                llama_num_attention_heads=self.num_llama_attention_heads,
                dtype=mol_emb.dtype,
            )

        hidden_states, llm_mask, position_ids = super().forward(
            [
                mol_emb,
                mol_padding_mask,
                text_embeds,
                expanded_graph_attn_bias,
                input_ids,
            ]
        )

        input_ids_left_pad = torch.cat(
            [
                torch.zeros(
                    [input_ids.size()[0], 1],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                input_ids[:, 1:],
            ],
            dim=-1,
        )
        is_first_atom_in_mol = (input_ids < 0) & (input_ids != input_ids_left_pad)
        mol_atom_pos_offset = torch.cumsum(
            (input_ids < 0 & (~is_first_atom_in_mol)).long(), dim=-1
        )

        position_ids -= mol_atom_pos_offset

        return (hidden_states, llm_mask, position_ids)


class LlamaDecoderLayerPPFused(LlamaDecoderLayerPP):
    def __init__(
        self,
        config: LlamaConfig,
        layer_index: int,
        enable_mem_efficient: bool = True,
        add_mol_attn_bias_in_llama: bool = False,
        mol_attn_bias_in_llama_layerwise: bool = False,
    ):
        super().__init__(config, layer_index, enable_mem_efficient)
        self.add_mol_attn_bias_in_llama = add_mol_attn_bias_in_llama
        self.mol_attn_bias_in_llama_layerwise = mol_attn_bias_in_llama_layerwise

    def forward(self, input_tuple) -> Tuple[FloatTensor, Tensor, Tensor]:
        hidden_states, attention_mask_with_bias, position_ids = input_tuple

        if self.add_mol_attn_bias_in_llama:
            if self.mol_attn_bias_in_llama_layerwise:
                attention_mask = attention_mask_with_bias[self.layer_index]
            else:
                attention_mask = attention_mask_with_bias
        else:
            attention_mask = torch.zeros(
                attention_mask_with_bias.size(),
                device=attention_mask_with_bias.device,
                dtype=hidden_states.dtype,
            )
            attention_mask = attention_mask.masked_fill(
                ~attention_mask_with_bias, torch.finfo(hidden_states.dtype).min
            )

        hidden_states = super(LlamaDecoderLayerPP, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

        outputs = (hidden_states, attention_mask_with_bias, position_ids)

        return outputs


class LlamaModelPPFused(LlamaModelPP):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(args, config)

    @classmethod
    def to_layers(
        cls, args, config, learnable_cutoff=0, new_num_tokens=None, load_ckpt=False
    ):
        cls.pipe_layer = []
        for i in range(config.num_hidden_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerPPFused,
                    config,
                    i,
                    load_ckpt=load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        args.llm_model_name_or_path, "model.layers.{}.pt".format(i)
                    ),
                    lora_mode="freeze" if not args.llm_lora else "lora",
                    add_mol_attn_bias_in_llama=args.add_mol_attn_bias_in_llama,
                    mol_attn_bias_in_llama_layerwise=args.mol_attn_bias_in_llama_layerwise,
                )
            )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaNorm,
                config,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.llm_model_name_or_path, "model.norm.pt"
                ),
                lora_mode="freeze",
            )
        )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaHead,
                config,
                new_num_tokens=new_num_tokens,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.llm_model_name_or_path, "model.lm_head.pt"
                ),
                lora_mode="freeze",
            )
        )

        return cls.pipe_layer
