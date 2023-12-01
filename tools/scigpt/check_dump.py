# -*- coding: utf-8 -*-
#%%
import torch
# %%
dump = torch.load('/hai1/shufxi/scigpt/7b/stageA/dump.pt', map_location='cpu')
# %%
input_tuple, attention_mask, position_ids, hidden_states = dump
# %%
hidden_states_in, attention_mask_bool, position_ids_in = input_tuple
# %%
hidden_states.shape
# %%
for i in range(hidden_states.shape[1]):
    if not torch.isinf(hidden_states[0,i]).any():
        continue
    for j in range(hidden_states.shape[2]):
        if torch.isinf(hidden_states[0,i,j]).any():
            print(i, j)
            break
# %%
hidden_states[0, 486, 2533]
# %%
attention_mask_bool.shape
# %%
hidden_states_in[0, 618].isinf().any()
# %%
from sfm.models.llama2.llama_modules import LlamaDecoderLayerPP
# %%
from transformers.models.llama import LlamaConfig
# %%
state_dict = torch.load('/hai1/ds_dataset/llama2/llama-2-7b/model.layers.30.pt', map_location='cpu')
state_dict['dummy.weight'] = torch.zeros(1, 1)
state_dict['dummy.bias'] = torch.zeros(1)
del state_dict['self_attn.rotary_emb.inv_freq']
# %%
config = LlamaConfig()
layer = LlamaDecoderLayerPP(config, 30, True)
layer.load_state_dict(state_dict)
layer.half()
layer.cuda()
# %%
input_tuple = tuple(t.cuda() for t in input_tuple)

# %%
layer(input_tuple)[0].isinf().any()

# %%
layer(input_tuple)[0][0, 1000]

# %%
www = 0.1

layer_w = LlamaDecoderLayerPP(config, 30, True, www)
layer_w.load_state_dict(state_dict)
layer_w.half()
layer_w.cuda()

# %%
input_tuple_w = [t.clone() for t in input_tuple]
input_tuple_w[0] = input_tuple_w[0] * www

# %%
layer_w(input_tuple)[0].isinf().any()

# %%
h1 = layer.input_layernorm(input_tuple[0])
# %%
h1_w = layer_w.input_layernorm(input_tuple_w[0])

# %%
h2, _, _ = layer.self_attn(
    hidden_states=h1,
    attention_mask=input_tuple[1],
    position_ids=input_tuple[2],
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
)

# %%
h2_w, _, _ = layer_w.self_attn(
    hidden_states=h1_w,
    attention_mask=input_tuple[1],
    position_ids=input_tuple[2],
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
)


# %%
(h1-h1_w).abs().max()
# %%
(h2-h2_w).abs().argmax()
# %%
h2

(layer.self_attn.q_proj.weight - layer_w.self_attn.q_proj.weight).abs().max()
# %%
(layer.self_attn.k_proj.weight - layer_w.self_attn.k_proj.weight).abs().max()
# %%
(layer.self_attn.v_proj.weight - layer_w.self_attn.v_proj.weight).abs().max()
# %%
(layer.self_attn.o_proj.weight - layer_w.self_attn.o_proj.weight).abs().max()
# %%
(layer.self_attn.rotary_emb.inv_freq - layer_w.self_attn.rotary_emb.inv_freq).abs().max()
# %%
layer.self_attn.config == layer_w.self_attn.config
# %%
input_tuple[0][0, 300]
# %%
