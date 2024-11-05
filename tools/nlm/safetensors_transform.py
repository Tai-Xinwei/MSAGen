# -*- coding: utf-8 -*-
import torch
from safetensors import safe_open

# Function 1: read safetensors format of Phi model checkpoints from huggingface.
# safetensors_path = "/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/model-00001-of-00002.safetensors"
# tensors = {}
# with safe_open(safetensors_path, framework="pt", device="cpu") as f:
#    for key in f.keys():
#        tensors[key] = f.get_tensor(key)
#        print("{0}: {1}".format(key, tensors[key].shape))



# Function 2: read pt format of Phi model checkpoints from SFM training.
# pt_path = "/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/phi35mini_instruct.pt"
# pt_path = "/home/guoqingliu/blob_nlm/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_test/global_step100/mp_rank_00_model_states.pt"

# checkpoints_state = torch.load(pt_path, map_location="cpu")
# print("len:", len(checkpoints_state["module"]))

# for key, value in checkpoints_state["module"].items():
#     print(key, value.shape)



# Function 3: transfer from safetensors to pt for SFM training.
# import torch
# from transformers import AutoModelForCausalLM

# torch.random.manual_seed(0)

# model = AutoModelForCausalLM.from_pretrained(
#     # "/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct",
#     "microsoft/Phi-3.5-mini-instruct",
#     # device_map="cuda",
#     device_map="cpu",
#     torch_dtype="auto",
#     trust_remote_code=True,
# )
# print("model.dtype: ", model.dtype)
# print("len(model.state_dict()): ", len(model.state_dict()))
# for key, value in model.state_dict().items():
#     print(key, value.shape)

# checkpoints_state = model.state_dict().copy()

# for key in list(checkpoints_state.keys()):
#     if "model." in key:
#         checkpoints_state[key.replace("model.", "")] = checkpoints_state.pop(key)

# for key in list(checkpoints_state.keys()):
#     if "qkv_proj" in key:
#         # split into q, k, v
#         qkv_proj = checkpoints_state.pop(key)
#         hidden_size = qkv_proj.shape[0] // 3
#         # print("hidden_size: ", hidden_size)
#         q_proj = qkv_proj[: hidden_size, :]
#         k_proj = qkv_proj[hidden_size : 2 * hidden_size, :]
#         v_proj = qkv_proj[2 * hidden_size :, :]
#         checkpoints_state[key.replace("qkv_proj", "q_proj")] = q_proj
#         checkpoints_state[key.replace("qkv_proj", "k_proj")] = k_proj
#         checkpoints_state[key.replace("qkv_proj", "v_proj")] = v_proj
#     elif "gate_up_proj" in key:
#         # split into gate, up
#         gate_up_proj = checkpoints_state.pop(key)
#         hidden_size = gate_up_proj.shape[0] // 2
#         # print("hidden_size: ", hidden_size)
#         gate_proj = gate_up_proj[: hidden_size, :]
#         up_proj = gate_up_proj[hidden_size:, :]
#         checkpoints_state[key.replace("gate_up_proj", "gate_proj")] = gate_proj
#         checkpoints_state[key.replace("gate_up_proj", "up_proj")] = up_proj
#         # print("gate_proj.shape: ", gate_proj.shape)
#         # print("up_proj.shape: ", up_proj.shape)
#     elif "embed_tokens.weight" in key:
#         checkpoints_state[key.replace("embed_tokens.weight", "word_embeddings.weight")] = checkpoints_state.pop(key)
#     # elif "mlp.down_proj" in key:
#     #     # transpose
#     #     assert checkpoints_state[key].dim() == 2
#     #     checkpoints_state[key] = checkpoints_state[key].transpose(0, 1)
#     #     print("mlp.down_proj.shape: ", checkpoints_state[key].shape)

# # print("len(checkpoints_state): ", len(checkpoints_state))
# # for key, value in checkpoints_state.items():
# #     print(key, value.shape)


# print((checkpoints_state["layers.0.mlp.gate_proj.weight"].mean() + checkpoints_state["layers.0.mlp.up_proj.weight"].mean()) / 2.0)
# print(checkpoints_state["layers.0.mlp.up_proj.weight"].mean())
# print(checkpoints_state["layers.0.mlp.gate_proj.weight"].mean())

# # # check two real vectors are close enough
# # def is_close(a, b, tol=1e-6):
# #     return torch.all(torch.lt(torch.abs(torch.add(a, -b)), tol))

# # print(is_close(model.state_dict()["lm_head.weight"], checkpoints_state["lm_head.weight"]))
# # print(is_close(model.state_dict()["model.norm.weight"], checkpoints_state["norm.weight"]))
# # print(is_close(model.state_dict()["model.embed_tokens.weight"], checkpoints_state["word_embeddings.weight"]))

# # print(is_close(model.state_dict()["model.layers.0.self_attn.o_proj.weight"], checkpoints_state["layers.0.self_attn.o_proj.weight"]))
# # print(is_close(model.state_dict()["model.layers.0.mlp.down_proj.weight"], checkpoints_state["layers.0.mlp.down_proj.weight"]))
# # print(is_close(model.state_dict()["model.layers.0.mlp.gate_up_proj.weight"][:8192, :], checkpoints_state["layers.0.mlp.gate_proj.weight"]))
# # print(is_close(model.state_dict()["model.layers.0.mlp.gate_up_proj.weight"][8192:, :], checkpoints_state["layers.0.mlp.up_proj.weight"]))
# # print(is_close(model.state_dict()["model.layers.0.self_attn.qkv_proj.weight"][:3072, :], checkpoints_state["layers.0.self_attn.q_proj.weight"]))
# # print(is_close(model.state_dict()["model.layers.0.self_attn.qkv_proj.weight"][3072:6144, :], checkpoints_state["layers.0.self_attn.k_proj.weight"]))
# # print(is_close(model.state_dict()["model.layers.0.input_layernorm.weight"], checkpoints_state["layers.0.input_layernorm.weight"]))
# # print(is_close(model.state_dict()["model.layers.0.post_attention_layernorm.weight"], checkpoints_state["layers.0.post_attention_layernorm.weight"]))

# # print(checkpoints_state["layers.0.post_attention_layernorm.weight"].dtype)
# # print(checkpoints_state["layers.0.mlp.down_proj.weight"].dtype)
# torch.save(checkpoints_state, "/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/phi35mini_instruct.pt")



# Function 4: Transfer from pt to safetensors(pt) for Huggingface model loading and inference.
import torch
import safetensors

# pt_path = "/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/phi35mini_instruct.pt"
# pt_path = "/home/guoqingliu/blob_nlm/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_local_debug/global_step100/mp_rank_00_model_states.pt"
# pt_path = "/home/guoqingliu/blob_nlm/guoqing/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_debug_grad_64/global_step100/mp_rank_00_model_states.pt"
# pt_path = "/home/guoqingliu/blob_nlm/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_local_debug/global_step400/mp_rank_00_model_states.pt"

# pt_path = "/home/guoqingliu/blob_nlm/guoqing/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_test_v4/global_step1000/mp_rank_00_model_states.pt"
# pt_path = "/home/guoqingliu/blob_nlm/guoqing/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_test_v5/global_step1000/mp_rank_00_model_states.pt"
# pt_path = "/home/guoqingliu/blob_nlm/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_local_debug_1031/global_step20/mp_rank_00_model_states.pt"

# debug training process
# pt_path = "/home/guoqingliu/blob_nlm/guoqing/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_debug_grad_64/global_step900/mp_rank_00_model_states.pt"

# 16 AMD GPUs, AdamW result checkpoints.
pt_path = "/home/guoqingliu/blob_nlm/guoqing/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_16cards_zero1_adamw_20241102/global_step9000/mp_rank_00_model_states.pt"

# Step 1: Load the .pt file
checkpoints_state = torch.load(pt_path, map_location="cpu")

if "module" in checkpoints_state:
    checkpoints_state = checkpoints_state["module"]

for key, value in checkpoints_state.items():
    print(key, value.shape)

for key in list(checkpoints_state.keys()):
    # assert "model." not in key and "net." not in key
    # checkpoints_state["model."+key] = checkpoints_state.pop(key)
    assert "net." in key
    checkpoints_state[key.replace("net.", "model.")] = checkpoints_state.pop(key)

q_proj_list = []
k_proj_list = []
v_proj_list = []

gate_proj_list = []
up_proj_list = []

for i, key in enumerate(list(checkpoints_state.keys())):
    if "q_proj" in key:
        print("key: ", key)
        q_proj = checkpoints_state.pop(key)
        q_proj_list.append([key, q_proj])
    elif "k_proj" in key:
        print("key: ", key)
        k_proj = checkpoints_state.pop(key)
        k_proj_list.append([key, k_proj])
    elif "v_proj" in key:
        print("key: ", key)
        v_proj = checkpoints_state.pop(key)
        v_proj_list.append([key, v_proj])
    elif "gate_proj" in key:
        print("key: ", key)
        gate_proj = checkpoints_state.pop(key)
        gate_proj_list.append([key, gate_proj])
    elif "up_proj" in key:
        print("key: ", key)
        up_proj = checkpoints_state.pop(key)
        up_proj_list.append([key, up_proj])
    elif "word_embeddings.weight" in key:
        checkpoints_state[key.replace("word_embeddings.weight", "embed_tokens.weight")] = checkpoints_state.pop(key)
    elif "model.lm_head.weight" in key:
        checkpoints_state[key.replace("model.lm_head.weight", "lm_head.weight")] = checkpoints_state.pop(key)
    elif "dummy" in key:
        checkpoints_state.pop(key)

for key_q_proj, key_k_proj, key_v_proj in zip(q_proj_list, k_proj_list, v_proj_list):
    assert key_q_proj[0].split("q_proj")[0] == key_k_proj[0].split("k_proj")[0] == key_v_proj[0].split("v_proj")[0]
    assert key_q_proj[0].split("q_proj")[1] == key_k_proj[0].split("k_proj")[1] == key_v_proj[0].split("v_proj")[1]
    qkv_proj = torch.cat([key_q_proj[1], key_k_proj[1], key_v_proj[1]], dim=0)
    print("qkv_proj.shape: ", qkv_proj.shape)
    checkpoints_state[key_q_proj[0].split("q_proj")[0] + "qkv_proj" + key_q_proj[0].split("q_proj")[1]] = qkv_proj

for key_gate_proj, key_up_proj in zip(gate_proj_list, up_proj_list):
    assert key_gate_proj[0].split("gate_proj")[0] == key_up_proj[0].split("up_proj")[0]
    assert key_gate_proj[0].split("gate_proj")[1] == key_up_proj[0].split("up_proj")[1]
    gate_up_proj = torch.cat([key_gate_proj[1], key_up_proj[1]], dim=0)
    print("gate_up_proj.shape: ", gate_up_proj.shape)

    if "layers.0" in key_gate_proj[0]:
        print("key_gate_proj[1]: ", key_gate_proj[1].mean())
        print("key_up_proj[1]: ", key_up_proj[1].mean())
        print("gate_up_proj.mean(): ", gate_up_proj.mean())
    checkpoints_state[key_gate_proj[0].split("gate_proj")[0] + "gate_up_proj" + key_gate_proj[0].split("gate_proj")[1]] = gate_up_proj

# check two real vectors are close enough
def is_close(a, b, tol=1e-6):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), tol))

print("len(checkpoints_state): ", len(checkpoints_state))
for key, value in checkpoints_state.items():
    print(key, value.shape)

# print(is_close(model.state_dict()["lm_head.weight"], checkpoints_state["lm_head.weight"]))
# print(is_close(model.state_dict()["model.norm.weight"], checkpoints_state["model.norm.weight"]))
# print(is_close(model.state_dict()["model.embed_tokens.weight"], checkpoints_state["model.embed_tokens.weight"]))

# print(is_close(model.state_dict()["model.layers.0.self_attn.o_proj.weight"], checkpoints_state["model.layers.0.self_attn.o_proj.weight"]))
# print(is_close(model.state_dict()["model.layers.0.self_attn.qkv_proj.weight"], checkpoints_state["model.layers.0.self_attn.qkv_proj.weight"]))
# print(is_close(model.state_dict()["model.layers.0.mlp.down_proj.weight"], checkpoints_state["model.layers.0.mlp.down_proj.weight"]))
# print(is_close(model.state_dict()["model.layers.0.mlp.gate_up_proj.weight"], checkpoints_state["model.layers.0.mlp.gate_up_proj.weight"]))

# print(model.state_dict()["model.layers.0.mlp.gate_up_proj.weight"].mean())
# print(checkpoints_state["model.layers.0.mlp.gate_up_proj.weight"].mean())

# print(is_close(model.state_dict()["model.layers.0.input_layernorm.weight"], checkpoints_state["model.layers.0.input_layernorm.weight"]))
# print(is_close(model.state_dict()["model.layers.0.post_attention_layernorm.weight"], checkpoints_state["model.layers.0.post_attention_layernorm.weight"]))

# Step 2: Save to a new pt file
# torch.save(checkpoints_state, '/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/round_trip_safetensor/phi35mini_instruct_amd_grad_64_step1000.pt')
# torch.save(checkpoints_state, '/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/round_trip_safetensor/phi35mini_instruct_random_grad_512_step20.pt')

# torch.save(checkpoints_state, '/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/round_trip_safetensor/debug_training/phi35mini_instruct_grad_64_step900.pt')

torch.save(checkpoints_state, '/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/round_trip_safetensor/16cards_zero1_adamw_20241102/phi35mini_instruct_step9000.pt')
