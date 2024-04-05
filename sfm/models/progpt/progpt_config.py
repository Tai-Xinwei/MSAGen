# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass


@dataclass
class ProGPTConfig:
    model_type = "progpt"

    dataset_splits: str = ""
    dataset_ratios: str = ""

    pool_mode: str = "full"
    embedding_length: int = 20
    model_max_length: int = 512

    smiles_dict_path: str = ""
    loadmfmcheck_path: str = ""
    llm_model_name_or_path: str = ""
    mol_size_path: str = ""

    mfm_lora: bool = False
    llm_lora: bool = False
    btn_adaptor: bool = False

    fused_graphormer_llama: bool = False
    add_mol_attn_bias_in_llama: bool = False
    mol_attn_bias_in_llama_layerwise: bool = False
    path_edge_cutoff: int = 0

    skip_num_datasets: str = ""
    num_data_loading_workers: int = 16
    use_global_padding: bool = False
    max_num_mol_per_sample: int = 8
    molecule_max_size: int = 128
