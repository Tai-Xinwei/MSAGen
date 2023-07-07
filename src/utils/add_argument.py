import argparse

import deepspeed


def add_argument():
    parser = argparse.ArgumentParser(description="MFMds")
    parser.add_argument(
        "--dataset-name", type=str, default="PM6-Full-3D", help="dataset name"
    )
    parser.add_argument(
        "--data_path", type=str, default="./dataset", help="path to dataset"
    )
    parser.add_argument(
        "--loadcheck_path", type=str, default="", help="path to dataset"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=1024,
        type=int,
        help="mini-batch size (default: 32)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="number of total epochs (default: 50)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--global_rank",
        type=int,
        default=-1,
        help="global rank passed from distributed launcher",
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", help="distributed backend"
    )
    parser.add_argument("--seed", type=int, default=666666, help="PRNG seed")
    parser.add_argument("--node_rank", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--num-classes", type=int, default=1, help="number of classes")
    parser.add_argument(
        "--encoder_embed_dim", type=int, default=768, help="encoder embedding dimension"
    )
    parser.add_argument("--encoder_ffn_embed_dim", type=int, default=768, help="")
    parser.add_argument(
        "--llm_hidden_size", type=int, default=256, help="encoder embedding dimension"
    )
    parser.add_argument("--llm_ffn_size", type=int, default=256, help="")
    parser.add_argument("--encoder_attention_heads", type=int, default=12, help="")
    parser.add_argument("--encoder_layers", type=int, default=12, help="")
    parser.add_argument("--max-nodes", type=int, default=8, help="")
    parser.add_argument("--add-3d", default=False, action="store_true", help="")
    parser.add_argument("--no-2d", default=False, action="store_true", help="")
    parser.add_argument("--num-3d-bias-kernel", type=int, default=128, help="")
    parser.add_argument("--num_pred_attn_layer", type=int, default=4, help="")
    parser.add_argument("--droppath_prob", type=float, default=0.0, help="")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="")
    parser.add_argument("--act_dropout", type=float, default=0.1, help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="")
    parser.add_argument("--sandwich_ln", default=True, action="store_true", help="")
    parser.add_argument("--ft", action="store_true", default=False, help="")
    parser.add_argument("--infer", action="store_true", default=False, help="")
    parser.add_argument("--noise_scale", type=float, default=0.2, help="")
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="")
    parser.add_argument("--log-interval", type=int, default=100, help="log per n steps")
    parser.add_argument(
        "--pipeline_parallelism", type=int, default=0, help="log per n steps"
    )
    parser.add_argument("--steps", type=int, default=10000000, help="log per n steps")
    parser.add_argument(
        "--output_path", type=str, default="/blob/output", help="log per n steps"
    )
    parser.add_argument(
        "--d_tilde", type=float, default=-1, help="mu transfer multiplier"
    )
    parser.add_argument("--max_lr", type=float, default=1e-3, help="max lr")
    parser.add_argument(
        "--total_num_steps",
        type=int,
        default=1000000,
    )
    parser.add_argument("--warmup_num_steps", type=int, default=60000)

    parser.add_argument(
        "--smiles_dict_path",
        type=str,
        default="/home/peiran/FMproj/moleculenet_data/data/mol2idx_dict.jsonl",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/peiran/FMproj/MetaLLM-converted",
    )
    parser.add_argument(
        "--loadmfmcheck_path",
        type=str,
        default="/home/peiran/FMproj/moleculenet_data/data/mol2idx_dict.jsonl",
    )
    parser.add_argument(
        "--loadllmcheck_path", type=str, default="/home/peiran/FMproj/MetaLLM-converted"
    )
    parser.add_argument("--dataset_names", type=str, default="")
    parser.add_argument("--dataset_ratios", type=str, default="")
    parser.add_argument("--dataset_splits", type=str, default="")
    parser.add_argument("--mol2idx_dict_path", type=str, default="")
    parser.add_argument("--in_memory", type=bool, default=False)
    parser.add_argument("--mol_size_path", type=str, default="")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args
