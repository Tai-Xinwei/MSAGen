# -*- coding: utf-8 -*-
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from sfm.data.mol_data.moltext_dataset import smiles2graph_removeh
from sfm.logging.loggers import logger as sfm_logger
from sfm.models.generalist import GraphormerLlamaModel
from sfm.models.generalist.generalist_config import GeneralistConfig
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.pipeline.accelerator.dataclasses import DistributedConfig, TrainerConfig
from sfm.utils.cli_utils import cli
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS


class EvalMethod(Enum):
    AROMATIC = "AROMATIC"
    NITRO = "NITRO"
    FUNCG = "FUNCG"
    NONE = "NONE"

    def __str__(self):
        return self.value


def test_nitro(smiles, response):
    label = int(smiles.find("n") != -1 or smiles.find("N") != -1)
    if (
        response.find("molecule contains no") != -1
        or response.find("molecule has no") != -1
        or response.find("molecule does not") != -1
        or (
            response.find("nitro") != -1
            and (response.find(" no ") != -1 or response.find(" not ") != -1)
        )
        or response.find("false") != -1
        or response.find("False") != -1
        or response.find("No") != -1
    ):
        score = -1
    elif (
        response.find("molecule contains") != -1
        or response.find("molecule has") != -1
        or response.find("molecule does contain") != -1
        or response.find("is a nitrogen") != -1
        or response.find("is a nitrile") != -1
        or response.find("Yes") != -1
        or response.find("yes") != -1
        or response.find("organonitrogen") != -1
        or response.find("is a nitroso") != -1
        or (
            response.find("nitr") != -1
            and response.find(" no ") == -1
            and response.find(" not ") == -1
        )
        or response.find("true") != -1
        or response.find("True") != -1
        or response.find(" N-") != -1
    ):
        score = 1
    else:
        score = 0
    return label, score


def test_aromatic(smiles, response):
    mol = Chem.MolFromSmiles(smiles)
    is_aromatic = False
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            is_aromatic = True
    label = int(is_aromatic)
    if (
        (response.find(" not ") != -1 or response.find(" no ") != -1)
        and response.find("aroma") != -1
    ) or response.find("No") != -1:
        score = -1
    elif (
        response.find("aroma") != -1
        or response.find("Yes") != -1
        or response.find("yes") != -1
        or response.find("YES") != -1
    ):
        score = 1
    else:
        score = 0
    return label, score


def test_funcg(smiles, response, local_rank):
    try:
        os.system(f"obabel -:'{smiles}' -omol > tmp{local_rank}.mol")
        func_groups = subprocess.check_output(["./checkmol", f"tmp{local_rank}.mol"])
        func_groups_str = func_groups.decode().strip()
        os.system(f"rm tmp{local_rank}.mol")
        func_groups = list(filter(lambda x: x != "", func_groups_str.split("\n")))
    except Exception as e:
        sfm_logger.info(f"{e}")
        func_groups = []

    response = response.lower()

    recall = 0.0
    for func_group in func_groups:
        if response.find(func_group) != -1:
            recall += 1
    if len(func_groups) > 0:
        recall /= len(func_groups)

    func_groups_in_response = list(
        set(
            list(
                filter(
                    lambda x: x != "" and (not x.endswith(" of")),
                    re.split(",| a | and a | and ", response),
                )
            )[1:]
        )
    )
    precision = 0.0
    for func_group in func_groups_in_response:
        for label_func_group in func_groups:
            if (
                func_group.find(label_func_group) != -1
                or label_func_group.find(func_group) != -1
            ):
                precision += 1
                break
    if len(func_groups_in_response) > 0:
        precision /= len(func_groups_in_response)

    return precision, recall, func_groups, func_groups_in_response


def get_test_question(eval_method: EvalMethod):
    if eval_method == EvalMethod.AROMATIC:
        question = "Does the molecule contain aromatic rings?"
    elif eval_method == EvalMethod.NITRO:
        question = "Does the molecule contain nitrogen atoms?"
    elif eval_method == EvalMethod.FUNCG:
        question = "In the given input molecular structure, determine and list the functional groups, presenting them in a natural sentence."
    else:
        raise ValueError(f"No question for EvalMethod {eval_method}.")

    return question


def gather_labels_and_scores(labels, scores, world_size, local_rank, num_total_smiless):
    if world_size > 1:
        labels_tensor = torch.tensor(labels, device=f"cuda:{local_rank}")
        scores_tensor = torch.tensor(scores, device=f"cuda:{local_rank}")
        labels_tensor_list = [
            torch.zeros_like(labels_tensor) for _ in range(world_size)
        ]
        scores_tensor_list = [
            torch.zeros_like(scores_tensor) for _ in range(world_size)
        ]
        dist.all_gather(labels_tensor_list, labels_tensor)
        dist.all_gather(scores_tensor_list, scores_tensor)
        all_labels = torch.cat(labels_tensor_list, dim=-1).cpu()
        all_scores = torch.cat(scores_tensor_list, dim=-1).cpu()
        return (
            [float(x) for x in all_labels][:num_total_smiless],
            [float(x) for x in all_scores][:num_total_smiless],
        )
    else:
        return labels, scores


@dataclass
class TestGeneralistConfig:
    num_gpus: int
    test_checkpoint_path: str
    test_global_step: int
    remote_checkpoint_query_string: str
    remote_checkpoint_storage_account: str
    remote_checkpoint_storage_container: str
    local_checkpoint_path: str
    smiles_list_fname: str
    infer_batch_size: int
    output_fname: str
    num_eval_repeats: int
    question_list_fname: Optional[str] = None
    eval_method: EvalMethod = EvalMethod.NONE


def create_device_map(num_gpus, num_llama_layers):
    device_map = {
        "graphormer_encoder": 0,
        "decoder.model.embed_tokens": 0,
        "adaptor": 0,
    }
    num_llama_layers_per_gpu = (num_llama_layers + num_gpus - 1) // num_gpus
    for i in range(num_gpus):
        start_layer = i * num_llama_layers_per_gpu
        end_layer = min(start_layer + num_llama_layers_per_gpu, num_llama_layers)
        for j in range(start_layer, end_layer):
            device_map[f"decoder.model.layers.{j}"] = i
    device_map["decoder.model.norm"] = num_gpus - 1
    device_map["decoder.lm_head"] = 0
    device_map["decoder.num_head"] = 0
    return device_map


def download_model(
    local_rank,
    model_path,
    global_step,
    local_path,
    remote_checkpoint_query_string,
    remote_checkpoint_storage_account,
    remote_checkpoint_storage_container,
):
    is_remote = model_path.find("remote:") != -1
    model_name = model_path.split("remote:")[-1]
    convert_offset = 0
    if model_name.find("7B") != -1 or model_name.find("7b") != -1:
        # support only llama2, drop llama
        num_layers = 37
        model_size = "7B"
    elif model_name.find("65B") != -1 or model_name.find("65b") != -1:
        num_layers = 84
        model_size = "65B"
    elif model_name.find("70B") != -1 or model_name.find("70b") != -1:
        num_layers = 85
        model_size = "70B"
    else:
        raise ValueError(f"Unknown model size in {model_name}")

    # support only llama2, drop llama
    convert_offset = 2

    if is_remote:
        sfm_logger.info("Downloading model...")
        sas_url = "https://{remote_checkpoint_storage_account}.blob.core.windows.net/{remote_checkpoint_storage_container}/{path}{remote_checkpoint_query_string}"
        dest_path = f"{local_path}/{model_name}/global_step{global_step}/"
        if local_rank == 0:
            os.system(f"mkdir -p {dest_path}")
            for i in tqdm(range(num_layers)):
                if not os.path.exists(f"{dest_path}/layer_{i:02d}-model_states.pt"):
                    os.system(
                        (f"azcopy copy '{sas_url}' {dest_path}/").format_map(
                            {
                                "path": f"{model_name.strip('/')}/global_step{global_step}/layer_{i:02d}-model_states.pt",
                                "remote_checkpoint_storage_account": remote_checkpoint_storage_account.strip(
                                    "/"
                                ),
                                "remote_checkpoint_storage_container": remote_checkpoint_storage_container.strip(
                                    "/"
                                ),
                                "remote_checkpoint_query_string": remote_checkpoint_query_string,
                            }
                        )
                    )
    else:
        dest_path = f"{model_path}/global_step{global_step}"
    return dest_path, num_layers, model_size, convert_offset


def convert(in_dir, out_dir, num_layers, llm_model_name_or_path, convert_offset):
    total_size = 0
    index_map = {"weight_map": {}}
    sfm_logger.info("Converting model...")

    need_convert = False
    need_convert |= not os.path.exists(f"{out_dir}/config.json")
    need_convert |= not os.path.exists(f"{out_dir}/pytorch_model.bin.index.json")
    for i in range(num_layers):
        need_convert |= not os.path.exists(f"{out_dir}/layer_{i:02d}-model_states.bin")

    if not need_convert:
        sfm_logger.info("Converted model exists.")
        return

    shutil.copy(f"{llm_model_name_or_path}/config.json", f"{out_dir}/")
    for i in tqdm(range(num_layers)):
        new_model_states = {}
        ckpt_path = f"{in_dir}/layer_{i:02d}-model_states.pt"
        model_states = torch.load(ckpt_path, map_location="cpu")
        all_keys = list(model_states.keys())
        for key in all_keys:
            if key.find("dummy") != -1:
                continue
            weight = model_states[key]
            if convert_offset == 2:  # llama2
                if i == 0:
                    # molecular model
                    new_key = "graphormer_encoder." + key
                elif i == 1:
                    # embed tokens
                    new_key = "decoder.model." + key
                elif i == 2:
                    # hybrid embedding
                    new_key = "adaptor." + key
                elif i < num_layers - convert_offset:
                    new_key = f"decoder.model.layers.{i - 3}." + key
                elif i == num_layers - convert_offset:
                    new_key = "decoder.model." + key
                else:
                    new_key = "decoder." + key
            else:  # llama
                if i == 0:
                    new_key = "model.molecule_embedding." + key
                elif i < num_layers - 3:
                    new_key = f"model.layers.{i - 1}." + key
                elif i == num_layers - 3:
                    new_key = "model.norm." + key
                else:
                    new_key = "llama_lm_head_and_loss." + key

            index_map["weight_map"][new_key] = f"layer_{i:02d}-model_states.bin"
            total_size += weight.nelement() * weight.element_size()
            new_model_states[new_key] = weight
        torch.save(new_model_states, f"{out_dir}/layer_{i:02d}-model_states.bin")

    index_map["total_size"] = total_size

    with open(f"{out_dir}/pytorch_model.bin.index.json", "w") as out_file:
        json.dump(index_map, out_file)


def get_tokenizer(llm_model_name_or_path):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        llm_model_name_or_path,
        cache_dir=False,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    special_tokens_dict["additional_special_tokens"] = SCIENCE_TAG_TOKENS
    new_num_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer, tokenizer.vocab_size + new_num_tokens


def batch_mol(molecules, num_batches):
    num_molecules = len(molecules)
    batch_size = (num_molecules + num_batches - 1) // num_batches
    batch_start = batch_size * np.arange(num_batches)
    batch_end = batch_start + batch_size
    batch_end[batch_end > num_molecules] = num_molecules
    batched_molecules = [
        molecules[batch_start[i] : batch_end[i]] for i in range(num_batches)
    ]
    for i in range(len(batched_molecules)):
        for _ in range(len(batched_molecules[i]), batch_size):
            batched_molecules[i].append(molecules[0])
    return batched_molecules


@cli(
    GraphormerConfig,
    GeneralistConfig,
    TestGeneralistConfig,
    DistributedConfig,
    TrainerConfig,
)
def main(args) -> None:
    tokenizer, vocab_size = get_tokenizer(args.llm_model_name_or_path)

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(
            backend="nccl", rank=int(os.environ["LOCAL_RANK"]), world_size=world_size
        )
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    local_model_path, num_layers, model_size, convert_offset = download_model(
        args.local_rank,
        args.test_checkpoint_path,
        args.test_global_step,
        args.local_checkpoint_path,
        args.remote_checkpoint_query_string,
        args.remote_checkpoint_storage_account,
        args.remote_checkpoint_storage_container,
    )

    if model_size == "7B":
        with torch.no_grad():
            model = GraphormerLlamaModel(args, vocab_size)
            model = model.to(device=f"cuda:{args.local_rank}")
        device_map = {"": f"cuda:{args.local_rank}"}
    else:
        with init_empty_weights():
            model = GraphormerLlamaModel(args, vocab_size)
        device_map = create_device_map(
            args.num_gpus, model.llama_config.num_hidden_layers
        )

    if args.local_rank == 0:
        convert(
            local_model_path,
            local_model_path,
            num_layers,
            args.llm_model_name_or_path,
            convert_offset,
        )

    if world_size > 1:
        dist.barrier()

    sfm_logger.info("Loading model...")
    model = load_checkpoint_and_dispatch(
        model,
        local_model_path,
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer", "GraphormerSentenceEncoder"],
        dtype=torch.float16,
    )

    assert (
        args.question_list_fname is not None or args.eval_method != EvalMethod.NONE
    ), "Either a question list file or a evaluation method must be provided for evaluation."
    if args.eval_method != EvalMethod.NONE:
        questions = [
            get_test_question(args.eval_method) for _ in range(args.num_eval_repeats)
        ]
    else:
        with open(args.question_list_fname, "r") as in_file:
            questions = [question.strip() for question in list(in_file.readlines())]

    with open(args.smiles_list_fname, "r") as in_file:
        smiless = [smi.strip() for smi in list(in_file.readlines())]

    num_total_smiless = len(smiless)

    if model_size == "7B":
        smiless = batch_mol(smiless, world_size)[args.local_rank]

    local_rank = args.local_rank if args.local_rank >= 0 else 0
    sfm_logger.info("Start testing...")

    if args.eval_method != EvalMethod.NONE:
        if args.eval_method != EvalMethod.FUNCG:
            scores = []
            labels = []
        else:
            precisions = []
            recalls = []

    with open(f"{args.output_fname}.{local_rank}", "w") as out_file:
        for i, smi in enumerate(tqdm(smiless)):
            num_atoms = smiles2graph_removeh(smi)["x"].size()[0]
            if args.eval_method != EvalMethod.NONE:
                if args.eval_method != EvalMethod.FUNCG:
                    score_sum = 0.0
                else:
                    precision_sum = 0.0
                    recall_sum = 0.0
            for question in questions:
                prompt = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{question}\n\n### Input:\n<mol> {''.join(['<unk>' for _ in range(num_atoms)])} </mol>\n\n### Response:\n"
                )
                input_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="longest",
                    max_length=512,
                    truncation=True,
                ).input_ids[0]
                input_ids[input_ids == 0] = -1
                input_ids = input_ids.to(f"cuda:{local_rank}")
                res = model.generate_with_smiles(
                    input_ids.unsqueeze(0),
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=128,
                    output_scores=True,
                    return_dict_in_generate=True,
                    smiles=[smi],
                )
                seq = res.sequences[0]
                seq[seq < 0] = 0
                response = (
                    tokenizer.decode(seq, skip_special_tokens=False)
                    .split("### Response:\n")[-1]
                    .strip()
                )
                json_obj = {"smiles": smi, "prompt": prompt, "response": response}

                if args.eval_method != EvalMethod.NONE:
                    if args.eval_method == EvalMethod.NITRO:
                        label, score = test_nitro(smi, response)
                        json_obj["label"] = label
                        json_obj["score"] = score
                        score_sum += score
                    elif args.eval_method == EvalMethod.AROMATIC:
                        label, score = test_aromatic(smi, response)
                        json_obj["label"] = label
                        json_obj["score"] = score
                        score_sum += score
                    elif args.eval_method == EvalMethod.FUNCG:
                        (
                            precision,
                            recall,
                            func_groups,
                            func_groups_in_response,
                        ) = test_funcg(smi, response, args.local_rank)
                        json_obj["precision"] = precision
                        json_obj["recall"] = recall
                        json_obj["func_groups"] = func_groups
                        json_obj["func_groups_in_response"] = func_groups_in_response
                        precision_sum += precision
                        recall_sum += recall

                out_file.write(
                    json.dumps(
                        json_obj,
                        indent=6,
                    )
                    + "\n"
                )
                out_file.flush()
            if args.eval_method != EvalMethod.NONE:
                if args.eval_method != EvalMethod.FUNCG:
                    labels.append(label)
                    scores.append(score_sum)
                else:
                    precisions.append(precision_sum / args.num_eval_repeats)
                    recalls.append(recall_sum / args.num_eval_repeats)
            if args.eval_method != EvalMethod.NONE:
                if args.eval_method != EvalMethod.FUNCG:
                    all_labels, all_scores = gather_labels_and_scores(
                        labels, scores, world_size, args.local_rank, num_total_smiless
                    )
                    try:
                        auc = roc_auc_score(all_labels, all_scores)
                    except Exception as e:
                        sfm_logger.info(f"{e}")
                        auc = np.nan
                    sfm_logger.info(
                        f"Evaluation in progress {i + 1}/{len(smiless)}, method: {args.eval_method}, auc score: {auc}"
                    )
                else:
                    all_precisions, all_recalls = gather_labels_and_scores(
                        precisions,
                        recalls,
                        world_size,
                        args.local_rank,
                        num_total_smiless,
                    )
                    sfm_logger.info(
                        f"Evaluation in progress {i + 1}/{len(smiless)}, method: {args.eval_method}, precision: {np.mean(all_precisions)}, recall: {np.mean(all_recalls)}"
                    )

    if args.eval_method != EvalMethod.NONE:
        if args.eval_method != EvalMethod.FUNCG:
            all_labels, all_scores = gather_labels_and_scores(
                labels, scores, world_size, args.local_rank, num_total_smiless
            )
            try:
                auc = roc_auc_score(all_labels, all_scores)
            except Exception as e:
                sfm_logger.info(f"{e}")
                auc = np.nan
            sfm_logger.info(
                f"Evaluation finished, method: {args.eval_method}, auc score: {auc}"
            )
        else:
            all_precisions, all_recalls = gather_labels_and_scores(
                precisions, recalls, world_size, args.local_rank, num_total_smiless
            )
            sfm_logger.info(
                f"Evaluation finished, method: {args.eval_method}, precision: {np.mean(all_precisions)}, recall: {np.mean(all_recalls)}"
            )


if __name__ == "__main__":
    main()
