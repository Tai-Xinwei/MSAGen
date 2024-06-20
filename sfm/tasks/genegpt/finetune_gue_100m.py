# -*- coding: utf-8 -*-
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from sklearn.metrics import roc_auc_score

# from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sfm.data.gene_data.GeneTokenizer import GeneKMerTokenizer

# from sfm.models.genegpt.genegpt_3d import GenegptModel
from sfm.models.genegpt.genegpt_config import GenegptConfig3D, genegpt3D_100m_config

# from torch.utils.data import Dataset


dist.init_process_group(
    backend="nccl", init_method="tcp://localhost:23457", rank=0, world_size=1
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(
        default=0.05, metadata={"help": "dropout rate for LoRA"}
    )
    lora_target_modules: str = field(
        default="query,value", metadata={"help": "where to perform LoRA"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    kmer: int = field(
        default=-1,
        metadata={"help": "k-mer for input sequence. -1 means not using k-mer."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = (field(default="steps"),)
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""


def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])


"""
Transform a dna sequence to k-mer string
"""


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i : i + k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning("Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)

    return kmer


def multi_hot(labels, num_labels):
    """
    Convert a numpy array to a one-hot encoded numpy array.
    """
    encoded = np.eye(num_labels, dtype=np.int64)[labels].sum(axis=0)
    return encoded


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: GeneKMerTokenizer, kmer: int = 6):
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            # texts = load_or_generate_kmer(data_path, texts, kmer)
            input_ids = []
            input_ids = tokenizer.encode_sequences(
                texts, padding=True, truncation=True
            )["input_ids"]
            self.input_ids = torch.tensor(input_ids)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        else:
            raise ValueError("Only 6 mer!!")

        self.num_labels = len(set(labels))
        print(self.num_labels)
        hot_label = []
        for label in labels:
            label = (
                list(map(int, label.split(","))) if isinstance(label, str) else [label]
            )
            label = multi_hot(label, self.num_labels)
            hot_label.append(label)
        self.labels = hot_label

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: GeneKMerTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.tensor(labels, dtype=torch.int64)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = (
        labels != -100
    )  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


def prepare_model(config):
    # model = GenegptModel(config)
    llama_config_path = "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/config/config_100m/config.json"
    llama_config = LlamaConfig.from_json_file(llama_config_path)
    model = LlamaForCausalLM(llama_config)
    model_dict = model.state_dict()
    ckpt_dict = {}
    print(model_dict.keys())
    flag = ""
    if not os.path.exists(
        os.path.join(config.pretrained_ckpt_path, "layer_00-model_states.pt")
    ):
        flag = "_00-model"
    ckpt_dict = {}
    layer0 = torch.load(
        os.path.join(config.pretrained_ckpt_path, f"layer_00-model{flag}_states.pt"),
        map_location=torch.device("cpu"),
    )
    print(layer0.keys())
    ckpt_dict["model.embed_tokens.weight"] = layer0["word_embeddings.weight"]

    for l in range(0, config.num_hidden_layers):
        l_index = str(l + 1).zfill(2)
        layer = torch.load(
            os.path.join(
                config.pretrained_ckpt_path, f"layer_{l_index}-model{flag}_states.pt"
            ),
            map_location=torch.device("cpu"),
        )
        for k in layer:
            if "dummy" in k or "rotary_emb" in k:
                continue
            if k == "self_attention.layernorm_qkv.query_weight":
                ckpt_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = layer[k]
            elif k == "self_attention.layernorm_qkv.key_weight":
                ckpt_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = layer[k]
            elif k == "self_attention.layernorm_qkv.value_weight":
                ckpt_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = layer[k]
            elif k == "self_attention.proj.weight":
                ckpt_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = layer[k]
            elif k == "self_attention.layernorm_qkv.layer_norm_weight":
                ckpt_dict[f"model.layers.{l}.input_layernorm.weight"] = layer[k]
            elif k == "layernorm_mlp.layer_norm_weight":
                ckpt_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = layer[
                    k
                ]
            elif k == "self_attention.proj.weight":
                ckpt_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = layer[k]
            elif k == "layernorm_mlp.fc2_weight":
                ckpt_dict[f"model.layers.{l}.mlp.down_proj.weight"] = layer[k]
            elif k == "layernorm_mlp.fc1_weight":
                splits = torch.split(layer[k], int(layer[k].size(0) / 2))
                ckpt_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = splits[0]
                ckpt_dict[f"model.layers.{l}.mlp.up_proj.weight"] = splits[1]
    layer = torch.load(
        os.path.join(
            config.pretrained_ckpt_path,
            f"layer_{config.num_hidden_layers+1}-model{flag}_states.pt",
        ),
        map_location=torch.device("cpu"),
    )
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]
    layer = torch.load(
        os.path.join(
            config.pretrained_ckpt_path,
            f"layer_{config.num_hidden_layers+2}-model{flag}_states.pt",
        ),
        map_location=torch.device("cpu"),
    )
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
    model_dict.update(ckpt_dict)

    model.load_state_dict(model_dict)
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Assuming the output needs to be sigmoid-transformed to represent probabilities
    probabilities = 1 / (1 + np.exp(-logits))  # Apply the sigmoid function

    # Number of labels
    num_labels = probabilities.shape[1]
    # Calculate AUC for each label
    auc_scores = []
    for label in range(num_labels):
        # Only consider labels with at least one positive reference to avoid ill-defined cases
        if np.sum(labels[:, label]) > 0:
            auc = roc_auc_score(labels[:, label], probabilities[:, label])
            auc_scores.append(auc)
        else:
            # If there are no positive references, handle this case as needed; here we can skip or use NaN
            auc_scores.append(np.nan)  # Skipped or not defined

    # Calculate the average AUC across all labels, ignoring NaN values
    average_auc = np.nanmean(auc_scores)  # Compute mean AUROC ignoring NaN values

    return {
        "average_auroc": average_auc,
        # "auroc_per_class": auc_scores  # Optionally include individual AUROC scores
    }


"""
Compute metrics used for huggingface trainer.
"""


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     if isinstance(logits, tuple):  # Unpack logits if it's a tuple
#         logits = logits[0]
#     return calculate_metric_with_sklearn(logits, labels)


class DNAClassification(PreTrainedModel):
    def __init__(self, config, num_labels, pooling_type="mean"):
        config.update({"num_labels": num_labels})
        super().__init__(config)
        self.num_labels = num_labels
        self.backbone = prepare_model(config)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss

        # self.cnn = nn.Sequential(
        #     nn.Conv1d(config.hidden_size, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1),
        #     nn.GELU(),
        # )

        # self.classifier = nn.Linear(64, num_labels)
        self.pooling_type = pooling_type

        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # frozenm backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def pooling(self, hidden_states):
        # hidden_states: (bs, seq_len, hidden_size)
        if self.pooling_type == "mean":
            return hidden_states.mean(dim=1)
        elif self.pooling_type == "max":
            return hidden_states.max(dim=1).values
        elif self.pooling_type == "min":
            return hidden_states.min(dim=1).values

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            **kwargs,
        )
        sequence_output = outputs["hidden_states"][-1]
        sequence_output = self.pooling(sequence_output)

        # sequence_output = outputs["hidden_states"][-1]  # (bs, seq_len, hidden_size)
        # # cls_hidden_state = sequence_output[:, 0, :] # (bs, hidden_size)
        # sequence_output = sequence_output[:, 1:, :]  # remove the [CLS] token
        # sequence_output = sequence_output.permute(0, 2, 1)  # (bs, hidden_size, seq_len)
        # sequence_output = self.cnn(sequence_output)  # (bs, 64, seq_len)
        # # re-permute
        # sequence_output = sequence_output.permute(0, 2, 1)  # (bs, seq_len, 64)
        # sequence_output = self.pooling(sequence_output)  # (bs, 64)

        logits = self.classifier(sequence_output)  # (bs, num_labels)

        loss = None
        # print(logits.shape)
        # print(labels.shape)
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = GeneKMerTokenizer()
    config = GenegptConfig3D()
    config = genegpt3D_100m_config(config)
    config.load_ckpt = True
    config.infer = True
    config.pretrained_ckpt_path = "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/100m6kmer160k/global_step3000"

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator

    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer,
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer,
    )
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "train.csv"),
        kmer=data_args.kmer,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    model = DNAClassification(config, num_labels=train_dataset.num_labels)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("name: ", name)
            print("parameter size: ", param.size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of trainable parameters is: ", total_params)
    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(
            training_args.output_dir, "results", training_args.run_name
        )
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    train()
