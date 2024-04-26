# -*- coding: utf-8 -*-
import os
import json
import random
import time
from tqdm import tqdm
from datetime import timedelta
import fire
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed, get_linear_schedule_with_warmup
from datasets import Dataset, disable_caching

from utils import process_pubmed_xml

SEED = 42

#empty cache before start the code
disable_caching()
torch.cuda.empty_cache()


class Model(nn.Module):
    def __init__(self, bert_model, num_classes, hidden_size, dropout=0.1):
        super().__init__()
        self.bert_model = bert_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self,
                input_ids,
                attention_mask,
                **kwargs,
        ):
        outputs = self.bert_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True,)
        pooled_output = outputs.hidden_states[-1][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {
            "logits": logits
        }



class Collator():
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def collate_fn(self, batch):
        inputs = [item["input"] for item in batch]
        if "label" in batch[0]:
            labels = [item["label"] for item in batch]
        else:
            labels = None
        outputs = self.tokenizer(
            inputs,
            add_special_tokens = True, # Add "[CLS]" and "[SEP]"
            max_length = self.max_len,
            truncation=True,
            padding=True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        labels = torch.tensor(labels) if labels is not None else None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input": inputs,
        }


def preprocess(data):
    data["input"] = data["input"].lower()
    return data

def move_to_device(batch, device='cuda'):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def train(
    model_name_or_path: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    data: str = "data/ABCD.jsonl",
    num_labels: int = 2,
    epochs: int = 20,
    batch_size: int = 32,
    max_length: int = 512,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-6,
    max_grad_norm: float = 1.0,
    output_dir: str = "output",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    model = Model(bert_model, num_labels, hidden_size=bert_model.config.hidden_size)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load data
    with open(data) as f:
        data = [json.loads(line) for line in f]
    pos_data = []
    neg_data = []
    for item in data:
        if item["answer"] == "None":
            neg_data.append(item["text"])
        else:
            pos_data.append(item["text"])

    random.shuffle(neg_data)
    neg_data = neg_data[:len(pos_data)]
    data = pos_data + neg_data
    labels = [1] * len(pos_data) + [0] * len(neg_data)

    dataset = Dataset.from_dict(
        {
            "input": data,
            "label": labels,
        }
    )

    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=SEED)
    collator = Collator(tokenizer, max_length)
    train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collator.collate_fn)
    test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=collator.collate_fn)

    # move model to GPU
    model.cuda()

    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=adam_epsilon, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(total_steps * 0.1),
        num_training_steps = total_steps
    )
    scaler = torch.cuda.amp.GradScaler()
    t0 = time.time()
    best_dev_score = -100.0
    for epoch in range(epochs):
        print("======== Epoch {:} / {:} ========".format(epoch + 1, epochs))
        for step, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch = move_to_device(batch)
                outputs = model(**batch)
                logits = outputs["logits"]
                loss = loss_f(logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step == 0 or (step+1) % 100 == 0:
                elapsed = timedelta(seconds=int(round(time.time() - t0)))
                print(f"Epoch {epoch}, Batch {step} / {len(train_dataloader)}, Elapsed: {elapsed}, Loss {loss.item()}")
        # evaluate on dev set
        model.eval()
        dev_loss = 0.0
        dev_steps = 0
        dev_correct = 0
        for batch in test_dataloader:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    batch = move_to_device(batch)
                    outputs = model(**batch)
                    logits = outputs["logits"]
                    loss = loss_f(logits, batch["labels"])
            dev_loss += loss.item()
            dev_steps += 1
            preds = torch.argmax(logits, dim=1)
            dev_correct += (preds == batch["labels"]).sum().item()
        dev_loss /= dev_steps
        dev_acc = dev_correct / len(dataset["test"])
        print(f"Dev loss {dev_loss}, Dev acc {dev_acc}")
        if dev_acc > best_dev_score:
            best_dev_score = dev_acc
            output_name = f"{model_name_or_path.split('/')[-1]}_e{epochs}_bs{batch_size}_lr{lr}_best-{epoch+1}"
            if not os.path.exists(os.path.join(output_dir, output_name)):
                os.makedirs(os.path.join(output_dir, output_name))
            torch.save(model.state_dict(), os.path.join(output_dir, output_name, "model.bin"))
            #model.save_pretrained(os.path.join(output_dir, output_name))
            tokenizer.save_pretrained(os.path.join(output_dir, output_name))
            print(f"Saving best model at epoch {epoch+1} with best dev score {best_dev_score}")


def infer(
    model_name_or_path: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    data: str = "/blob/renqian/data/PubMed/all.txt",
    num_labels: int = 2,
    batch_size: int = 1024,
    max_length: int = 512,
    output_dir: str = "outputs/BiomedNLP-PubMedBERT-base-uncased-abstract_e20_bs32_lr3e-05_best-20",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    model = Model(bert_model, num_labels, hidden_size=bert_model.config.hidden_size)
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.bin")))
    model.eval()

    lines = []
    with open(data) as f:
        for line in f:
            lines.append(line.strip())
    dataset = Dataset.from_dict({"input": lines})
    collator = Collator(tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.collate_fn)

    output = os.path.join(output_dir, f"infer.jsonl")
    with open(output, "w") as f:
        for batch in tqdm(dataloader):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    batch = move_to_device(batch)
                    outputs = model(**batch)
                    logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)
            for text, preds in zip(batch["input"], preds):
                f.write(json.dumps({"text": text, "label": preds.item()}) + "\n")


def main(
    do_train: bool = True,
    do_infer: bool = False,
    model_name_or_path: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    data: str = "data/ABCD.jsonl",
    num_labels: int = 2,
    epochs: int = 20,
    batch_size: int = 32,
    max_length: int = 512,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-6,
    max_grad_norm: float = 1.0,
    output_dir: str = "outputs",
):
    set_seed(SEED)

    if do_train:
        train(
            model_name_or_path,
            data,
            num_labels,
            epochs,
            batch_size,
            max_length,
            lr,
            weight_decay,
            adam_epsilon,
            max_grad_norm,
            output_dir,
        )
    elif do_infer:
        infer(
            model_name_or_path,
            data,
            num_labels,
            batch_size,
            max_length,
            output_dir,
        )

if __name__ == "__main__":
    fire.Fire(main)
