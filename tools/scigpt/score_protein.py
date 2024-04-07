# -*- coding: utf-8 -*-
import torch
import os
import argparse

from multiprocessing import Process, Queue, set_start_method

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

def load_ckpt(ckpt_home):
    ckpt_dict = {}

    layer0 = torch.load(os.path.join(ckpt_home, "layer_00-model_states.pt"), map_location=torch.device("cpu"))
    ckpt_dict['model.embed_tokens.weight'] = layer0['embed_tokens.weight']

    for l in range(0, 32):
        l_index = str(l + 1).zfill(2)
        layer = torch.load(os.path.join(ckpt_home, f"layer_{l_index}-model_states.pt"), map_location=torch.device("cpu"))
        for k in layer:
            if "dummy" in k or 'rotary_emb' in k:
                continue
            ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]

    layer = torch.load(os.path.join(ckpt_home, "layer_33-model_states.pt"), map_location=torch.device("cpu"))
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]

    layer = torch.load(os.path.join(ckpt_home, "layer_34-model_states.pt"), map_location=torch.device("cpu"))
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]

    return ckpt_dict

def load_model(ckpt_dict, gpu_id):
    tokenizer_home = '/hai1/ds_dataset/llama2/llama-2-7b'
    tokenizer = SFMDecTokenizer.from_pretrained(
        tokenizer_home,
        prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
        dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe',
        rna_spm_path='/blob/shufxi/data/scigpt/rnabpe/bpe',
    )

    model = AutoModelForCausalLM.from_pretrained(tokenizer_home)
    model_dict = model.state_dict()
    model_dict.update(ckpt_dict)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(model_dict)
    model.eval()
    model.to(gpu_id)
    return model, tokenizer


def worker(ckpt_home, gpu_id, input_queue, output_queue):
    ckpt_dict = load_ckpt(ckpt_home)
    model, tokenizer = load_model(ckpt_dict, gpu_id)
    print('model loaded on', gpu_id)

    def compute_score(seq):
        inputs = tokenizer(seq, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda(gpu_id)
        attention_mask = inputs["attention_mask"].cuda(gpu_id)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        logits = logits[0, :-1, :]
        label = input_ids[0, 1:]

        loss = torch.nn.functional.cross_entropy(logits, label)
        return loss.item()

    while True:
        sentences = input_queue.get()
        scores = []
        if sentences is None:
            break

        for sentence in sentences:
            try:
                score = compute_score(sentence)
                scores.append((sentence, score))
            except Exception as e:
                print(e)
                pass

        output = []
        for sentence, score in scores:
            output.append((sentence, score))
        output_queue.put(output)


def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Ckpt path')
    parser.add_argument('--input', type=str, help='Input file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.input + '.scored'

    inputs = []
    with open(input_path, 'r') as f:
        for line in f:
            inputs.append(line.strip())

    computed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                computed.add(line.split('\t')[0])
    to_compute = list(filter(lambda x: x not in computed, inputs))

    print('total input', len(inputs), 'computed', len(computed), 'to compute', len(to_compute))

    num_gpus = torch.cuda.device_count()
    print('processing on', num_gpus, 'GPUs')

    sentences_queue = Queue()
    results_queue = Queue()
    processes = []

    for gpu_id in range(num_gpus):
        p = Process(target=worker, args=(args.model, gpu_id, sentences_queue, results_queue))
        p.start()
        processes.append(p)

    batch_size = args.batch_size

    output_file = open(output_path, 'a')

    for i in range(0, len(to_compute), batch_size):
        sentences_queue.put(to_compute[i:i+batch_size])

    n_success = 0
    bar = tqdm(total=len(to_compute))
    while True:
        results = results_queue.get()
        for sentence, score in results:
            output_file.write(f"{sentence}\t{score}\n")
            output_file.flush()
            n_success += 1
            bar.update(1)

        if n_success == len(to_compute):
            break

    print('done')

    for gpu_id in range(num_gpus):
        sentences_queue.put(None)

    for p in processes:
        p.join()

    output_file.close()


if __name__ == '__main__':
    main()
