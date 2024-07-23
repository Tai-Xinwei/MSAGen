# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from sfm.tasks.nlm.eval.moe_inference_module import SFMMoEGenerator
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument("--mixtral_path", type=str, required=True)
    parser.add_argument("--nlm_path", type=str, required=True)
    parser.add_argument("--local_path", type=str, default="/dev/shm/nlm")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    generator = SFMMoEGenerator(
        args.mixtral_path,
        args.nlm_path,
        args.local_path,
    )

    inst, resp = [], []
    with open(args.input_file, "r") as f:
        for line in f:
            i, r = line.strip().split("\t")
            inst.append(i)
            resp.append(r)

    printed = False
    with open(args.output_file, "w") as f:
        for i, r in tqdm(zip(inst, resp), total=len(inst)):
            for pred in generator.chat(i, response_only=True):
                if not printed:
                    print(pred)
                pred = pred.replace('<m>', '').replace(' ', '')
                if not printed:
                    print(pred)
                f.write(pred + "\n")
                printed = True

if __name__ == "__main__":
    main()
