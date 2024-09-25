# -*- coding: utf-8 -*-
import argparse
import os
import pickle as pkl
from glob import glob

from mixtral_8x7b_vllm_infer_mod import SFMMoEVLLMGenerator
from tqdm import tqdm


class NLMMoEVLLMInferencer:
    def __init__(self, input_dir, output_dir, mixtral_path, nlm_local_path):
        self.generator = SFMMoEVLLMGenerator(mixtral_path, nlm_local_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_names = [
            "test.bbbp.instruct.tsv",
            "test.bace.instruct.tsv",
            "test.desc2mol.tsv",
            "test.mol2desc.tsv",
            "test.raw.i2s_i.txt",
            "test.raw.s2i_s.txt",
            "test.molinstruct.reaction.tsv",
            "test.hERG.tsv",
            "test.uspto50k.retro.osmi.tsv",
        ]
        self.file_paths = [os.path.join(input_dir, fn) for fn in file_names]
        print("Found {} test files".format(len(self.file_paths)))
        print("Test files: {}".format(self.file_paths))

    def generate_responses(self):
        print("Generating responses for test files")
        for fn in self.file_paths:
            print("Checking file: {}".format(fn))
            basename = os.path.basename(fn)
            fn_out = basename.replace(".txt", "").replace(".tsv", "") + ".response.pkl"
            fn_out = os.path.join(self.output_dir, fn_out)

            if os.path.exists(fn_out):
                print("File {} already exists, skipping.".format(fn_out))
                continue
            else:
                print("File {} does not exist, processing.".format(fn_out))

            with open(fn, "r", encoding="utf8") as fr:
                all_lines = [e.strip() for e in fr]

            qa_gt_pairs = []
            questions = []
            for idx, test_sample in tqdm(enumerate(all_lines), total=len(all_lines)):
                question = test_sample.split("\t")[0].strip()
                qa_gt_pairs.append(test_sample)
                questions.append(question)
                # r0 = self.generator.chat(q, do_sample=False)
                # r1 = self.generator.chat(q, do_sample=True)
                # buffer.append((test_sample, r0, r1))

            beam_search_responses = self.generator.chat_batch(
                questions, do_sample=False
            )
            sampled_responses = self.generator.chat_batch(questions, do_sample=True)

            buffer = list(zip(qa_gt_pairs, beam_search_responses, sampled_responses))

            with open(fn_out, "wb") as fw:
                pkl.dump(buffer, fw)

    def generate_scores(self):
        print("Generating scores for test files")
        score_pred_file_paths = []
        for fn in self.file_paths:
            if "/test.bace.instruct.tsv" in fn:
                score_pred_file_paths.append(fn)
            if "/test.bbbp.instruct.tsv" in fn:
                score_pred_file_paths.append(fn)
        for fn in score_pred_file_paths:
            print("Checking file: {}".format(fn))
            basename = os.path.basename(fn)
            fn_out = basename.replace(".txt", "").replace(".tsv", "") + ".score.pkl"
            fn_out = os.path.join(self.output_dir, fn_out)

            if os.path.exists(fn_out):
                print("File {} already exists, skipping.".format(fn_out))
                continue
            else:
                print("File {} does not exist, processing.".format(fn_out))

            with open(fn, "r", encoding="utf8") as fr:
                all_lines = [e.strip() for e in fr]

            questions = []
            for idx, test_sample in tqdm(enumerate(all_lines), total=len(all_lines)):
                question = test_sample.split("\t")[0].strip()
                questions.append(question)
            probs = self.generator.extract_batch_first_token_prob(questions)

            with open(fn_out, "wb") as fw:
                pkl.dump(probs, fw)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for inference")
    parser.add_argument("--mixtral_path", type=str, required=True, help="")
    parser.add_argument("--nlm_local_path", type=str, required=True, help="")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory path"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory path"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = NLMMoEVLLMInferencer(
        args.input_dir,
        args.output_dir,
        args.mixtral_path,
        args.nlm_local_path,
    )
    inferencer.generate_scores()
    inferencer.generate_responses()


if __name__ == "__main__":
    main()
