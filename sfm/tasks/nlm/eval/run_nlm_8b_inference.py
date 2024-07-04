# -*- coding: utf-8 -*-
import argparse
import os
import pickle as pkl
from glob import glob

from nlm_8b_inference_module import NLMGenerator
from tqdm import tqdm


class NLMInferencer:
    def __init__(self, ckpt_home, tokenizer_home, input_dir, output_dir, use_gen_data):
        self.generator = NLMGenerator(ckpt_home, tokenizer_home)
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not use_gen_data:
            self.file_paths = [
                os.path.join(input_dir, "test.instruct.predict_bbbp.tsv"),
                os.path.join(input_dir, "test.instruct.predict_bace.tsv"),
                os.path.join(input_dir, "test.uspto50k.retro.osmi.tsv"),
                os.path.join(input_dir, "test.uspto50k.reaction.osmi.tsv"),
                os.path.join(input_dir, "test.molinstruct.reagent_prediction.tsv"),
                os.path.join(input_dir, "test.desc2mol.tsv"),
                os.path.join(input_dir, "test.mol2desc.tsv"),
                os.path.join(input_dir, "iupac_smiles_translation/test.new.i2s_i.txt"),
                os.path.join(input_dir, "iupac_smiles_translation/test.new.s2i_s.txt"),
            ]
        else:
            folder = r"/home/t-kaiyuangao/workspace/proj_logs/nlm_inst/Drug_Assist/"
            self.file_paths = [
                folder + "QED/QED_test.instruct.tsv",
                folder + "solubility/solubility_test.instruct.tsv",
                folder + "donor/donor_test.instruct.tsv",
                folder + "logP/logP_test.instruct.tsv",
            ]
            folder = r"/home/t-kaiyuangao/ml-container/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617.test/"
            self.file_paths = [
                folder + "test.CYP1A2.normal.osmi.tsv",
                folder + "test.CYP2C19.normal.osmi.tsv",
                folder + "test.CYP2C9.normal.osmi.tsv",
                folder + "test.CYP2D6.normal.osmi.tsv",
                folder + "test.CYP3A4.normal.osmi.tsv",
                folder + "test.hERG.optim.v2.tsv",
                folder + "test.instruct.gen_bbbp.tsv",
                folder + "test.instruct.increase_bbbp.tsv",
                folder + "test.instruct.decrease_bbbp.tsv",
                folder + "test.instruct.gen_bace.tsv",
                folder + "test.instruct.increase_bace.tsv",
                folder + "test.instruct.decrease_bace.tsv",
            ]

    def generate_responses(self):
        for fn in self.file_paths:
            print("Checking file: {}".format(fn))
            basename = os.path.basename(fn)
            fn_out = basename + ".response.pkl"
            fn_out = os.path.join(self.output_dir, fn_out)

            if os.path.exists(fn_out):
                print("File {} already exists, skipping.".format(fn_out))
                continue
            else:
                print("File {} does not exist, processing.".format(fn_out))

            with open(fn, "r", encoding="utf8") as fr:
                all_lines = [e.strip() for e in fr]

            buffer = []
            for idx, test_sample in tqdm(enumerate(all_lines), total=len(all_lines)):
                q = test_sample.split("\t")[0].strip()
                r0 = self.generator.chat(q, do_sample=False)
                r1 = self.generator.chat(q, do_sample=True)
                buffer.append((test_sample, r0, r1))

            with open(fn_out, "wb") as fw:
                pkl.dump(buffer, fw)

    def generate_scores(self):
        print("Generating scores for test files")
        score_pred_file_paths = []
        for fn in self.file_paths:
            if "/test.instruct.predict_bbbp.tsv" in fn:
                score_pred_file_paths.append(fn)
            if "/test.instruct.predict_bace.tsv" in fn:
                score_pred_file_paths.append(fn)
        for fn in score_pred_file_paths:
            print("Checking file: {}".format(fn))
            basename = os.path.basename(fn)
            fn_out = basename + ".score.pkl"
            fn_out = os.path.join(self.output_dir, fn_out)

            if os.path.exists(fn_out):
                print("File {} already exists, skipping.".format(fn_out))
                continue
            else:
                print("File {} does not exist, processing.".format(fn_out))

            buffer = self.generator.extract_first_token_prob(fn)

            with open(fn_out, "wb") as fw:
                pkl.dump(buffer, fw)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for inference")
    parser.add_argument(
        "--ckpt_home", type=str, required=True, help="Checkpoint directory path"
    )
    parser.add_argument(
        "--tokenizer_home", type=str, required=True, help="Tokenizer directory path"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory path"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory path"
    )
    parser.add_argument(
        "--use_gen_data", action="store_true", help="Generative datasets", default=False
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = NLMInferencer(
        args.ckpt_home,
        args.tokenizer_home,
        args.input_dir,
        args.output_dir,
        args.use_gen_data,
    )
    inferencer.generate_scores()
    inferencer.generate_responses()


if __name__ == "__main__":
    main()
