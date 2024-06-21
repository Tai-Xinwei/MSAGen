# -*- coding: utf-8 -*-
import os
import pickle as pkl
import numpy as np
import csv
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import argparse


RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Small molecule evaluator for NLM')

    parser.add_argument(
        '--results_dir',
        type=str,
        help='the folder containing the inference pickle results'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='the folder containing the input txt & tsv data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='the folder to save inference results'
    )
    parser.add_argument(
        '--bbbp_pkl',
        type=str,
        help='pickle file name of blood-brain barrier prediction results'
    )
    parser.add_argument(
        '--herg_pkl',
        type=str,
        help='pickle file name of human Ether-à-go-go-Related Gene (hERG) prediction results'
    )
    parser.add_argument(
        '--i2s_i_pkl',
        type=str,
        help='pickle file name of i2s_i prediction results'
    )
    parser.add_argument(
        '--i2s_s_txt',
        type=str,
        help='text file name of i2s_s input data'
    )
    parser.add_argument(
        '--s2i_i_txt',
        type=str,
        help='text file name of s2i_i input data'
    )
    parser.add_argument(
        '--s2i_s_pkl',
        type=str,
        help='pickle file name of s2i_s prediction results'
    )
    parser.add_argument(
        '--desc2mol_pkl',
        type=str,
        help='pickle file name of desc2mol prediction results'
    )
    parser.add_argument(
        '--molinstruct_pkl',
        type=str,
        help='pickle file name of molinstruct prediction results'
    )
    parser.add_argument(
        '--mol2desc_pkl',
        type=str,
        help='pickle file name of mol2desc prediction results'
    )
    parser.add_argument(
        '--bace_pkl',
        type=str,
        help='pickle file name of BACE prediction results'
    )
    parser.add_argument(
        '--bace_tsv',
        type=str,
        help='tsv file name of BACE instruction input data'
    )

    args = parser.parse_args()
    return args


class NLMMoleculeEvaluator():
    def __init__(
            self, results_dir, input_dir, output_dir,
            bbbp_pkl, herg_pkl, i2s_s_txt, i2s_i_pkl,
            s2i_i_txt, s2i_s_pkl, desc2mol_pkl,
            molinstruct_pkl, mol2desc_pkl, bace_pkl, bace_tsv):
        self.results_dir = results_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # e.g. 'test.bbbp.instruct.tsv.run2.step12064.response.pkl'
        self.bbbp_pkl = bbbp_pkl
        # e.g. 'test.hERG.run2.step12064.response.pkl'
        self.herg_pkl = herg_pkl
        self.i2s_s_txt = i2s_s_txt
        self.i2s_i_pkl = i2s_i_pkl
        self.s2i_i_txt = s2i_i_txt
        self.s2i_s_pkl = s2i_s_pkl
        self.desc2mol_pkl = desc2mol_pkl
        self.molinstruct_pkl = molinstruct_pkl
        self.mol2desc_pkl = mol2desc_pkl
        self.bace_pkl = bace_pkl
        self.bace_tsv = bace_tsv
        self.results = {}

    def store_result(self, key, value):
        self.results[key] = value

    def write_results_to_csv(self, csv_file_path):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the headers
            writer.writerow(['Metric', 'Value'])
            # Write the metrics and values
            for key, value in self.results.items():
                writer.writerow([key, value])

    def eval_bbbp(self):
        bbbp_response = os.path.join(
            self.results_dir, self.bbbp_pkl)
        if not os.path.exists(bbbp_response):
            print('BBBP response file does not exist \n')
            return
        fr = open(bbbp_response, 'rb')
        bbbp_records = pkl.load(fr)
        fr.close()
        use_beam = False
        if use_beam:
            gidx = 1
        else:
            gidx = 2

        correct_sample, correct_sample_vote, total = 0, 0, 0
        positive, negative = 0, 0
        predict_positive, predict_negative = 0, 0

        for r in bbbp_records:
            segs = r[0].strip().split('\t')
            label = segs[1].strip().lower()
            if label == 'yes':
                positive += 1
            elif label == 'no':
                negative += 1
            else:
                continue

            predict = r[gidx][0].strip().lower()
            if predict == 'yes':
                predict_positive += 1
            elif predict == 'no':
                predict_negative += 1

            if label == predict:
                correct_sample += 1

            S = [e.strip().lower() for e in r[gidx][:3]]
            if S.count('yes') == S.count('no'):
                predict = r[2][0].strip().lower()
            elif S.count('yes') > S.count('no'):
                predict = 'yes'
            else:
                predict = 'no'

            if label == predict:
                correct_sample_vote += 1

            total += 1

        self.store_result('BBBP Accuracy', correct_sample / total)
        self.store_result('BBBP Correct Predictions', correct_sample)
        self.store_result('BBBP Voting Accuracy', correct_sample_vote / total)
        self.store_result('BBBP Correct Voting Predictions', correct_sample_vote)
        self.store_result('BBBP Positive Instances', positive)
        self.store_result('BBBP Negative Instances', negative)
        self.store_result('BBBP Positive Predictions', predict_positive)
        self.store_result('BBBP Negative Predictions', predict_negative)
        self.store_result('BBBP Total Instances', total)

        print('### Performing evaluation on BBBP dataset ###')
        print('Accuracy:', correct_sample / total)
        print('Correct Predictions:', correct_sample)
        print('Voting Accuracy:', correct_sample_vote / total)
        print('Correct Voting Predictions:', correct_sample_vote)
        print('Positive Instances:', positive)
        print('Negative Instances:', negative)
        print('Positive Predictions:', predict_positive)
        print('Negative Predictions:', predict_negative)
        print('Total Instances:', total)
        print('')

    def eval_herg(self):
        hERG_response = os.path.join(
            self.results_dir, self.herg_pkl)
        if not os.path.exists(hERG_response):
            print('hERG response file does not exist \n')
            return
        fr = open(hERG_response, 'rb')
        hERG_records = pkl.load(fr)
        fr.close()
        use_beam = True
        if use_beam:
            gidx = 1
        else:
            gidx = 2


        predict_list, label_list = [], []

        for r in hERG_records:
            segs = r[0].strip().split('\t')
            label = segs[1].strip()
            if '>' in label:
                label_list.append(label)
            elif label == 'not active':
                label_list.append(label)
            else:
                label_list.append(float(label))

            segs = [e.strip() for e in r[gidx]]
            if segs[0] == 'not active':
                predict_list.append(segs[0])
            elif '>' in segs[0]:
                predict_list.append(segs[0])
            else:
                try:
                    predict_list.append(float(segs[0]))
                except:
                    predict_list.append('illegal')

        Y, Yhat = [], []
        for (y, yhat) in zip(label_list, predict_list):
            if isinstance(y, str) or isinstance(yhat, str):
                continue
            Y.append(y)
            Yhat.append(yhat)

        self.store_result('hERG PearsonR', pearsonr(Y, Yhat))
        self.store_result('hERG Total Number of Instances', len(predict_list))
        self.store_result('hERG Total Number of Vaild Predictions', len(Y))
        print('### Performing evaluation on hERG dataset ###')
        print('pearsonr(Y, Yhat):', pearsonr(Y, Yhat))
        print('len(predict_list):', len(predict_list))
        print('len(Y):', len(Y))
        print('len(hERG_records):', len(hERG_records))
        print('')

    def eval_i2s_i(self):
        i2s_s_txt = os.path.join(self.input_dir, self.i2s_s_txt)
        i2s_i_response = os.path.join(self.results_dir, self.i2s_i_pkl)
        if not os.path.exists(i2s_i_response):
            print('i2s_i response file does not exist \n')
            return
        if not os.path.exists(i2s_s_txt):
            print('i2s_s txt file does not exist \n')
            return
        with open(i2s_i_response, 'rb') as fr:
            i2s_i_records = pkl.load(fr)

        use_beam = True
        gidx = 1 if use_beam else 2

        predicted_smiles = []
        for r in i2s_i_records:
            S = r[gidx]
            flag = False
            for s in S:
                s = s.strip().replace('<mol>', '').replace('</mol>', '')
                s = s.replace(' ', '')
                s = s.replace('<m>', '')
                m = Chem.MolFromSmiles(s)
                if m:
                    s2 = Chem.MolToSmiles(m)
                    predicted_smiles.append(s2)
                    flag = True
                    break
            if not flag:
                predicted_smiles.append('error')

        with open(i2s_s_txt, 'r') as fr:
            ref_smiles = [line.strip() for line in fr]

        # Convert reference SMILES to canonical SMILES for comparison
        ref_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
            if smiles != 'error' else 'error' for smiles in ref_smiles]

        correct, total = 0, 0
        for r, p in zip(ref_smiles, predicted_smiles):
            if r == p:
                correct += 1
            total += 1

        # If total is zero, avoid division by zero in accuracy calculation
        accuracy = correct / total if total > 0 else 0

        self.store_result('I2S_I Total Number of Instances', total)
        self.store_result('I2S_I Correct Predictions', correct)
        self.store_result('I2S_I Accuracy', accuracy)
        print('### Performing evaluation on i2s_i dataset ###')
        print('Total instances:', total)
        print('Correct predictions:', correct)
        print('Accuracy:', accuracy)
        print('')

    def eval_s2i_s(self):
        s2i_i_txt_path = os.path.join(self.input_dir, self.s2i_i_txt)
        s2i_s_response_path = os.path.join(self.results_dir, self.s2i_s_pkl)
        if not os.path.exists(s2i_s_response_path):
            print('s2i_s response file does not exist \n')
            return
        if not os.path.exists(s2i_i_txt_path):
            print('s2i_i txt file does not exist \n')
            return
        with open(s2i_s_response_path, 'rb') as fr:
            s2i_s_records = pkl.load(fr)

        use_beam = True
        gidx = 1 if use_beam else 2

        predicted_iupac = []
        ref_iupac = []

        for r in s2i_s_records:
            S = r[gidx]
            predicted_iupac.append(S[0].strip())

        with open(s2i_i_txt_path, 'r') as fr:
            ref_iupac = [line.strip() for line in fr]

        correct, total = 0, 0
        for r, p in zip(ref_iupac, predicted_iupac):
            if r == p:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        self.store_result('S2I_S Total Number of Instances', total)
        self.store_result('S2I_S Correct Predictions', correct)
        self.store_result('S2I_S Accuracy', accuracy)
        print('### Performing evaluation on s2i_i dataset ###')
        print('Total instances:', total)
        print('Correct predictions:', correct)
        print('Accuracy:', accuracy)
        print('')

    def eval_desc2mol(self):
        desc2mol_response_path = os.path.join(
            self.results_dir, self.desc2mol_pkl)
        if not os.path.exists(desc2mol_response_path):
            print('desc2mol response file does not exist \n')
            return
        with open(desc2mol_response_path, 'rb') as fr:
            desc2mol_records = pkl.load(fr)

        use_beam = True
        gidx = 1 if use_beam else 2

        predicted_smiles = []
        ref_smiles = []
        similarity = []

        for r in desc2mol_records:
            refsmi = r[0].split('\t')[-1].replace('<mol>', '').replace('</mol>', '')
            m = Chem.MolFromSmiles(refsmi)
            refsmi = Chem.MolToSmiles(m)
            ref_smiles.append(refsmi)
            S = r[gidx]
            s = S[0].strip().replace('<mol>', '').replace('</mol>', '')
            s = s.replace(' ', '')
            s = s.replace('<m>', '')
            m = Chem.MolFromSmiles(s)
            if m:
                s2 = Chem.MolToSmiles(m)
                predicted_smiles.append(s2)
            else:
                predicted_smiles.append('error')

        exact, total = 0, 0
        for (r, p) in zip(ref_smiles, predicted_smiles):
            if r == p:
                exact += 1

            mr = Chem.MolFromSmiles(r)
            mp = Chem.MolFromSmiles(p)
            if mr is None or mp is None:
                similarity.append(0)
                continue
            fp1 = AllChem.GetMorganFingerprint(mr, 2)
            fp2 = AllChem.GetMorganFingerprint(mp, 2)
            s = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarity.append(s)
            total += 1

        accuracy = exact / total if total > 0 else 0
        mean_similarity = np.mean(similarity)

        self.store_result('Desc2Mol Total Number of Instances', total)
        self.store_result('Desc2Mol Number of Exact Matches', exact)
        self.store_result('Desc2Mol Accuracy', accuracy)
        self.store_result('Desc2Mol Mean Tanimoto Similarity', mean_similarity)
        print('### Performing evaluation on desc2mol dataset ###')
        print('Total instances:', total)
        print('Exact matches:', exact)
        print('Accuracy:', accuracy)
        print('Mean Tanimoto Similarity:', mean_similarity)
        print('')

    def eval_molinstruct(self):
        molinstruct_response = os.path.join(self.results_dir, self.molinstruct_pkl)
        if not os.path.exists(molinstruct_response):
            print('molinstruct response file does not exist \n')
            return
        with open(molinstruct_response, 'rb') as fr:
            molinstruct_records = pkl.load(fr)

        reagent = []
        forward = []
        backward = []

        for r in molinstruct_records:
            query = r[0].split('\t')[0]
            if '<product>' in query and '<reactants>' in query:
                reagent.append(r)
            elif '<product>' in query:
                backward.append(r)
            elif '<reactants>' in query:
                forward.append(r)

        def clean_generated_smiles(smi):
            s = smi.replace('<mol>', '').replace('</mol>', '')
            s = s.replace('<product>', '').replace('</product>', '')
            s = s.replace('<reactants>', '').replace('</reactants>', '')
            s = s.replace(' ', '')
            s = s.replace('<m>', '')
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            return Chem.MolToSmiles(m)

        def get_acc(r):
            ans = clean_generated_smiles(r[0].split('\t')[1].strip())
            beam_results = clean_generated_smiles(r[1][0])
            random_results = clean_generated_smiles(r[2][0])
            flag_random, flag_beam = False, False
            if random_results is not None:
                flag_random = random_results == ans
            if beam_results is not None:
                flag_beam = beam_results == ans
            return flag_random, flag_beam

        correct_random, correct_beam, total = 0, 0, 0

        for r in reagent:
            flag_random, flag_beam = get_acc(r)
            if flag_random:
                correct_random += 1
            if flag_beam:
                correct_beam += 1
            total += 1

        self.store_result('MolInstruct Total Number of Instances', total)
        self.store_result('MolInstruct Beam Accuracy', correct_beam / total)
        self.store_result('MolInstruct Random Accuracy', correct_random / total)
        print('### Performing evaluation on MolInstruct dataset ###')
        print('Beam accuracy:', correct_beam / total)
        print('Random accuracy:', correct_random / total)
        print('Total instances:', total)
        print('')

    def eval_mol2desc(self):
        mol2desc_response_path = os.path.join(self.results_dir, self.mol2desc_pkl)
        if not os.path.exists(mol2desc_response_path):
            print('mol2desc response file does not exist \n')
            return
        with open(mol2desc_response_path, 'rb') as fr:
            mol2desc_records = pkl.load(fr)

        ref, beam, sample = [], [], []
        for r in mol2desc_records:
            ref.append([r[0].split('\t')[-1].strip()])
            beam.append(r[1][0])
            sample.append(r[2][0])

        bleu2_beam = corpus_bleu(ref, beam, weights=(.5, .5))
        bleu4_beam = corpus_bleu(ref, beam, weights=(.25, .25, .25, .25))
        bleu2_sample = corpus_bleu(ref, sample, weights=(.5, .5))
        bleu4_sample = corpus_bleu(ref, sample, weights=(.25, .25, .25, .25))

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores_beam = [scorer.score(out, gt[0]) for gt, out in zip(ref, beam)]

        self.store_result('Mol2Desc Beam BLEU-2', bleu2_beam)
        self.store_result('Mol2Desc Beam BLEU-4', bleu4_beam)
        self.store_result('Mol2Desc Sample BLEU-2', bleu2_sample)
        self.store_result('Mol2Desc Sample BLEU-4', bleu4_sample)
        self.store_result(
            'Mol2Desc Beam ROUGE-1',
            np.mean([rs['rouge1'].fmeasure for rs in rouge_scores_beam]))
        self.store_result(
            'Mol2Desc Beam ROUGE-2',
            np.mean([rs['rouge2'].fmeasure for rs in rouge_scores_beam]))
        self.store_result(
            'Mol2Desc Beam ROUGE-L',
            np.mean([rs['rougeL'].fmeasure for rs in rouge_scores_beam]))
        self.store_result(
            'Mol2Desc Total Number of Instances',
            len(mol2desc_records))
        print('### Performing evaluation on mol2desc dataset ###')
        print('Beam BLEU-2 score:', bleu2_beam)
        print('Beam BLEU-4 score:', bleu4_beam)
        print('Sample BLEU-2 score:', bleu2_sample)
        print('Sample BLEU-4 score:', bleu4_sample)
        print('Beam ROUGE scores:')
        rouge_1_beam = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores_beam])
        rouge_2_beam = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores_beam])
        rouge_l_beam = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores_beam])
        print('rouge1:', rouge_1_beam)
        print('rouge2:', rouge_2_beam)
        print('rougeL:', rouge_l_beam)
        print('')

    def eval_bace(self):
        bace_response_path = os.path.join(
            self.results_dir, self.bace_pkl)
        if not os.path.exists(bace_response_path):
            print('bace response file does not exist \n')
            return
        with open(bace_response_path, 'rb') as fr:
            bace_records = pkl.load(fr)
        bace_tsv_path = os.path.join(
            self.input_dir, self.bace_tsv)
        test_pairs = []
        y_label = []
        n_pos = 0
        n_neg = 0
        with open(bace_tsv_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                partitions = line.split("\t")
                assert len(partitions) == 2
                assert partitions[1].strip() == "Yes" or partitions[1].strip() == "No"
                test_pairs.append(partitions[0])
                if partitions[1].strip() == "Yes":
                    y_label.append(1)
                    n_pos = n_pos + 1
                else:
                    y_label.append(0)
                    n_neg = n_neg + 1
        print('### Performing evaluation on BACE dataset ###')
        print("Number of positive instances:", n_pos)
        print("Number of negative instances:", n_neg)

        # For AUROC calculation
        y_pred = []
        # test_positive_pairs
        yes_pred = 0
        no_pred = 0
        illeg_pred = 0

        for i, data_tuple in enumerate(bace_records):
            response_text = data_tuple[1][0]
            if response_text.startswith("Yes"):
                yes_pred += 1
                y_pred.append(1)
            elif response_text.startswith("No"):
                no_pred += 1
                y_pred.append(0)
            else:
                illeg_pred += 1
                y_pred.append(0)

        self.store_result('BACE Total Number of Instances', len(bace_records))
        self.store_result('BACE Number of Positive Instances', n_pos)
        self.store_result('BACE Number of Negative Instances', n_neg)
        self.store_result('BACE Number of Positive Predictions', yes_pred)
        self.store_result('BACE Number of Negative Predictions', no_pred)
        self.store_result('BACE Number of Illegal Predictions', illeg_pred)

        print("yes_pred: ", yes_pred)
        print("no_pred: ", no_pred)
        print("illeg_pred: ", illeg_pred)
        print("total_pred: ", yes_pred + no_pred + illeg_pred)

        assert len(y_pred) == len(y_label)
        print("Accuracy:", sum(
            [1 if y1 == y2 else 0 for y1, y2 in zip(y_label, y_pred)]) / len(y_label))
        roc_auc = roc_auc_score(y_label, y_pred)
        print("AUROC:", roc_auc)

        self.store_result('BACE Accuracy',sum(
            [1 if y1 == y2 else 0 for y1, y2 in zip(y_label, y_pred)]) / len(y_label))
        self.store_result('BACE AUROC', roc_auc)

        roc_png_path = os.path.join(self.output_dir, 'bace_roc.png')

        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        # Plot ROC curve
        plt.figure()
        lw = 2  # Line width
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # Save the figure as a PNG file
        plt.savefig(roc_png_path)


def main():
    args = parse_args()
    evaluator = NLMMoleculeEvaluator(
        results_dir=args.results_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        bbbp_pkl=args.bbbp_pkl,
        herg_pkl=args.herg_pkl,
        i2s_s_txt=args.i2s_s_txt,
        i2s_i_pkl=args.i2s_i_pkl,
        s2i_i_txt=args.s2i_i_txt,
        s2i_s_pkl=args.s2i_s_pkl,
        desc2mol_pkl=args.desc2mol_pkl,
        molinstruct_pkl=args.molinstruct_pkl,
        mol2desc_pkl=args.mol2desc_pkl,
        bace_tsv=args.bace_tsv,
        bace_pkl=args.bace_pkl,
    )

    evaluator.eval_bbbp()
    evaluator.eval_herg()
    evaluator.eval_i2s_i()
    evaluator.eval_s2i_s()
    evaluator.eval_desc2mol()
    evaluator.eval_molinstruct()
    evaluator.eval_mol2desc()
    evaluator.eval_bace()

    csv_output_path = os.path.join(evaluator.output_dir, 'evaluation_results.csv')
    evaluator.write_results_to_csv(csv_output_path)


if __name__ == '__main__':
    main()
