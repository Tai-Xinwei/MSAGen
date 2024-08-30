# -*- coding: utf-8 -*-
import argparse
import csv
import json
import math
import os
import pickle as pkl
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rouge_score import rouge_scorer
from scipy.stats import pearsonr, spearmanr
from six import iteritems
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from transformers import BertTokenizerFast

RDLogger.DisableLog("rdApp.*")


class MolScorer:
    """
    Class to score a single molecule based on various metrics.
    Metrics include:
    - Synthetic accessibility score (self.sas)
    - Quantitative estimate of drug-likeness (self.qed)
    - LogP (self.logp)
    - LogP within the range of 0 to 5 (self.good_logp)
    - Topological polar surface area (self.tpsa)
    - Number of hydrogen bond donors (self.hbd)
    - Number of hydrogen bond acceptors (self.hba)
    - Molecular weight (self.molwt)
    - Number of rotatable bonds (self.rotatable_bonds)
    - Ghose filter (self.ghose)
    - Lipinski filter (self.lipinski)
    - Veber filter (self.veber)
    - Muegge filter (self.muegge)
    - Egan filter (self.egan)
    """

    global _fscores
    _fscores = None

    def __init__(self, smiles=None, mol=None):
        if smiles is None and mol is None:
            raise ValueError("Either smiles or mol must be provided")
        if mol is None:
            self.mol = Chem.MolFromSmiles(smiles)
        else:
            self.mol = mol

        if self.mol is None:
            raise ValueError("Invalid SMILES or RDKit molecule")
        self._score()

    @staticmethod
    def compute_sas(mol):
        def _numBridgeheadsAndSpiro(mol, ri=None):
            nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            return nBridgehead, nSpiro

        def _readFragmentScores():
            import gzip

            global _fscores
            _fscores = pkl.load(
                gzip.open(
                    "/home/v-zekunguo/smile/fpscores.pkl.gz"
                )  # Path to the fpscores file
            )
            outDict = {}
            for i in _fscores:
                for j in range(1, len(i)):
                    outDict[i[j]] = float(i[0])
            _fscores = outDict

        def _calculateScore(m):
            if _fscores is None:
                _readFragmentScores()

            # fragment score
            fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
            fps = fp.GetNonzeroElements()
            score1 = 0.0
            nf = 0

            for bitId, v in iteritems(fps):
                nf += v
                sfp = bitId
                score1 += _fscores.get(sfp, -4) * v
            score1 /= nf

            # features score
            nAtoms = m.GetNumAtoms()
            nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
            ri = m.GetRingInfo()
            nBridgeheads, nSpiro = _numBridgeheadsAndSpiro(m, ri)
            nMacrocycles = 0
            for x in ri.AtomRings():
                if len(x) > 8:
                    nMacrocycles += 1
            sizePenalty = nAtoms**1.005 - nAtoms
            stereoPenalty = math.log10(nChiralCenters + 1)
            spiroPenalty = math.log10(nSpiro + 1)
            bridgePenalty = math.log10(nBridgeheads + 1)
            macrocyclePenalty = 0.0
            # ---------------------------------------
            # This differs from the paper, which defines:
            #  macrocyclePenalty = math.log10(nMacrocycles+1)
            # This form generates better results when 2 or more macrocycles are present
            if nMacrocycles > 0:
                macrocyclePenalty = math.log10(2)

            score2 = (
                0.0
                - sizePenalty
                - stereoPenalty
                - spiroPenalty
                - bridgePenalty
                - macrocyclePenalty
            )

            # correction for the fingerprint density
            # not in the original publication, added in version 1.1
            # to make highly symmetrical molecules easier to synthetise
            score3 = 0.0
            if nAtoms > len(fps):
                score3 = math.log(float(nAtoms) / len(fps)) * 0.5

            sascore = score1 + score2 + score3
            # need to transform "raw" value into scale between 1 and 10
            min = -4.0
            max = 2.5
            sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
            # smooth the 10-end
            if sascore > 8.0:
                sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
            if sascore > 10.0:
                sascore = 10.0
            elif sascore < 1.0:
                sascore = 1.0
            return sascore

        sas = _calculateScore(mol)
        sas_norm = round((10 - sas) / 9, 2)
        return sas_norm

    @staticmethod
    def obey_lipinski(mol):
        mol = deepcopy(mol)
        Chem.SanitizeMol(mol)
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        rule_4 = (logp := Crippen.MolLogP(mol) >= -2) & (logp <= 5)
        return (
            np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4]], dtype=float) >= 3
        )

    @staticmethod
    def obey_ghose(mol):
        mol = deepcopy(mol)
        Chem.SanitizeMol(mol)
        molwt = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        molar_refractivity = Descriptors.MolMR(mol)
        atom_count = mol.GetNumHeavyAtoms()
        return (
            (160 <= molwt <= 480)
            and (0.4 <= logp <= 5.6)
            and (40 <= molar_refractivity <= 130)
            and (20 <= atom_count <= 70)
        )

    @staticmethod
    def obey_veber(mol):
        mol = deepcopy(mol)
        Chem.SanitizeMol(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        return tpsa <= 140 and rotatable_bonds <= 10

    @staticmethod
    def obey_muegge(mol):
        mol = deepcopy(mol)
        Chem.SanitizeMol(mol)
        molwt = Descriptors.ExactMolWt(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        rings = mol.GetRingInfo().NumRings()
        atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        carbon = atomic_nums.count(6)
        hetero = len(atomic_nums) - carbon - atomic_nums.count(1)
        tpsa = Descriptors.TPSA(mol)
        xlogp = Crippen.MolLogP(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        return (
            (200 <= molwt <= 450)
            and (hba <= 10)
            and (hbd <= 5)
            and (rings <= 7)
            and (carbon > 4)
            and (hetero > 1)
            and (tpsa <= 150)
            and (-2 <= xlogp <= 5)
            and (rotatable_bonds <= 15)
        )

    @staticmethod
    def obey_egan(mol):
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        return logp <= 5.88 and tpsa <= 131.6 and (logp * tpsa) <= 570

    def _score(self):
        self.sas = self.compute_sas(self.mol)
        self.qed = QED.qed(self.mol)
        self.logp = Crippen.MolLogP(self.mol)
        self.good_logp = 0 <= self.logp <= 5
        self.tpsa = Descriptors.TPSA(self.mol)
        self.hbd = Lipinski.NumHDonors(self.mol)
        self.hba = Lipinski.NumHAcceptors(self.mol)
        self.molwt = Descriptors.ExactMolWt(self.mol)
        self.rotatable_bonds = Lipinski.NumRotatableBonds(self.mol)
        self.ghose = self.obey_ghose(self.mol)
        self.lipinski = self.obey_lipinski(self.mol)
        self.veber = self.obey_veber(self.mol)
        self.muegge = self.obey_muegge(self.mol)
        self.egan = self.obey_egan(self.mol)


class MolsScorer:
    def __init__(self, smiles_list):
        self.scores = defaultdict(list)
        self.smiles_list = smiles_list
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self._score()

    @staticmethod
    def compute_diversity(mols):
        def similarity_fp(fp1, fp2):
            if fp1 is None or fp2 is None:
                return 0.0
            return DataStructs.TanimotoSimilarity(fp1, fp2)

        if not mols:
            return None
        if len(mols) == 1:
            return 0.0
        div = 0.0
        tot = 0
        mols_fps = [
            Chem.RDKFingerprint(mol) if mol is not None else None for mol in mols
        ]
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                div += 1 - similarity_fp(mols_fps[i], mols_fps[j])
                tot += 1
        return div / tot

    def _score(self):
        for i, mol in enumerate(self.mols):
            if mol is None:
                self.scores[i] = None
                print(
                    f"Invalid SMILES at index {i}: {self.smiles_list[i]}, skipping..."
                )
                continue
            try:
                self.scores[i] = MolScorer(mol=mol)
            except Exception as e:
                print(f"error {e}")
                # self.scores[i] = None
        self.diversity = self.compute_diversity(self.mols)
        if self.diversity is None:
            print(self.mols)

    @property
    def qed(self):
        return [
            score.qed if (score is not None and hasattr(score, "qed")) else None
            for score in self.scores.values()
        ]

    @property
    def sas(self):
        return [
            score.sas if (score is not None and hasattr(score, "sas")) else None
            for score in self.scores.values()
        ]

    @property
    def logp(self):
        return [
            score.logp if (score is not None and hasattr(score, "logp")) else None
            for score in self.scores.values()
        ]

    @property
    def good_logp(self):
        return [
            (
                score.good_logp
                if (score is not None and hasattr(score, "good_logp"))
                else None
            )
            for score in self.scores.values()
        ]

    @property
    def tpsa(self):
        return [
            score.tpsa if (score is not None and hasattr(score, "tpsa")) else None
            for score in self.scores.values()
        ]

    @property
    def hbd(self):
        return [
            score.hbd if (score is not None and hasattr(score, "hbd")) else None
            for score in self.scores.values()
        ]

    @property
    def hba(self):
        return [
            score.hba if (score is not None and hasattr(score, "hba")) else None
            for score in self.scores.values()
        ]

    @property
    def molwt(self):
        return [
            score.molwt if (score is not None and hasattr(score, "molwt")) else None
            for score in self.scores.values()
        ]

    @property
    def rotatable_bonds(self):
        return [
            (
                score.rotatable_bonds
                if (score is not None and hasattr(score, "rotatable_bonds"))
                else None
            )
            for score in self.scores.values()
        ]

    @property
    def ghose(self):
        return [
            score.ghose if (score is not None and hasattr(score, "ghose")) else None
            for score in self.scores.values()
        ]

    @property
    def lipinski(self):
        return [
            (
                score.lipinski
                if (score is not None and hasattr(score, "lipinski"))
                else None
            )
            for score in self.scores.values()
        ]

    @property
    def veber(self):
        return [
            score.veber if (score is not None and hasattr(score, "veber")) else None
            for score in self.scores.values()
        ]

    @property
    def muegge(self):
        return [
            score.muegge if (score is not None and hasattr(score, "muegge")) else None
            for score in self.scores.values()
        ]

    @property
    def egan(self):
        return [
            score.egan if (score is not None and hasattr(score, "egan")) else None
            for score in self.scores.values()
        ]

    def to_dict(self):
        return {
            i: {k: getattr(score, k) for k in dir(score) if not k.startswith("_")}
            for i, score in self.scores.items()
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Small molecule evaluator for NLM")

    parser.add_argument(
        "--results_dir",
        type=str,
        help="the folder containing the inference pickle results",
    )
    parser.add_argument(
        "--input_dir", type=str, help="the folder containing the input txt & tsv data"
    )
    parser.add_argument(
        "--output_dir", type=str, help="the folder to save inference results"
    )
    parser.add_argument(
        "--bbbp_pkl",
        type=str,
        help="pickle file name of blood-brain barrier prediction results",
    )
    parser.add_argument(
        "--bbbp_score_pkl",
        type=str,
        help="pickle file name of blood-brain barrier prediction scores",
    )
    parser.add_argument(
        "--herg_pkl",
        type=str,
        help="pickle file name of human Ether-à-go-go-Related Gene (hERG) prediction results",
    )
    parser.add_argument(
        "--i2s_i_pkl", type=str, help="pickle file name of i2s_i prediction results"
    )
    parser.add_argument(
        "--i2s_s_txt", type=str, help="text file name of i2s_s input data"
    )
    parser.add_argument(
        "--s2i_i_txt", type=str, help="text file name of s2i_i input data"
    )
    parser.add_argument(
        "--s2i_s_pkl", type=str, help="pickle file name of s2i_s prediction results"
    )
    parser.add_argument(
        "--desc2mol_pkl",
        type=str,
        help="pickle file name of desc2mol prediction results",
    )
    parser.add_argument(
        "--molinstruct_pkl",
        type=str,
        help="pickle file name of molinstruct prediction results",
    )
    parser.add_argument(
        "--mol2desc_pkl",
        type=str,
        help="pickle file name of mol2desc prediction results",
    )
    parser.add_argument(
        "--bace_pkl", type=str, help="pickle file name of BACE prediction results"
    )
    parser.add_argument(
        "--bace_tsv", type=str, help="tsv file name of BACE instruction input data"
    )
    parser.add_argument(
        "--bace_score_pkl", type=str, help="pickle file name of BACE prediction scores"
    )
    parser.add_argument(
        "--regress_pkl", type=str, help="pickle file name of regress task,split by ','"
    )
    parser.add_argument(
        "--class_pkl", type=str, help="pickle file name of class task,split by ','"
    )
    parser.add_argument(
        "--retro_pkl", type=str, help="pickle file name of retro task,split by ','"
    )
    parser.add_argument(
        "--absolute_correct_pkl",
        type=str,
        help="pickle file name of absolute correct task,split by ','",
    )
    parser.add_argument(
        "--target_to_drug_pkl",
        type=str,
        help="pickle file name of target to drug task,split by ','",
    )
    parser.add_argument(
        "--antibody_pkl",
        type=str,
        help="pickle file name of antibody design task,split by ','",
    )
    parser.add_argument(
        "--drug_assist_folder", type=str, help="drug assist meatedata folder"
    )
    parser.add_argument(
        "--drug_assist_pkl",
        type=str,
        help="pickle file name of drug_assist_pkl task,split by ','",
    )
    parser.add_argument(
        "--grna_filter_pkl",
        type=str,
        help="pickle file name of grna_filter_pkl task,split by ','",
    )
    parser.add_argument(
        "--result_pkl", type=str, help="pickle file name of result,split by ','"
    )
    parser.add_argument(
        "--merge_result",
        action="store_true",
    )
    parser.add_argument("--save_excel_path", type=str, help="save excel path ")
    parser.add_argument("--ckpt_name", type=str, help="ckpt_name")
    parser.add_argument(
        "--gen_cyp_pkl",
        type=str,
        help="Generate small molecules about bbbp, pkl file from wangyue. split by ','",
    )

    parser.add_argument(
        "--base_gen_cyp_path",
        type=str,
        help="Generate small molecules about cyp base file path.",
    )
    parser.add_argument(
        "--gen_bbbp_pkl",
        type=str,
        help="Generate small molecules about bbbp, pkl file from wangyue.",
    )
    parser.add_argument("--protein2desc_pkl", type=str, help="protein2desc_pkl")
    args = parser.parse_args()
    return args


class NLMMoleculeEvaluator:
    def __init__(self, args):
        self.class_pkl = args.class_pkl
        self.results_dir = args.results_dir
        self.regress_pkl = args.regress_pkl
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.bbbp_pkl = args.bbbp_pkl
        self.bbbp_score_pkl = args.bbbp_score_pkl
        self.herg_pkl = args.herg_pkl
        self.i2s_s_txt = args.i2s_s_txt
        self.i2s_i_pkl = args.i2s_i_pkl
        self.s2i_i_txt = args.s2i_i_txt
        self.s2i_s_pkl = args.s2i_s_pkl
        self.desc2mol_pkl = args.desc2mol_pkl
        self.molinstruct_pkl = args.molinstruct_pkl
        self.mol2desc_pkl = args.mol2desc_pkl
        self.bace_pkl = args.bace_pkl
        self.bace_score_pkl = args.bace_score_pkl
        self.bace_tsv = args.bace_tsv
        self.args = args
        self.results = {}
        self.results_infer_pkl = {}

    def write_results_to_pkl(self, pkl_file_path):
        with open(pkl_file_path, "wb") as file:
            pkl.dump([self.results, self.results_infer_pkl], file)

    def eval_bbbp(self):
        bbbp_response = os.path.join(self.results_dir, self.bbbp_pkl)
        if not os.path.exists(bbbp_response):
            print("BBBP response file does not exist \n")
            return
        fr = open(bbbp_response, "rb")
        bbbp_records = pkl.load(fr)
        fr.close()
        bbbp_score = os.path.join(self.results_dir, self.bbbp_score_pkl)
        if not os.path.exists(bbbp_score):
            print("BBBP score file does not exist \n")
            return
        fr = open(bbbp_score, "rb")
        pred_scores = pkl.load(fr)
        fr.close()

        use_beam = False
        if use_beam:
            gidx = 1
        else:
            gidx = 2

        correct_sample, correct_sample_vote, total = 0, 0, 0
        positive, negative = 0, 0
        predict_positive, predict_negative = 0, 0

        gt_labels = []
        for r in bbbp_records:
            segs = r[0].strip().split("\t")
            label = segs[1].strip().lower()
            if label == "yes":
                positive += 1
                gt_labels.append(1)
            elif label == "no":
                negative += 1
                gt_labels.append(0)
            else:
                gt_labels.append(0)
                continue

            predict = r[gidx][0].strip().lower()
            if "yes" in predict:
                predict_positive += 1
            elif "no" in predict:
                predict_negative += 1

            if label in predict:
                correct_sample += 1

            # S = [e.strip().lower() for e in r[gidx][:3]]

            S = []
            for e in r[gidx][:3]:
                if "yes" in e.strip().lower():
                    S.append("yes")
                elif "no" in e.strip().lower():
                    S.append("no")
            if S.count("yes") == S.count("no"):
                predict = r[2][0].strip().lower()
            elif S.count("yes") > S.count("no"):
                predict = "yes"
            else:
                predict = "no"

            if label == predict:
                correct_sample_vote += 1

            total += 1

        assert len(pred_scores) == len(gt_labels)
        roc_auc = roc_auc_score(gt_labels, pred_scores)

        roc_png_path = os.path.join(self.output_dir, "bbbp_roc.png")
        fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
        # Plot ROC curve
        plt.figure()
        lw = 2  # Line width
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("BBBP Instruction Task: Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        # Save the figure as a PNG file
        plt.savefig(roc_png_path)

        print("### Performing evaluation on BBBP dataset ###")
        print("Accuracy:", correct_sample / total)
        print("Correct Predictions:", correct_sample)
        print("Voting Accuracy:", correct_sample_vote / total)
        print("Correct Voting Predictions:", correct_sample_vote)
        print("AUROC:", roc_auc)
        print("Positive Instances:", positive)
        print("Negative Instances:", negative)
        print("Positive Predictions:", predict_positive)
        print("Negative Predictions:", predict_negative)
        print("Total Instances:", total)
        print("")
        self.results["BBBP"] = {}
        self.results["BBBP"]["Accuracy"] = round(correct_sample / total, 3)
        self.results["BBBP"]["Voting Accuracy"] = round(correct_sample_vote / total, 3)
        self.results["BBBP"]["AUROC"] = round(roc_auc, 3)
        self.results_infer_pkl["BBBP"] = {}
        self.results_infer_pkl["BBBP"]["infer result path"] = bbbp_response

    def eval_herg(self):
        hERG_response = os.path.join(self.results_dir, self.herg_pkl)
        if not os.path.exists(hERG_response):
            print("hERG response file does not exist \n")
            return
        fr = open(hERG_response, "rb")
        hERG_records = pkl.load(fr)
        fr.close()
        use_beam = True
        if use_beam:
            gidx = 1
        else:
            gidx = 2

        predict_list, label_list = [], []

        for r in hERG_records:
            segs = r[0].strip().split("\t")
            label = segs[1].strip()
            if ">" in label:
                label_list.append(label)
            elif label == "not active":
                label_list.append(label)
            else:
                label_list.append(float(label))

            segs = [e.strip() for e in r[gidx]]
            if segs[0] == "not active":
                predict_list.append(segs[0])
            elif ">" in segs[0]:
                predict_list.append(segs[0])
            else:
                try:
                    predict_list.append(float(segs[0]))
                except:
                    predict_list.append("illegal")

        Y, Yhat = [], []
        for y, yhat in zip(label_list, predict_list):
            if isinstance(y, str) or isinstance(yhat, str):
                continue
            Y.append(y)
            Yhat.append(yhat)

        if len(Y) == 0:
            pearson_r_value = 0
            pearson_r_p_value = 0
        else:
            pearson_r_value, pearson_r_p_value = pearsonr(Y, Yhat)

        # print("### Performing evaluation on hERG dataset ###")
        # print("pearsonr(Y, Yhat):", pearsonr(Y, Yhat))
        # print("len(predict_list):", len(predict_list))
        # print("len(Y):", len(Y))
        # print("len(hERG_records):", len(hERG_records))
        # print("")

        print("### Performing evaluation on hERG dataset ###")
        print("pearsonr(Y, Yhat):", (pearson_r_value, pearson_r_p_value))
        print("len(predict_list):", len(predict_list))
        print("len(Y):", len(Y))
        print("len(hERG_records):", len(hERG_records))
        print("")

    def eval_i2s_i(
        self,
    ):
        def extract_iupac(line):
            if line.startswith("The SMILES of IUPAC name ") and line.endswith(" is"):
                iupac = line[25:-3]
            elif line.startswith(
                "The SMILES notation for the IUPAC name "
            ) and line.endswith(" is"):
                iupac = line[39:-3]
            elif line.startswith("In SMILES code, the IUPAC ") and line.endswith(
                " is denoted as"
            ):
                iupac = line[26:-14]
            elif line.startswith("The SMILES representation of the ") and line.endswith(
                " is"
            ):
                iupac = line[33:-3]
            else:
                print(line)
                raise ValueError("Invalid prompt")
            return iupac

        i2s_s_txt = os.path.join(
            self.input_dir, "iupac_smiles_translation/test.raw.i2s_s.txt"
        )
        i2s_i_txt = os.path.join(
            self.input_dir, "iupac_smiles_translation/test.raw.i2s_i.txt"
        )
        new_i2s_i_txt = os.path.join(
            self.input_dir, "iupac_smiles_translation/test.new.i2s_i.txt"
        )

        i_list = []
        s_list = []
        new_i_list = []
        with open(i2s_i_txt, "r") as f:
            for line in f.readlines():
                i_list.append(extract_iupac(line.strip()))
        with open(new_i2s_i_txt, "r") as f:
            for line in f.readlines():
                new_i_list.append(line.strip())
        with open(i2s_s_txt, "r") as f:
            for line in f.readlines():
                s_list.append(line.strip())
        new_old_i = {}
        for i, s in zip(new_i_list, i_list):
            new_old_i[i] = s
        i_s_dic = {}
        for i, s in zip(i_list, s_list):
            i_s_dic[i] = s

        i2s_i_response = os.path.join(self.results_dir, self.i2s_i_pkl)
        if not os.path.exists(i2s_i_response):
            print("i2s_i response file does not exist \n")
            return
        if not os.path.exists(i2s_s_txt):
            print("i2s_s txt file does not exist \n")
            return
        with open(i2s_i_response, "rb") as fr:
            i2s_i_records = pkl.load(fr)

        use_beam = True
        gidx = 1 if use_beam else 2

        correct, total = 0, 0
        for r in i2s_i_records:
            S = r[gidx]
            total += 1
            for s in S:
                s = s.strip().replace("<mol>", "").replace("</mol>", "")
                s = s.replace(" ", "")
                s = s.replace("<m>", "")
                m = Chem.MolFromSmiles(s)
                if m:
                    s2 = Chem.MolToSmiles(m)
                    if i_s_dic[new_old_i[r[0].strip()]] == s2:
                        correct += 1
                        break

        # If total is zero, avoid division by zero in accuracy calculation
        accuracy = correct / total if total > 0 else 0

        print("### Performing evaluation on i2s_i dataset ###")
        print("Total instances:", total)
        print("Correct predictions:", correct)
        print("Accuracy:", accuracy)
        print("")
        self.results["I2S_I"] = {}
        self.results["I2S_I"]["Accuracy"] = round(accuracy, 3)
        self.results_infer_pkl["I2S_I"] = {}
        self.results_infer_pkl["I2S_I"]["infer result path"] = i2s_i_response

    def eval_s2i_s(self):
        s2i_s_txt = os.path.join(
            self.input_dir, "iupac_smiles_translation/test.raw.s2i_s.txt"
        )
        s2i_i_txt = os.path.join(
            self.input_dir, "iupac_smiles_translation/test.raw.s2i_i.txt"
        )

        i_list = []
        s_list = []
        with open(s2i_s_txt, "r") as f:
            for line in f.readlines():
                s_list.append(line.strip().split("<mol>")[-1].split("</mol>")[0])

        with open(s2i_i_txt, "r") as f:
            for line in f.readlines():
                i_list.append(line.strip())

        s_i_dic = {}
        for s, i in zip(s_list, i_list):
            s_i_dic[s] = i
        i2s_i_response = os.path.join(self.results_dir, self.s2i_s_pkl)
        if not os.path.exists(i2s_i_response):
            print("i2s_i response file does not exist \n")
            return
        if not os.path.exists(s2i_s_txt):
            print("s2i_s txt file does not exist \n")
            return
        with open(i2s_i_response, "rb") as fr:
            i2s_i_records = pkl.load(fr)

        use_beam = True
        gidx = 1 if use_beam else 2

        correct, total = 0, 0
        for r in i2s_i_records:
            S = r[gidx]
            total += 1
            for s in S:
                if (
                    s_i_dic[r[0].strip().split("<mol>")[-1].split("</mol>")[0]]
                    == s.strip()
                ):
                    correct += 1
                    break

        accuracy = correct / total if total > 0 else 0

        print("### Performing evaluation on s2i_i dataset ###")
        print("Total instances:", total)
        print("Correct predictions:", correct)
        print("Accuracy:", accuracy)
        print("")
        self.results["S2I_S"] = {}
        self.results["S2I_S"]["Accuracy"] = round(accuracy, 3)
        self.results_infer_pkl["S2I_S"] = {}
        self.results_infer_pkl["S2I_S"]["infer result path"] = i2s_i_response

    def eval_desc2mol(self):
        desc2mol_response_path = os.path.join(self.results_dir, self.desc2mol_pkl)
        if not os.path.exists(desc2mol_response_path):
            print("desc2mol response file does not exist \n")
            return
        with open(desc2mol_response_path, "rb") as fr:
            desc2mol_records = pkl.load(fr)

        use_beam = True
        gidx = 1 if use_beam else 2

        predicted_smiles = []
        ref_smiles = []
        similarity = []

        for r in desc2mol_records:
            refsmi = r[0].split("\t")[-1].replace("<mol>", "").replace("</mol>", "")
            m = Chem.MolFromSmiles(refsmi)
            refsmi = Chem.MolToSmiles(m)
            ref_smiles.append(refsmi)
            S = r[gidx]
            s = S[0].strip().replace("<mol>", "").replace("</mol>", "")
            s = s.replace(" ", "")
            s = s.replace("<m>", "")
            m = Chem.MolFromSmiles(s)
            if m:
                s2 = Chem.MolToSmiles(m)
                predicted_smiles.append(s2)
            else:
                predicted_smiles.append("error")

        exact, total = 0, 0
        for r, p in zip(ref_smiles, predicted_smiles):
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

        print("### Performing evaluation on desc2mol dataset ###")
        print("Total instances:", total)
        print("Exact matches:", exact)
        print("Accuracy:", accuracy)
        print("Mean Tanimoto Similarity:", mean_similarity)
        print("")
        self.results["Desc2Mol"] = {}
        self.results["Desc2Mol"]["Accuracy"] = round(accuracy, 3)
        self.results["Desc2Mol"]["Mean Tanimoto Similarity"] = round(mean_similarity, 3)
        self.results_infer_pkl["Desc2Mol"] = {}
        self.results_infer_pkl["Desc2Mol"]["infer result path"] = desc2mol_response_path

    def eval_molinstruct(self):
        molinstruct_response = os.path.join(self.results_dir, self.molinstruct_pkl)
        if not os.path.exists(molinstruct_response):
            print("molinstruct response file does not exist \n")
            return
        with open(molinstruct_response, "rb") as fr:
            molinstruct_records = pkl.load(fr)

        reagent = []
        forward = []
        backward = []

        for r in molinstruct_records:
            query = r[0].split("\t")[0]
            if "<product>" in query and "<reactants>" in query:
                reagent.append(r)
            elif "<product>" in query:
                backward.append(r)
            elif "<reactants>" in query:
                forward.append(r)

        def clean_generated_smiles(smi):
            s = smi.replace("<mol>", "").replace("</mol>", "")
            s = s.replace("<product>", "").replace("</product>", "")
            s = s.replace("<reactants>", "").replace("</reactants>", "")
            s = s.replace(" ", "")
            s = s.replace("<m>", "")
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            return Chem.MolToSmiles(m)

        def get_acc(r):
            ans = clean_generated_smiles(r[0].split("\t")[1].strip())
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

        print("### Performing evaluation on MolInstruct dataset ###")
        print("Beam accuracy:", correct_beam / total)
        print("Random accuracy:", correct_random / total)
        print("Total instances:", total)
        print("")
        self.results["MolInstruct"] = {}
        self.results["MolInstruct"]["Beam accuracy"] = round(correct_beam / total, 3)
        self.results["MolInstruct"]["Random accuracy"] = round(
            correct_random / total, 3
        )
        self.results_infer_pkl["MolInstruct"] = {}
        self.results_infer_pkl["MolInstruct"][
            "infer result path"
        ] = molinstruct_response

    def eval_mol2desc(self):
        mol2desc_response_path = os.path.join(self.results_dir, self.mol2desc_pkl)
        if not os.path.exists(mol2desc_response_path):
            print("mol2desc response file does not exist \n")
            return
        with open(mol2desc_response_path, "rb") as fr:
            mol2desc_records = pkl.load(fr)

        ref, beam, sample = [], [], []
        for r in mol2desc_records:
            ref.append([r[0].split("\t")[-1].strip()])
            beam.append(r[1][0])
            sample.append(r[2][0])

        bleu2_beam = corpus_bleu(ref, beam, weights=(0.5, 0.5))
        bleu4_beam = corpus_bleu(ref, beam, weights=(0.25, 0.25, 0.25, 0.25))
        bleu2_sample = corpus_bleu(ref, sample, weights=(0.5, 0.5))
        bleu4_sample = corpus_bleu(ref, sample, weights=(0.25, 0.25, 0.25, 0.25))

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        rouge_scores_beam = [scorer.score(out, gt[0]) for gt, out in zip(ref, beam)]

        print("### Performing evaluation on mol2desc dataset ###")
        print("Beam BLEU-2 score:", bleu2_beam)
        print("Beam BLEU-4 score:", bleu4_beam)
        print("Sample BLEU-2 score:", bleu2_sample)
        print("Sample BLEU-4 score:", bleu4_sample)
        print("Beam ROUGE scores:")
        rouge_1_beam = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores_beam])
        rouge_2_beam = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores_beam])
        rouge_l_beam = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores_beam])
        print("rouge1:", rouge_1_beam)
        print("rouge2:", rouge_2_beam)
        print("rougeL:", rouge_l_beam)
        print("")
        self.results["mol2desc"] = {}
        self.results["mol2desc"]["Beam BLEU-2"] = round(bleu2_beam, 3)
        self.results["mol2desc"]["Beam BLEU-4"] = round(bleu4_beam, 3)
        self.results["mol2desc"]["Sample BLEU-2"] = round(bleu2_sample, 3)
        self.results["mol2desc"]["Sample BLEU-4"] = round(bleu4_sample, 3)
        self.results["mol2desc"]["rouge1"] = round(rouge_1_beam, 3)
        self.results["mol2desc"]["rouge2"] = round(rouge_2_beam, 3)
        self.results["mol2desc"]["rougeL"] = round(rouge_l_beam, 3)
        self.results_infer_pkl["mol2desc"] = {}
        self.results_infer_pkl["mol2desc"]["infer result path"] = mol2desc_response_path

    def eval_bace(self):
        bace_response_path = os.path.join(self.results_dir, self.bace_pkl)
        if not os.path.exists(bace_response_path):
            print("bace response file does not exist \n")
            return
        with open(bace_response_path, "rb") as fr:
            bace_records = pkl.load(fr)
        bace_score = os.path.join(self.results_dir, self.bace_score_pkl)
        if not os.path.exists(bace_score):
            print("BACE score file does not exist \n")
            return
        fr = open(bace_score, "rb")
        pred_scores = pkl.load(fr)
        fr.close()
        bace_tsv_path = os.path.join(self.input_dir, self.bace_tsv)
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
        print("### Performing evaluation on BACE dataset ###")
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

        print("yes_pred: ", yes_pred)
        print("no_pred: ", no_pred)
        print("illeg_pred: ", illeg_pred)
        print("total_pred: ", yes_pred + no_pred + illeg_pred)

        assert len(y_pred) == len(y_label)
        acc = sum([1 if y1 == y2 else 0 for y1, y2 in zip(y_label, y_pred)]) / len(
            y_label
        )
        print(
            "Accuracy:",
            sum([1 if y1 == y2 else 0 for y1, y2 in zip(y_label, y_pred)])
            / len(y_label),
        )
        roc_auc = roc_auc_score(y_label, pred_scores)
        print("AUROC:", roc_auc)

        self.results["BACE"] = {}
        self.results["BACE"]["Accuracy"] = round(acc, 3)
        self.results["BACE"]["AUROC"] = round(roc_auc, 3)
        self.results_infer_pkl["BACE"] = {}
        self.results_infer_pkl["BACE"]["infer result path"] = bace_tsv_path

        roc_png_path = os.path.join(self.output_dir, "bace_roc.png")

        fpr, tpr, thresholds = roc_curve(y_label, pred_scores)
        # Plot ROC curve
        plt.figure()
        lw = 2  # Line width
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("BACE Instruction Task: Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        # Save the figure as a PNG file
        plt.savefig(roc_png_path)

    def eval_regress(self):
        regress_pkl_list = []
        if "," in self.regress_pkl:
            for regress_pkl in self.regress_pkl.split(","):
                regress_pkl_list.append(os.path.join(self.results_dir, regress_pkl))
        else:
            regress_pkl_list = [os.path.join(self.results_dir, self.regress_pkl)]
        for file_name in self.regress_pkl.split(","):
            regress_response_path = os.path.join(self.results_dir, file_name)
            print("")
            if not os.path.exists(regress_response_path):
                print(f"{regress_response_path} file does not exist \n")
                continue

            with open(regress_response_path, "rb") as fr:
                regress_records = pkl.load(fr)
            error_count = 0
            ref_score_list = []
            sample_score_list = []
            beam_score_list = []
            for record in regress_records:
                pattern = r"\b\d+(\.\d+)?\b"
                try:
                    matches = re.findall(pattern, record[1][0].split("\t")[-1])
                    b = float(matches[0])
                    matches = re.findall(pattern, record[2][0].split("\t")[-1])
                    s = float(matches[0])
                    matches = re.findall(pattern, record[0].split("\t")[-1])
                    r = float(matches[0])
                    beam_score_list.append(b)
                    sample_score_list.append(s)
                    ref_score_list.append(r)

                except:
                    error_count += 1
                    continue
            print(f"pkl file:{regress_response_path}")
            print(f"total count:{len(regress_records)}")
            print(f"error count:{error_count}")
            print()
            print(
                f"beam_MAE:{round(mean_absolute_error(ref_score_list,beam_score_list),3)}"
            )
            print(
                f"sample_MAE:{round(mean_absolute_error(ref_score_list,sample_score_list),3)}"
            )
            print(
                f"beam_RMSE:{np.sqrt(mean_squared_error(ref_score_list,beam_score_list))}"
            )
            print(
                f"sample_RMSE:{np.sqrt(mean_squared_error(ref_score_list,sample_score_list))}"
            )
            print(f"beam_pearsonr:{pearsonr(ref_score_list,beam_score_list)}")
            print(f"sample_pearsonr:{pearsonr(ref_score_list,sample_score_list)}")
            print(
                f"beam_spearmanr:{round(spearmanr(ref_score_list,beam_score_list).statistic,3)}"
            )
            print(
                f"sample_spearmanr:{round(spearmanr(ref_score_list,sample_score_list).statistic,3)}"
            )
            if file_name not in self.results:
                self.results[file_name] = {}
                self.results[file_name]["beam_MAE"] = round(
                    mean_absolute_error(ref_score_list, beam_score_list), 3
                )
                self.results[file_name]["sample_MAE"] = round(
                    mean_absolute_error(ref_score_list, sample_score_list), 3
                )
                self.results[file_name]["beam_RMSE"] = round(
                    np.sqrt(mean_squared_error(ref_score_list, beam_score_list)), 3
                )
                self.results[file_name]["sample_RMSE"] = round(
                    np.sqrt(mean_squared_error(ref_score_list, sample_score_list)), 3
                )
                self.results[file_name]["beam_pearsonr"] = round(
                    pearsonr(ref_score_list, beam_score_list).statistic, 3
                )
                self.results[file_name]["sample_pearsonr"] = round(
                    pearsonr(ref_score_list, sample_score_list).statistic, 3
                )
                self.results[file_name]["beam_spearmanr"] = round(
                    spearmanr(ref_score_list, beam_score_list).statistic, 3
                )
                self.results[file_name]["sample_spearmanr"] = round(
                    spearmanr(ref_score_list, sample_score_list).statistic, 3
                )
                self.results_infer_pkl[file_name] = {}
                self.results_infer_pkl[file_name][
                    "infer result path"
                ] = regress_response_path

            else:
                print(f"{file_name} has test.")

    def eval_class_yes_or_no(self):
        class_pkl_list = []
        if "," in self.class_pkl:
            for class_pkl in self.class_pkl.split(","):
                class_pkl_list.append(os.path.join(self.results_dir, class_pkl))
        else:
            class_pkl_list = [os.path.join(self.results_dir, self.class_pkl)]
        for file_name in self.class_pkl.split(","):
            class_response_path = os.path.join(self.results_dir, file_name)
            print("################################")
            if not os.path.exists(class_response_path):
                print(f"{class_response_path} file does not exist \n")
                continue
            with open(class_response_path, "rb") as fr:
                regress_records = pkl.load(fr)
            error_count = 0
            ref_score_list = []
            sample_score_list = []
            beam_score_list = []
            for record in regress_records:
                try:
                    if "yes" in record[0].lower():
                        ref_score_list.append(1)
                    else:
                        ref_score_list.append(0)
                    if "yes" in record[1][0].lower():
                        beam_score_list.append(1)
                    else:
                        beam_score_list.append(0)
                    if "yes" in record[2][0].lower():
                        sample_score_list.append(1)
                    else:
                        sample_score_list.append(0)

                except:
                    error_count += 1
                    continue
            # print(ref_score_list)
            print(f"pkl file:{class_response_path}")
            print(f"total count:{len(regress_records)}")
            print(f"error count:{error_count}")
            beam_accuracy = sum(
                [
                    1 if y1 == y2 else 0
                    for y1, y2 in zip(ref_score_list, beam_score_list)
                ]
            ) / len(ref_score_list)
            print("Beam Accuracy:", round(beam_accuracy, 3))
            sample_accuracy = sum(
                [
                    1 if y1 == y2 else 0
                    for y1, y2 in zip(ref_score_list, sample_score_list)
                ]
            ) / len(ref_score_list)
            print("Sample Accuracy:", round(sample_accuracy, 3))
            beam_report = classification_report(
                ref_score_list, beam_score_list, output_dict=True
            )
            print(
                f"beam macro avg f1-score: {round(beam_report['macro avg']['f1-score'],3)}"
            )
            sample_report = classification_report(
                ref_score_list, sample_score_list, output_dict=True
            )
            print(
                f"sample macro avg f1-score: {round(sample_report['macro avg']['f1-score'],3)}"
            )
            print(
                f"Beam MCC: {round(matthews_corrcoef(ref_score_list, beam_score_list),3)}"
            )
            print(
                f"random MCC: {round(matthews_corrcoef(ref_score_list, sample_score_list),3)}"
            )
            if file_name not in self.results:
                self.results[file_name] = {}
                self.results[file_name]["Beam Accuracy"] = round(beam_accuracy, 3)
                self.results[file_name]["Sample Accuracy"] = round(sample_accuracy, 3)
                self.results[file_name]["Beam macro avg f1-score"] = round(
                    beam_report["macro avg"]["f1-score"], 3
                )
                self.results[file_name]["Sample macro avg f1-score"] = round(
                    sample_report["macro avg"]["f1-score"], 3
                )
                self.results[file_name]["Beam MCC"] = round(
                    matthews_corrcoef(ref_score_list, beam_score_list), 3
                )
                self.results[file_name]["Sample MCC"] = round(
                    matthews_corrcoef(ref_score_list, sample_score_list), 3
                )
                self.results_infer_pkl[file_name] = {}
                self.results_infer_pkl[file_name][
                    "infer result path"
                ] = class_response_path

            else:
                print(f"{file_name} has test.")

    def eval_retro(self):
        def remove_atom_mapping_and_canonicalize(smiles):
            try:
                # 1. Remove atom mapping.
                mol = Chem.MolFromSmiles(smiles)
                for atom in mol.GetAtoms():
                    if atom.HasProp("molAtomMapNumber"):
                        atom.ClearProp("molAtomMapNumber")
                # 2. Canonicalize.
                smiles_wo_atom_mapping = Chem.MolToSmiles(mol)
                mol = Chem.MolFromSmiles(smiles_wo_atom_mapping)
                return Chem.MolToSmiles(mol)
            except:
                return ""

        for file_name in self.args.retro_pkl.split(","):
            file_path = os.path.join(self.args.results_dir, file_name)
            print("######################")
            print(f"Test file paht: {file_path}")
            if not os.path.exists(file_path):
                print(f"{file_path} file does not exist \n")
                continue
            results = pkl.load(open(file_path, "rb"))
            top_1_acc_num = 0
            top_3_acc_num = 0
            illegel_predicted_smiles_num = 0

            for i, result in enumerate(results):
                assert len(result) == 3
                instructon_and_gt = result[0]
                predictions = result[1]
                # print("result:", result)
                assert (
                    len(predictions) == 4
                ), f"{predictions}"  # assume that beam size is 4

                gt_smiles = instructon_and_gt.split("<reactants>")[1].split(
                    "</reactants>"
                )[0]
                product_smiles = instructon_and_gt.split("<product>")[1].split(
                    "</product>"
                )[0]
                canonical_gt_smiles = remove_atom_mapping_and_canonicalize(gt_smiles)
                remove_atom_mapping_and_canonicalize(product_smiles)
                predicted_smiles_list = []
                for prediction in predictions:
                    if "<m>" in prediction:
                        curr_smiles = (
                            prediction.replace("<m>", "")
                            .replace("<reactants>", "")
                            .replace("</reactants>", "")
                            .replace(" ", "")
                        )
                    else:
                        curr_smiles = prediction.replace("<reactants>", "").replace(
                            "</reactants>", ""
                        )
                    curr_smiles = curr_smiles.replace(" ", "")
                    # print("curr_smiles:", curr_smiles)
                    canonical_curr_smiles = remove_atom_mapping_and_canonicalize(
                        curr_smiles
                    )
                    # print("canonical_curr_smiles:", canonical_curr_smiles)
                    if canonical_curr_smiles == "":
                        illegel_predicted_smiles_num += 1
                    else:
                        # predicted_smiles_list.add(canonical_curr_smiles)
                        predicted_smiles_list.append(canonical_curr_smiles)

                # predicted_smiles_list = list(predicted_smiles_list)  # since set is not subscriptable
                # print("canonical_product_smiles:", canonical_product_smiles)
                # print("canonical_gt_smiles:", canonical_gt_smiles)

                if (
                    len(predicted_smiles_list) > 0
                    and canonical_gt_smiles == predicted_smiles_list[0]
                ):
                    top_1_acc_num += 1

                if (
                    len(predicted_smiles_list) > 0
                    and canonical_gt_smiles in predicted_smiles_list[:3]
                ):
                    top_3_acc_num += 1

            print("len:", len(results))
            print(f"Top-1 accuracy: {round(top_1_acc_num / len(results),3)}")
            print(f"Top-3 accuracy: {round(top_3_acc_num / len(results),3)}")
            print(f"Illegel predicted smiles num: {illegel_predicted_smiles_num}")
            if file_name not in self.results:
                self.results[file_name] = {}
                self.results[file_name]["Top-1 accuracy"] = round(
                    top_1_acc_num / len(results), 3
                )
                self.results[file_name]["Top-3 accuracy"] = round(
                    top_3_acc_num / len(results), 3
                )
            else:
                print(f"{file_name} has test.")

    def eval_absolute_correct(self):
        for file_name in self.args.absolute_correct_pkl.split(","):
            file_path = os.path.join(self.args.results_dir, file_name)
            print("######################")
            print(f"Test file paht: {file_path}")
            if not os.path.exists(file_path):
                print(f"{file_path} file does not exist \n")
                continue
            results = pkl.load(open(file_path, "rb"))
            top_1_acc_num = 0
            top_3_acc_num = 0
            ref_result_list = []
            predict_result_list = []
            for i, result in enumerate(results):
                predictions = result[1]
                ref_result = result[0].split("\t")[-1].strip()
                ref_result_list.append(ref_result)
                tmp_list = []
                for prediction in predictions:
                    prediction = (
                        prediction.split("</s>")[0]
                        .split("<|end_of_text|>")[0]
                        .replace("<i>", "")
                        .replace("<m>", "")
                        .replace("<a>", "")
                        .replace("<r>", "")
                        .replace(" ", "")
                    )
                    tmp_list.append(prediction)
                predict_result_list.append(tmp_list)
                if len(ref_result) > 0 and ref_result == tmp_list[0]:
                    top_1_acc_num += 1
                if len(ref_result) > 0 and ref_result in tmp_list[:3]:
                    top_3_acc_num += 1
            print("len:", len(results))
            print(f"Top-1 accuracy: {round(top_1_acc_num / len(results),3)}")
            print(f"Top-3 accuracy: {round(top_3_acc_num / len(results),3)}")
            if file_name not in self.results:
                self.results[file_name] = {}
                self.results[file_name]["Top-1 accuracy"] = round(
                    top_1_acc_num / len(results), 3
                )
                self.results[file_name]["Top-3 accuracy"] = round(
                    top_3_acc_num / len(results), 3
                )
                self.results_infer_pkl[file_name] = {}
                self.results_infer_pkl[file_name]["infer result path"] = file_path

            else:
                print(f"{file_name} has test.")

    def eval_t2d(self):
        def reconstruct_molecule_from_fragments(fragments):
            # Initialize an empty molecule to store the reconstructed molecule
            reconstructed_mol = Chem.RWMol()
            # Combine the fragments into a single molecule
            combined_mol = fragments[0]
            for fragment in fragments[1:]:
                combined_mol = Chem.CombineMols(combined_mol, fragment)

            # Convert the combined molecule to an RWMol
            reconstructed_mol = Chem.RWMol(combined_mol)
            # Find the dummy atoms and their corresponding neighbors

            dummy_atoms = []
            for atom_idx, atom in enumerate(reconstructed_mol.GetAtoms()):
                if atom.GetAtomicNum() == 0:
                    neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
                    dummy_atoms.append((atom_idx, atom.GetAtomMapNum(), neighbors[0]))

            # Sort dummy atoms by atom map number
            dummy_atoms.sort(key=lambda x: x[1])

            # Connect the fragments using the dummy atoms and remove them
            for i in range(0, len(dummy_atoms), 2):
                try:
                    atom_idx1, map_num1, neighbor1 = dummy_atoms[i]
                    atom_idx2, map_num2, neighbor2 = dummy_atoms[i + 1]
                except:
                    return fragments[0]
                if map_num1 == map_num2:
                    # Add a bond between the atoms connected to the dummy atoms
                    reconstructed_mol.AddBond(
                        neighbor1, neighbor2, Chem.rdchem.BondType.SINGLE
                    )

                    # Remove the dummy atoms
                    # reconstructed_mol.RemoveAtom(atom_idx2)
                    # reconstructed_mol.RemoveAtom(atom_idx1)
                    #
            dummy_atoms = [e[0] for e in dummy_atoms]
            dummy_atoms = sorted(dummy_atoms, reverse=True)
            for atomidx in dummy_atoms:
                reconstructed_mol.RemoveAtom(atomidx)
                # reconstructed_mol.RemoveAtom(atom_idx1)

            # Remove atom map numbers from the reconstructed molecule
            for atom in reconstructed_mol.GetAtoms():
                atom.SetAtomMapNum(0)

            return reconstructed_mol

        for file_name in self.args.target_to_drug_pkl.split(","):
            file_path = os.path.join(self.args.results_dir, file_name)
            print("######################")
            print(f"Test file paht: {file_path}")
            if not os.path.exists(file_path):
                print(f"{file_path} file does not exist \n")
                continue
            results = pkl.load(open(file_path, "rb"))
            merge_mol_count = 0
            success_protein_count = 0
            all_mol_list = []
            gen_smi_per_count = len(results[0][-1])
            if "frag" in results[0][0]:
                for item in results:
                    fraga = item[0].split("<fragA>")[1].split("</fragA>")[0]
                    mol_list = []
                    # do sample
                    for frag in item[-1]:
                        try:
                            fragb = frag.split("<fragB>")[1].split("</fragB>")[0]
                            fragb = fragb.replace("<m>", "").replace(" ", "")

                            # print(fraga,fragb)
                            t = reconstruct_molecule_from_fragments(
                                [Chem.MolFromSmiles(fraga), Chem.MolFromSmiles(fragb)]
                            )
                            s = Chem.MolToSmiles(t)
                            # print(s)
                            if "." in s:
                                continue
                            mol = Chem.MolFromSmiles(s)
                            if mol is None:
                                continue
                            mol_list.append(s)
                        except Exception as e:
                            print(e)
                            continue
                    merge_mol_count += len(mol_list)
                    if len(mol_list) > 0:
                        # print('11111111111111111111111111111')
                        success_protein_count += 1
                        all_mol_list.append(mol_list)
            else:
                for item in results:
                    mol_list = []
                    for smi in item[-1]:
                        tmp_smi = (
                            smi.replace("<m>", "")
                            .replace(" ", "")
                            .replace("<mol>", "")
                            .replace("</mol>", "")
                        )
                        mol = Chem.MolFromSmiles(tmp_smi)
                        if mol is None:
                            continue
                        if tmp_smi is not None and tmp_smi != "":
                            mol_list.append(tmp_smi)
                    all_mol_list.append(mol_list)
                merge_mol_count = -1
            all_scores = {}
            i = 0
            for mol_list in all_mol_list:
                all_scores[i] = MolsScorer(mol_list)
                i += 1
            diversity = np.mean([score.diversity for score in all_scores.values()])
            qed = np.mean([np.mean(score.qed) for score in all_scores.values()])
            sas = np.mean([np.mean(score.sas) for score in all_scores.values()])
            logp = np.mean([np.mean(score.logp) for score in all_scores.values()])
            # total_count = sum([len(scores) for scores in all_scores.values()])
            total_count = sum(
                [1 for scores in all_scores.values() for score in scores.logp]
            )
            satisfying_count = sum(
                [
                    1
                    for scores in all_scores.values()
                    for score in scores.logp
                    if 0 < score < 5
                ]
            )
            # total_count = len(data) * 50
            percentage = (satisfying_count / total_count) * 100

            good_logp = np.mean(
                [np.mean(score.good_logp) for score in all_scores.values()]
            )
            tpsa = np.mean([np.mean(score.tpsa) for score in all_scores.values()])
            hbd = np.mean([np.mean(score.hbd) for score in all_scores.values()])
            hba = np.mean([np.mean(score.hba) for score in all_scores.values()])
            molwt = np.mean([np.mean(score.molwt) for score in all_scores.values()])
            rotatable_bonds = np.mean(
                [np.mean(score.rotatable_bonds) for score in all_scores.values()]
            )
            ghose = np.mean([np.mean(score.ghose) for score in all_scores.values()])
            lipinski = np.mean(
                [np.mean(score.lipinski) for score in all_scores.values()]
            )
            veber = np.mean([np.mean(score.veber) for score in all_scores.values()])
            muegge = np.mean([np.mean(score.muegge) for score in all_scores.values()])
            egan = np.mean([np.mean(score.egan) for score in all_scores.values()])
            valid = np.mean([len(x) for x in all_mol_list]) / gen_smi_per_count
            print(f"file path: {file_path}")
            print(f"QED: {qed}")
            print(f"SAS: {sas}")
            print(f"Diversity: {diversity}")
            print(f"logP 0-5 percentage: {percentage}")
            print(f"Lipinski: {lipinski}")
            print(f"logP: {logp}")

            print(f"Good logP: {good_logp}")
            print(f"TPSA: {tpsa}")
            print(f"HBD: {hbd}")
            print(f"HBA: {hba}")
            print(f"MolWT: {molwt}")
            print(f"Rotatable Bonds: {rotatable_bonds}")
            print(f"Ghose: {ghose}")
            print(f"Veber: {veber}")
            print(f"Muegge: {muegge}")
            print(f"Egan: {egan}")
            print(f"validity: {valid}")
            print(f"protein count: {len(results)}")
            print(f"success protein count: {success_protein_count}")
            print(f"merge mol count:{merge_mol_count}")
            print(
                f"merge success ratio:{merge_mol_count/(len(results)*gen_smi_per_count)}"
            )
            if file_name not in self.results:
                self.results[file_name] = {}
                self.results[file_name]["QED"] = round(qed, 3)
                self.results[file_name]["SAS"] = round(sas, 3)
                self.results[file_name]["Diversity"] = round(diversity, 3)
                self.results[file_name]["logP 0-5 percentage"] = round(percentage, 3)
                self.results[file_name]["Lipinski"] = round(lipinski, 3)
                self.results[file_name]["logP"] = round(logp, 3)
                self.results[file_name]["Good logP"] = round(good_logp, 3)
                self.results[file_name]["TPSA"] = round(tpsa, 3)
                self.results[file_name]["HBD"] = round(hbd, 3)
                self.results[file_name]["HBA"] = round(hba, 3)
                self.results[file_name]["MolWT"] = round(molwt, 3)
                self.results[file_name]["Rotatable Bonds"] = round(rotatable_bonds, 3)
                self.results[file_name]["Ghose"] = round(ghose, 3)
                self.results[file_name]["Veber"] = round(veber, 3)
                self.results[file_name]["Muegge"] = round(muegge, 3)
                self.results[file_name]["Egan"] = round(egan, 3)
                self.results[file_name]["Validity"] = round(valid, 3)
                self.results_infer_pkl[file_name] = {}
                self.results_infer_pkl[file_name]["infer result path"] = file_path

                if merge_mol_count > 0:
                    self.results[file_name]["merge success ratio"] = round(
                        merge_mol_count / (len(results) * gen_smi_per_count), 3
                    )
                    self.results[file_name]["success protein ratio"] = round(
                        success_protein_count / len(results), 3
                    )

                # self.results[file_name]['infer result path']=file_path

            else:
                print(f"{file_name} has test.")

    def eval_antibody_design(self):
        def edit_distance(str1, str2):
            len_str1 = len(str1) + 1
            len_str2 = len(str2) + 1

            dp = [[0 for n in range(len_str2)] for m in range(len_str1)]

            for i in range(len_str1):
                dp[i][0] = i
            for j in range(len_str2):
                dp[0][j] = j

            for i in range(1, len_str1):
                for j in range(1, len_str2):
                    if str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            return dp[-1][-1]

        def post_calc(pred, gt):
            # length prediction
            length_diff = np.abs(np.array(float(len(gt) - len(pred))))
            # edit distance
            distance = edit_distance(pred, gt)
            # forced AAR
            if len(pred) < len(gt):
                # add special token
                pred += (len(gt) - len(pred)) * "X"
            match = [int(a == b) for a, b in zip(pred, gt)]
            return match, length_diff, distance

        for file_name in self.args.antibody_pkl.split(","):
            file_path = os.path.join(self.args.results_dir, file_name)
            print("######################")
            print(f"Test file paht: {file_path}")
            if not os.path.exists(file_path):
                print(f"{file_path} file does not exist \n")
                continue
            results = pkl.load(open(file_path, "rb"))
            succ, tot = 0, 0
            length_diffs, distances = [], []
            for line in results:
                gt = (
                    line[0]
                    .split("\t")[1]
                    .replace("<antibody>", "")
                    .replace("</antibody>", "")
                )
                tmp_results = [
                    line[1][0]
                    .replace("<a>", "")
                    .replace("<antibody>", "")
                    .replace("</antibody>", "")
                    .replace(" ", ""),
                    line[1][1]
                    .replace("<a>", "")
                    .replace("<antibody>", "")
                    .replace("</antibody>", "")
                    .replace(" ", ""),
                    line[2][0]
                    .replace("<a>", "")
                    .replace("<antibody>", "")
                    .replace("</antibody>", "")
                    .replace(" ", ""),
                    line[2][1]
                    .replace("<a>", "")
                    .replace("<antibody>", "")
                    .replace("</antibody>", "")
                    .replace(" ", ""),
                ]
                match_list, length_diff_list, distance_list = [], [], []
                sum_match_list = []
                for pred in tmp_results:
                    match, length_diff, distance = post_calc(pred, gt)
                    sum_match_list.append(sum(match))
                    match_list.append(match)
                    length_diff_list.append(length_diff)
                    distance_list.append(distance)
                # choose best match
                index = np.argmax(sum_match_list)
                match = match_list[index]
                length_diff = length_diff_list[index]
                distance = distance_list[index]

                succ += sum(match)
                # tot += len(match)
                tot += len(gt)
                length_diffs.append(length_diff)
                distances.append(distance)

            print(
                f"{file_name} AAR: {succ/tot:.4f}, length diff: {np.mean(length_diffs):.4f}, edit distance: {np.mean(distances):.4f}"
            )
            if file_name not in self.results:
                self.results[file_name] = {}
                self.results[file_name]["AAR"] = round(succ / tot, 3)
                self.results[file_name]["length diff"] = round(np.mean(length_diffs), 3)
                self.results[file_name]["edit distance"] = round(
                    np.mean(distances) / len(results), 3
                )
            else:
                print(f"{file_name} has test.")

    def eval_drug(self):
        def compare_sim(mol1, mol2):
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
            s = DataStructs.FingerprintSimilarity(
                fp1, fp2, metric=DataStructs.TanimotoSimilarity
            )
            return s

        def clean_smiles(line, return_mol=False):
            s = line.replace("<mol>", "")
            s = s.replace("</mol>", "")
            s = s.replace("<m>", "")
            s = s.replace(" ", "")
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            if return_mol:
                return Chem.MolToSmiles(m), m
            return Chem.MolToSmiles(m)

        def evaluate_comparison(
            v1, v2, similarity, similarity_thr=0.6, delta=0, direction="increase"
        ):
            if similarity < similarity_thr:
                return False
            if direction == "increase":
                d = v2 - v1
                return d >= delta
            elif direction == "decrease":
                d = v1 - v2
                return d >= delta
            elif direction == "the same":
                return abs(v1 - v2) < 1e-7

        def evaluate_range(
            v2, similarity, similarity_thr=0.6, range_lower=0, range_upper=100
        ):
            if similarity < similarity_thr:
                return False
            if v2 >= range_lower and v2 <= range_upper:
                return True
            return False

        def evaluator(
            results,
            similarity_thr=0.6,
            delta=0,
            direction="increase",
            range_lower=0,
            range_upper=100,
        ):
            positive = 0
            for v in results:
                if "range" not in direction:
                    if evaluate_comparison(
                        v[0], v[1], v[2], similarity_thr, delta, direction
                    ):
                        positive += 1
                else:
                    if evaluate_range(
                        v[1], v[2], similarity_thr, range_lower, range_upper
                    ):
                        positive += 1
            return positive / len(results)

        property_functions = {
            # 'MW': Chem.rdMolDescriptors.CalcExactMolWt,
            "HBD": Chem.rdMolDescriptors.CalcNumHBD,
            "donor": Chem.rdMolDescriptors.CalcNumHBD,
            # 'HBA': Chem.rdMolDescriptors.CalcNumHBA,
            # 'RotBonds': Chem.rdMolDescriptors.CalcNumRotatableBonds,
            # 'FSP3': Chem.rdMolDescriptors.CalcFractionCSP3,
            # 'TPSA': Chem.rdMolDescriptors.CalcTPSA,
            "logP": Descriptors.MolLogP,
            "QED": QED.qed,
        }
        if not os.path.exists(self.args.drug_assist_folder):
            print(f"{self.args.drug_assist_folder} do not exits!")
            return
        for file_name in self.args.drug_assist_pkl.split(","):
            file_path = os.path.join(self.args.results_dir, file_name)
            print("######################")
            print(f"Test file paht: {file_path}")
            if not os.path.exists(file_path):
                print(f"{file_path} file does not exist \n")
                continue
            optimized_property = None
            for key in property_functions.keys():
                if key.lower() in file_name:
                    optimized_property = key
            if optimized_property is None:
                print(f"File {file_name} test is not supported")
                continue
            original_fn = os.path.join(
                self.args.drug_assist_folder, optimized_property, "test.instruct.tsv"
            )
            meta_fn = os.path.join(
                self.args.drug_assist_folder, optimized_property, "test.metadata.json"
            )
            results = pkl.load(open(file_path, "rb"))
            with open(meta_fn, "r") as fr:
                meta = json.load(fr)
            original_inputs = []
            with open(original_fn, "r", encoding="utf8") as fr:
                for line in fr:
                    original_inputs.append(line.strip())

            input_metainfo_dict = dict((k, v) for (k, v) in zip(original_inputs, meta))
            statistics = {}
            unique_r = set([e["requirement"] for e in meta])
            for r in unique_r:
                statistics[r] = []

            unique_cmpds = set()
            for ele in results:
                k = ele[0]
                srcsmi = input_metainfo_dict[k]["source_smiles"]
                msrc = Chem.MolFromSmiles(srcsmi)
                for e in ele[1]:
                    ret = clean_smiles(e, return_mol=True)
                    if ret is None:
                        continue
                    v2 = property_functions[optimized_property](ret[1])
                    v1 = property_functions[optimized_property](msrc)
                    similarity = compare_sim(msrc, ret[1])
                    r = input_metainfo_dict[k]["requirement"]
                    statistics[r].append((v1, v2, similarity))
                    unique_cmpds.add(ret[0])

            results_dict = {}
            sim_threshold = 0.65

            for k, values in statistics.items():
                if k == "decrease":
                    r = evaluator(
                        values,
                        similarity_thr=sim_threshold,
                        delta=0,
                        direction="decrease",
                    )
                elif k == "increase":
                    r = evaluator(
                        values,
                        similarity_thr=sim_threshold,
                        delta=0,
                        direction="increase",
                    )
                elif k == "the same":
                    r = evaluator(
                        values,
                        similarity_thr=sim_threshold,
                        delta=0,
                        direction="the same",
                    )
                elif "decrease, >=" in k:
                    delta = k.replace("decrease, >=", "").strip()
                    delta = float(delta) - 1e-9
                    r = evaluator(
                        values,
                        similarity_thr=sim_threshold,
                        delta=delta,
                        direction="decrease",
                    )
                elif "increase, >=" in k:
                    delta = k.replace("increase, >=", "").strip()
                    delta = float(delta) - 1e-9
                    r = evaluator(
                        values,
                        similarity_thr=sim_threshold,
                        delta=delta,
                        direction="increase",
                    )
                elif "range" in k:
                    segs = k.split(",")
                    lb = float(segs[1])
                    ub = float(segs[2])
                    r = evaluator(
                        values,
                        similarity_thr=sim_threshold,
                        direction="range",
                        range_lower=lb,
                        range_upper=ub,
                    )
                results_dict[k] = (r, len(values))
            success_count = 0
            total_count = 0
            results = []
            for k, v in results_dict.items():
                results.append([k, "{:.3f}".format(v[0]), v[1]])

            results.sort(key=lambda x: x[0])
            print(results)
            success_count = 0
            total_count = 0
            for item in results:
                success_count += float(item[1]) * item[2]
                total_count += item[2]
            print(f"success rate: {round(success_count/total_count,3)}")

            if file_name not in self.results:
                self.results[file_name] = {}
                for key, value in results_dict.items():
                    print(key, value)
                    self.results[file_name][key] = round(float(value[0]), 3)
                self.results[file_name]["success rate"] = round(
                    success_count / total_count, 3
                )
                self.results_infer_pkl[file_name] = {}
                self.results_infer_pkl[file_name]["infer result path"] = file_path
            else:
                print(f"{file_name} has test.")

    def eval_grna_filter(self):
        for file_name in self.args.grna_filter_pkl.split(","):
            file_path = os.path.join(self.args.results_dir, file_name)
            print("######################")
            print(f"Test file paht: {file_path}")
            if not os.path.exists(file_path):
                print(f"{file_path} file does not exist \n")
                continue
            results = pkl.load(open(file_path, "rb"))
            correct_ngg = 0
            correct_in = 0
            total = 0
            for item in results:
                for rna in item[1]:
                    rna = (
                        rna.replace("<rna>", "")
                        .replace("</rna>", "")
                        .replace("<r>", "")
                        .replace(" ", "")
                    )
                    if rna == "" or rna is None:
                        continue
                    cur_dna = item[0].split("<dna>")[-1].split("</dna>")[0]
                    # (1) len=>[17,24]
                    # (2) belong to dna seq
                    # (3) ending with NGG
                    if (
                        rna in cur_dna
                        and cur_dna.split(rna)
                        and len(rna) <= 24
                        and len(rna) >= 17
                    ):
                        correct_in += 1
                        if cur_dna.split(rna)[1][1:3] == "GG":
                            correct_ngg += 1
                        break
                total += 1
            print(round(correct_in / total, 3))
            print(round(correct_ngg / total, 3))
            if file_name not in self.results:
                self.results[file_name] = {}
                self.results[file_name]["Correct in DNA"] = round(correct_in / total, 3)
                self.results[file_name]["Correct with NGG"] = round(
                    correct_ngg / total, 3
                )
                self.results_infer_pkl[file_name] = {}
                self.results_infer_pkl[file_name]["infer result path"] = file_path
            else:
                print(f"{file_name} has test.")

    def eval_protein2desc(self):
        # mol-instructions protein understanding task
        for file_name in self.args.protein2desc_pkl.split(","):
            protein2desc_response_path = os.path.join(self.results_dir, file_name)
            if not os.path.exists(protein2desc_response_path):
                print(f"{protein2desc_response_path}  file does not exist \n")
                return
            with open(protein2desc_response_path, "rb") as fr:
                protein2desc_records = pkl.load(fr)
            # To keep consistent with the benchmark
            rouge = evaluate.load("rouge")
            ref, beam, sample = [], [], []
            for r in protein2desc_records:
                try:
                    beam.append(r[1][0])
                    sample.append(r[2][0])
                    ref.append([r[0].split("\t")[-1].strip()])
                except:
                    continue
                    # print(r)
            beam_results = rouge.compute(predictions=beam, references=ref)
            sample_results = rouge.compute(predictions=sample, references=ref)
            print("######################")
            print(f"Test file path: {protein2desc_response_path}")
            print("Beam ROUGE scores:")
            print("rouge1:", beam_results["rouge1"])
            print("rouge2:", beam_results["rouge2"])
            print("rougeL:", beam_results["rougeL"])
            print("rougeLsum:", beam_results["rougeLsum"])
            print("")
            print("sample ROUGE scores:")
            print("rouge1:", sample_results["rouge1"])
            print("rouge2:", sample_results["rouge2"])
            print("rougeL:", sample_results["rougeL"])
            print("rougeLsum:", sample_results["rougeLsum"])
            print("")
            self.results[file_name] = {}
            self.results[file_name]["beam rouge1"] = round(beam_results["rouge1"], 3)
            self.results[file_name]["beam rouge2"] = round(beam_results["rouge2"], 3)
            self.results[file_name]["beam rougeL"] = round(beam_results["rougeL"], 3)
            self.results[file_name]["beam rougeLsum"] = round(
                beam_results["rougeLsum"], 3
            )
            self.results[file_name]["sample rouge1"] = round(
                sample_results["rouge1"], 3
            )
            self.results[file_name]["sample rouge2"] = round(
                sample_results["rouge2"], 3
            )
            self.results[file_name]["sample rougeL"] = round(
                sample_results["rougeL"], 3
            )
            self.results[file_name]["sample rougeLsum"] = round(
                sample_results["rougeLsum"], 3
            )
            self.results_infer_pkl[file_name] = {}
            self.results_infer_pkl[file_name][
                "infer result path"
            ] = protein2desc_response_path

    def merge_result_to_excel(self, index=0):
        all_result = {}
        task_name_list = set()
        print("###############")
        print(self.args.result_pkl)
        for file_name in sorted(os.listdir(self.args.result_pkl)):
            if not file_name.endswith("pkl"):
                continue
            file_path = os.path.join(self.args.result_pkl, file_name)
            with open(file_path, "rb") as f:
                data = pkl.load(f)[index]
                all_result[file_name] = data
                task_name_list.update(list(data.keys()))
        task_result = {}
        for task in task_name_list:
            for ckpt_name, result in all_result.items():
                if task in result:
                    if task in task_result:
                        task_result[task]["ckpt_name"].append(ckpt_name)
                        for key, value in result[task].items():
                            task_result[task][key].append(value)
                    else:
                        task_result[task] = {}
                        task_result[task]["ckpt_name"] = [ckpt_name]
                        for key, value in result[task].items():
                            task_result[task][key] = [value]

        if index == 0:
            gen_bbbp = self.merge_bbbp_to_result()
            gen_cpy = self.merge_cyp_to_result()
            bbbp_result = {}
            for bbbp in gen_bbbp.keys():
                if bbbp not in bbbp_result:
                    bbbp_result[bbbp] = {}
                for ckpt_name, all_value in gen_bbbp[bbbp].items():
                    if "ckpt_name" in bbbp_result[bbbp]:
                        bbbp_result[bbbp]["ckpt_name"].append(ckpt_name)
                    else:
                        bbbp_result[bbbp]["ckpt_name"] = [ckpt_name]
                    for key, value in all_value.items():
                        if key in bbbp_result[bbbp]:
                            bbbp_result[bbbp][key].append(value)
                        else:
                            bbbp_result[bbbp][key] = [value]
            cpy_result = {}
            for cpy in gen_cpy.keys():
                if cpy not in cpy_result:
                    cpy_result[cpy] = {}

                for ckpt_name, all_value in gen_cpy[cpy].items():
                    if "ckpt_name" in cpy_result[cpy]:
                        cpy_result[cpy]["ckpt_name"].append(ckpt_name)
                    else:
                        cpy_result[cpy]["ckpt_name"] = [ckpt_name]
                    for key, value in all_value.items():
                        if key in cpy_result[cpy]:
                            cpy_result[cpy][key].append(value)
                        else:
                            cpy_result[cpy][key] = [value]

            task_result.update(bbbp_result)
            task_result.update(cpy_result)

        wb = Workbook()
        ws = wb.active
        ws.title = "CombinedResultSheet"

        current_row = 1
        print(task_result)
        for title, data in task_result.items():
            ws.cell(row=current_row, column=1, value=title)
            current_row += 1
            df = pd.DataFrame(data)

            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(r)
                current_row += 1

            current_row += 1

        save_path = os.path.join(self.args.save_excel_path, f"{index}.all_result.xlsx")
        wb.save(save_path)

    def merge_bbbp_to_result(self):
        with open(self.args.gen_bbbp_pkl, "rb") as f:
            data = pkl.load(f)
        bbbp_task_result = {}
        bbbp_task_list = [
            "decrease_bbbp.task.csv",
            "gen_bbbp.task.csv",
            "increase_bbbp.task.csv",
        ]
        for bbbp_task in bbbp_task_list:
            bbbp_task_result[bbbp_task] = {}
            for ckpt_name in data.keys():
                if bbbp_task in ckpt_name:
                    correct = 0
                    total = 0
                    sample_len = len(data[ckpt_name]) // 2
                    print(f"#######{sample_len}")
                    if "decrease" in bbbp_task:
                        for i in range(sample_len):
                            print(f"response{i+sample_len+1}")
                            for result in data[ckpt_name][f"response{i+sample_len+1}"][
                                "predictions"
                            ]:
                                total += 1
                                if result is None:
                                    continue
                                if "no" in result.lower():
                                    correct += 1
                    else:
                        for i in range(sample_len):
                            for result in data[ckpt_name][f"response{i+sample_len+1}"][
                                "predictions"
                            ]:
                                total += 1
                                if result is None:
                                    continue
                                if "yes" in result.lower():
                                    correct += 1
                    cur_ckpt_name = ckpt_name.split(bbbp_task)[0][:-1] + "_results.pkl"
                    bbbp_task_result[bbbp_task][cur_ckpt_name] = {}
                    bbbp_task_result[bbbp_task][cur_ckpt_name]["Success rate"] = round(
                        correct / (total), 3
                    )
                    bbbp_task_result[bbbp_task][cur_ckpt_name][
                        "sample len"
                    ] = sample_len
        print("##################")
        print(bbbp_task_result)
        return bbbp_task_result

    def merge_cyp_to_result(self):
        cyp_task_result = {}
        for pkl_name in self.args.gen_cyp_pkl.split(","):
            pkl_path = os.path.join(self.args.base_gen_cyp_path, pkl_name)
            if not os.path.exists(pkl_path):
                print(f"{pkl_path} file does not exist \n")
                continue
            with open(pkl_path, "rb") as f:
                data = pkl.load(f)
                print("#########")
                print(pkl_path)
            cyp_task_list = [
                "test.CYP1A2.normal.osmi.task.csv",
                "test.CYP2C19.normal.osmi.task.csv",
                "test.CYP2C9.normal.osmi.task.csv",
                "test.CYP2D6.normal.osmi.task.csv",
                "test.CYP3A4.normal.osmi.task.csv",
            ]
            for cyp_task in cyp_task_list:
                if cyp_task not in cyp_task_result:
                    cyp_task_result[cyp_task] = {}
                for ckpt_name in data.keys():
                    if cyp_task in ckpt_name:
                        ref_score_list = []
                        sample_len = len(data[ckpt_name]) // 2
                        # for score in data[ckpt_name]['input']['predictions']:
                        #     ref_score_list.extend([float(score)]*sample_len)
                        gen_score_list = []
                        error_count = 0
                        for i in range(sample_len):
                            for ref, gen in zip(
                                data[ckpt_name]["input"]["predictions"],
                                data[ckpt_name][f"response{i+sample_len+1}"][
                                    "predictions"
                                ],
                            ):
                                # tmp.append(float(score))
                                try:
                                    gen_score_list.append(float(gen))
                                    ref_score_list.append(float(ref))
                                except:
                                    error_count += 1

                        # reduce_list=ref_score_list-gen_score_list
                        reduce_list = [
                            a - b for a, b in zip(ref_score_list, gen_score_list)
                        ]
                        cur_ckpt_name = (
                            ckpt_name.split(cyp_task)[0][:-1] + "_results.pkl"
                        )
                        count = len([score for score in reduce_list if score > 0])
                        cyp_task_result[cyp_task][cur_ckpt_name] = {}
                        cyp_task_result[cyp_task][cur_ckpt_name][
                            "Success rate"
                        ] = round(count / len(ref_score_list), 3)
                        cyp_task_result[cyp_task][cur_ckpt_name][
                            "Average reduction"
                        ] = round(sum(reduce_list) / len(ref_score_list), 3)

        print("##################")
        print(cyp_task_result)
        return cyp_task_result


def main():
    args = parse_args()
    print(args.regress_pkl)
    evaluator = NLMMoleculeEvaluator(args)
    if args.merge_result:
        # 0 Evaluation Metrics
        # 1 infer result location
        evaluator.merge_result_to_excel(0)
        evaluator.merge_result_to_excel(1)
    else:
        evaluator.eval_bbbp()
        evaluator.eval_herg()
        evaluator.eval_i2s_i()
        evaluator.eval_s2i_s()
        evaluator.eval_desc2mol()
        evaluator.eval_molinstruct()
        evaluator.eval_mol2desc()
        evaluator.eval_bace()
        evaluator.eval_regress()
        evaluator.eval_class_yes_or_no()
        evaluator.eval_absolute_correct()
        evaluator.eval_antibody_design()
        evaluator.eval_grna_filter()
        evaluator.eval_drug()
        evaluator.eval_t2d()
        evaluator.eval_retro()
        evaluator.eval_protein2desc()
        csv_output_path = os.path.join(
            evaluator.output_dir, f"{args.ckpt_name}_results.pkl"
        )
        evaluator.write_results_to_pkl(csv_output_path)


if __name__ == "__main__":
    main()
