#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Union

import lmdb
import pandas as pd
from tqdm import tqdm

from lddt4SinglePair import lddt4SinglePair
from LGA4SinglePair import LGA4SinglePair
from TMscore4SinglePair import TMscore4SinglePair
from utils import bstr2obj


logger = logging.getLogger(__name__)


def calculate_score(predlines: list, natilines: list, residx: set) -> dict:
    """Calculate score between predicted and native structure by TM-score"""

    def _select_residues_by_residx(atomlines: list):
        lines = []
        for line in atomlines:
            if line.startswith("ATOM"):
                resnum = int(line[22:26].strip())
                if resnum in residx:
                    lines.append(line)
        lines.append("TER\n")
        lines.append("END\n")
        return lines

    with (
        tempfile.NamedTemporaryFile() as predpdb,
        tempfile.NamedTemporaryFile() as natipdb,
    ):
        with open(predpdb.name, "w") as fp:
            fp.writelines(_select_residues_by_residx(predlines))
        with open(natipdb.name, "w") as fp:
            fp.writelines(_select_residues_by_residx(natilines))
        score = TMscore4SinglePair(predpdb.name, natipdb.name)
        score["LDDT"] = lddt4SinglePair(predpdb.name, natipdb.name)["LDDT"]
        return score



def evaluate_predicted_structure(
    metadata: Mapping[str, Union[list, str]],
    preddir: str,
    max_model_num: int = 1,
) -> pd.DataFrame:
    scores = []
    for target in tqdm(metadata["keys"]):
        taridx = metadata["keys"].index(target)
        # calculate score for each domain
        for domstr, domlen, domgroup in metadata["domains"][taridx]:
            try:
                residx = set()
                domseg = domstr.split(":")[1]
                for seg in domseg.split(","):
                    start, finish = [int(_) for _ in seg.split("-")]
                    residx.update(range(start, finish + 1))
                assert domlen == len(residx), f"domain length!={domlen}"
            except Exception as e:
                logger.error(f"Domain {domstr} parsing error, {e}")
                continue

            # process score for each predicted model
            for num in range(1, max_model_num + 1):
                score = {
                    "Target": domstr.split(":")[0],
                    "Length": domlen,
                    "Group": domgroup,
                    "Type": metadata["types"][taridx],
                    "ModelIndex": num,
                }
                try:
                    pdb_file = os.path.join(preddir, f"{target}-{num}.pdb")
                    with open(pdb_file, "r") as fp:
                        predlines = fp.readlines()
                    assert predlines, f" wrong predicted file {pdb_file}"
                    plddts = [float(_[60:66].strip())
                              for _ in predlines if _.startswith('ATOM')]
                    score["pLDDT"] = sum(plddts) / len(plddts)
                    natilines = metadata["pdbs"][taridx]
                    score.update(calculate_score(predlines, natilines, residx))
                except Exception as e:
                    logger.error(f"Failed to evaluate {domstr}, {e}.")
                    continue
                scores.append(score)
    df = pd.DataFrame(scores)
    return df


def calculate_average_score(df: pd.DataFrame) -> pd.DataFrame:
    CATEGORY = {
        "CAMEO  Full": ["Easy", "Medium", "Hard"],
        "CASP14 Full": ["MultiDom"],
        "CASP15 Full": ["MultiDom"],
        "CAMEO  Easy": ["Easy"],
        "CAMEO  M+H": ["Medium", "Hard"],
        "CAMEO  Medi": ["Medium"],
        "CAMEO  Hard": ["Hard"],
        "CASP14 TBM-easy": ["TBM-easy"],
        "CASP14 TBM-hard": ["TBM-hard"],
        "CASP14 FM": ["FM/TBM", "FM"],
        "CASP15 TBM-Easy": ["TBM-easy"],
        "CASP15 TBM-Hard": ["TBM-hard"],
        "CASP15 FM": ["FM/TBM", "FM"],
        "(   0, 384]": ["Easy", "Medium", "Hard", "MultiDom"],
        "( 384, 512]": ["Easy", "Medium", "Hard", "MultiDom"],
        "( 512,8192]": ["Easy", "Medium", "Hard", "MultiDom"],
    }
    # group score by target
    records = []
    for target, gdf in df.groupby("Target"):
        record = {
            "Target": target,
            "Length": gdf["Length"].iloc[0],
            "Group": gdf["Group"].iloc[0],
            "Type": gdf["Type"].iloc[0],
        }
        numtop1 = gdf["ModelIndex"][gdf["pLDDT"].idxmax()]
        for col in ["TMscore", "RMSD", "GDT_TS", "LDDT"]:
            top1score = gdf[gdf["ModelIndex"] == numtop1][col].iloc[0]
            maxscore = float("-inf")
            for num in gdf["ModelIndex"].to_list():
                score = gdf[gdf["ModelIndex"] == num][col].iloc[0]
                record[f"Model{num}_{col}"] = score
                maxscore = max(maxscore, score)
            record[f"ModelTop1_{col}"] = top1score
            record[f"ModelMax_{col}"] = maxscore
        records.append(record)
    newdf = pd.DataFrame(records)
    # calculate average score for each category
    scores = []
    for key, groups in CATEGORY.items():
        if key.startswith("("):
            low, high = [int(_) for _ in key.strip('(]').split(",")]
            subdf = newdf[(newdf["Length"] > low) &
                          (newdf["Length"] <= high) &
                          newdf["Group"].isin(groups)]
        else:
            cate_type = key.split()[0]
            subdf = newdf[(newdf["Type"] == cate_type) &
                          newdf["Group"].isin(groups)]
        scores.append(
            {
                "CatAndGroup": key,
                "Number": len(subdf),
                "Rnd1TMscore": subdf["Model1_TMscore"].mean() * 100,
                "Top1TMscore": subdf["ModelTop1_TMscore"].mean() * 100,
                "BestTMscore": subdf["ModelMax_TMscore"].mean() * 100,
                "Rnd1LDDT": subdf["Model1_LDDT"].mean() * 100,
                "Top1LDDT": subdf["ModelTop1_LDDT"].mean() * 100,
                "BestLDDT": subdf["ModelMax_LDDT"].mean() * 100,
            }
        )
    # calculate average score for dataframe
    meandf = pd.DataFrame(scores).set_index("CatAndGroup")
    return newdf, meandf


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit(f"Usage: {sys.argv[0]} <proteintest_lmdb> <prediction_directory> <max_model_num> <global_step>")
    inplmdb, preddir, max_model_num, global_step = sys.argv[1:5]
    max_model_num = int(max_model_num)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    logger.info(f"Loading metadata from {inplmdb}.")
    with lmdb.open(inplmdb, readonly=True).begin(write=False) as txn:
        metadata = bstr2obj(txn.get("__metadata__".encode()))
    logger.info(f"Metadata contains {len(metadata['keys'])} keys, pdbs, ...")

    logger.info(f"TMscore between predicted.pdb and native.pdb {preddir}. ")
    df = evaluate_predicted_structure(metadata, preddir, max_model_num)
    print(df)

    logger.info(f"Average TMscore for different categories.")
    df.to_csv(Path(preddir) / f"../Score4EachModel_{global_step}.csv")
    newdf, meandf = calculate_average_score(df)
    newdf.to_csv(Path(preddir) / f"../Score4Target_{global_step}.csv")
    print(newdf)
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(meandf)
