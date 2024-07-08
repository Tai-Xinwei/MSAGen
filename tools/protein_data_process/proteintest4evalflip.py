#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import tempfile
from pathlib import Path

import lmdb
import pandas as pd
import parse
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr

sys.path.append(str(Path.cwd().parent.parent))
from sfm.logging import logger
from sfm.tasks.psm.evaluate_psm_protein import calculate_score


def flip(inppdb: str, outpdb: str):
    pattern_str = ("ATOM  {atomidx:>5d}  CA  {resname} {chain}{resnumb:>4d}    "
                   "{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  \n")
    with open(inppdb, "r") as in_file, open(outpdb, "w") as out_file:
        for line in in_file:
            if line.startswith("ATOM"):
                results = parse.parse(pattern_str, line)
                results = {
                    'atomidx': results['atomidx'],
                    'resname': results['resname'],
                    'chain': results['chain'],
                    'resnumb': results['resnumb'],
                    'x': results['x'],
                    'y': results['y'],
                    'z': results['z']
                    }
                results['y'] = -results['y']
                new_line = pattern_str.format_map(results)
                out_file.write(new_line)
            else:
                out_file.write(line)


def evaluate_predicted_structure(metadata: dict, preddir: str):
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
                    pdb_file = Path(preddir) / f"{target}-{num}.pdb"
                    with open(pdb_file, "r") as fp:
                        predlines = fp.readlines()
                    natilines = metadata["pdbs"][taridx]
                    score.update(calculate_score(predlines, natilines, residx))

                    with tempfile.NamedTemporaryFile() as flippdb:
                        flip(pdb_file, flippdb.name)
                        with open(flippdb.name, "r") as fp:
                            fliplines = fp.readlines()
                        sco = calculate_score(fliplines, natilines, residx)
                        score['TMscore_flip'] = sco['TMscore']
                        score['RMSD_flip'] = sco['RMSD']
                        score['GDT_TS_flip'] = sco['GDT_TS']
                        score['LDDT_flip'] = sco['LDDT']
                except Exception as e:
                    logger.error(f"Failed to evaluate {domstr}, {e}.")
                    continue
                scores.append(score)
    df = pd.DataFrame(scores)
    return df


def calculate_average_score(df: pd.DataFrame) -> pd.DataFrame:
    CATEGORY = {
        "CAMEO  Easy": ["Easy"],
        "CAMEO  Medi": ["Medium", "Hard"],
        "CASP14 Full": ["MultiDom"],
        "CASP14 Easy": ["TBM-easy", "TBM-hard"],
        "CASP14 Hard": ["FM/TBM", "FM"],
        "CASP15 Full": ["MultiDom"],
        "CASP15 Easy": ["TBM-easy", "TBM-hard"],
        "CASP15 Hard": ["FM/TBM", "FM"],
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
        max_model_num = gdf["ModelIndex"].max()
        for col in ["TMscore", "LDDT"]:
            maxscore = float("-inf")
            for num in range(1, max_model_num + 1):
                score = gdf[gdf["ModelIndex"] == num][col].iloc[0]
                record[f"Model{num}_{col}"] = score
                maxscore = max(maxscore, score)
                score = gdf[gdf["ModelIndex"] == num][f"{col}_flip"].iloc[0]
                record[f"Model{num}_{col}_flip"] = score
                maxscore = max(maxscore, score)
            record[f"ModelMax_{col}"] = maxscore
        records.append(record)
    newdf = pd.DataFrame(records)
    # calculate average score for each category
    scores = []
    for key, groups in CATEGORY.items():
        _type = key.split()[0]
        subdf = newdf[(newdf["Type"] == _type) & newdf["Group"].isin(groups)]
        scores.append(
            {
                "CatAndGroup": key,
                "Number": len(subdf),
                "Top1TMscore": subdf["Model1_TMscore"].mean() * 100,
                "MaxOrigFlipTMscore": subdf["ModelMax_TMscore"].mean() * 100,
                "Top1LDDT": subdf["Model1_LDDT"].mean() * 100,
                "MaxOrigFlipLDDT": subdf["ModelMax_LDDT"].mean() * 100,
            }
        )
    meandf = pd.DataFrame(scores).set_index("CatAndGroup")
    return newdf, meandf


if __name__ == '__main__':
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <proteintest_lmdb> <prediction_directory> [max_model_num=1]")
    inplmdb, preddir = sys.argv[1:3]
    max_model_num = int(sys.argv[3]) if len(sys.argv) == 4 else 1

    logger.info(f"Loading metadata from {inplmdb}.")
    with lmdb.open(inplmdb, readonly=True).begin(write=False) as txn:
        metadata = bstr2obj(txn.get("__metadata__".encode()))
    logger.info(f"Metadata contains {len(metadata['keys'])} keys, pdbs, ...")

    logger.info(f"TMscore between predicted.pdb and native.pdb {preddir}. ")
    df = evaluate_predicted_structure(metadata, preddir)
    print(df)

    logger.info(f"Average TMscore for different categories.")
    # df.to_csv("TM-score-full.csv")
    newdf, meandf = calculate_average_score(df)
    # newdf.to_csv("TM-score-only.csv")
    print(newdf)
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(meandf)
