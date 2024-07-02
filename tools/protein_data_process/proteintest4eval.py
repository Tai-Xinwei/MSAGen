#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import lmdb
import pandas as pd

from commons import bstr2obj
from commons import obj2bstr

sys.path.append(str(Path.cwd().parent.parent))
from sfm.logging import logger
from sfm.tasks.psm.evaluate_psm_protein import evaluate_predicted_structure
from sfm.tasks.psm.evaluate_psm_protein import calculate_average_score


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
    df = evaluate_predicted_structure(metadata, preddir, max_model_num)
    print(df)

    logger.info(f"Average TMscore for different categories.")
    #df.to_csv(Path(preddir) / "TM-score-full.csv")
    newdf, meandf = calculate_average_score(df)
    #newdf.to_csv(Path(preddir) / "TM-score-only.csv")
    print(newdf)
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(meandf)
