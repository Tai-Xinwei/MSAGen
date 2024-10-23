#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lmdb
import numpy as np
from absl import logging
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr


logging.set_verbosity(logging.INFO)


def chunks(lst: list, n: int) -> list:
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i : i + n]


def process_one_pdb(pdbid: str, inplmdb: str, cutoff: float) -> dict:
    try:
        with lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn:
            data = bstr2obj( inptxn.get(pdbid.encode()) )
        assert data, f"PDB {pdbid} not in {inplmdb}"
        assert 'pdbid' in data and data['pdbid'] == pdbid, (
            f"data['pdbid']={data['pdbid']} wrong with {pdbid} in {inplmdb}")

        assert 'polymer_chains' in data and len(data['polymer_chains']) == 1, (
            f"number of 'polymer_chains' must equal to 1")
        chain = data['polymer_chains'].popitem()[1]
        score = np.mean(chain['confidence'])
        assert score >= cutoff, f"Preidction score {score:.2f} < {cutoff}"

        bstr = obj2bstr({
            'aa': chain['seqres'],
            'pos': chain['center_coord'],
            'confidence': chain['confidence'],
        })
        return pdbid, len(chain['seqres']), score, bstr
    except Exception as e:
        logging.error("Failed to processing %s, %s", pdbid, e)
        return None


def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input_lmdb> <output_lmdb> "
                 f"[prediction_score_cutoff=(70.0)]")
    inplmdb, outlmdb = sys.argv[1:3]
    score_cutoff = float(sys.argv[3]) if len(sys.argv) == 4 else 70.0

    assert Path(inplmdb).exists(), f"{inplmdb} not exists."
    assert not Path(outlmdb).exists(), f"{outlmdb} exists, please remove first."

    with lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn:
        inpmeta = bstr2obj( inptxn.get('__metadata__'.encode()) )
    assert inpmeta, f"ERROR: {inplmdb} has no key '__metadata__'"

    logging.info("Processing original lmdb %s", inplmdb)
    print(inpmeta['comment'], end='')

    assert 'keys' in inpmeta, f"'keys' not in {inplmdb}"
    logging.info("Total original structures: %d", len(inpmeta['keys']))

    pdbids = inpmeta['keys']
    logging.info("Processing %d structures in %s", len(pdbids), inplmdb)

    # data = process_one_pdb(pdbids[0], inplmdb, score_cutoff)
    # print(data[:-1])

    env = lmdb.open(outlmdb, map_size=1024**4) # 1TB max size

    metadata = {
        'keys': [],
        'sizes': [],
        'confidences': [],
        'comment' : (
            f'Postprocessed time: {datetime.now()}\n'
            f'Original lmdb: {inplmdb}\n'
            f'Postprocessed lmdb: {outlmdb}\n'
            f'Prediction score cutoff: {score_cutoff}\n'
        ),
    }

    for pdbid_chunk in tqdm(list(chunks(pdbids, 10000)),
                            desc='Processing structures for 10k/chunk'):
        result_chunk = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(process_one_pdb)(_, inplmdb, score_cutoff)
            for _ in tqdm(pdbid_chunk, desc='Processing pdb')
        )
        with env.begin(write=True) as txn:
            for result in tqdm(result_chunk, desc='Saving to lmdb'):
                if not result:
                    # skip empty data
                    continue
                _key, _size, _confidence, _bstr = result
                txn.put(_key.encode(), _bstr)
                metadata['keys'].append(_key)
                metadata['sizes'].append(_size)
                metadata['confidences'].append(_confidence)

    metadata['comment'] = inpmeta['comment'] + metadata['comment']
    with env.begin(write=True) as txn:
        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    env.close()


if __name__ == "__main__":
    main()
