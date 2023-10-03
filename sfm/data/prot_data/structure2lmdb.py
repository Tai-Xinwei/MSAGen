# -*- coding: utf-8 -*-
import datetime
from pathlib import Path
from typing import List

import click
import lmdb
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from sfm.data.prot_data.process import bstr2obj, obj2bstr, process_cif, process_conf
from sfm.logging import logger


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_item(base: Path, name: str, angles: List[str]):
    key_stru = name + "-model_v4.cif"
    cif_str = (base / key_stru).read_text()
    struc_dict = process_cif(cif_str, angles)
    # confidence score is optional
    key = name + "-confidence_v4.json"
    jsonpath = base / key
    if not jsonpath.exists():
        json_str = ""
        logger.warning(f"Confidence file {jsonpath} not found, set to all 1.")
    else:
        json_str = jsonpath.read_text()
    conf_dict = process_conf(json_str)
    result = {**struc_dict, **conf_dict, "name": name}
    if result["conf"] is None:
        result["conf"] = np.full(len(result["aa"]), 1, dtype=np.float32)
    bstr = obj2bstr(result)
    return name, len(result["aa"]), bstr


@click.group()
def cli():
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(writable=True))
@click.argument(
    "angles",
    nargs=-1,
    type=str,
)  # default=["-1C:N:CA:C", "N:CA:C:1N", "-1CA:-1C:N:CA"])
@click.option("--num-workers", type=int, default=-1)
@click.option("--comment", type=str, default="")
def process(
    directory: str, output_file: str, angles: List[str], num_workers: int, comment: str
):
    logger.warning(f"Processing *-model_v4.cif and *-confidence_v4.json in {directory}")
    logger.warning(f"Save to {output_file}")
    logger.warning(f"Angles will be included: {angles}")
    directory = Path(directory)
    # namelst = [i.name.split('-model_v4.cif')[0] for i in directory.glob('**/*-model_v4.cif')]
    pathlst = [i for i in directory.glob("**/*-model_v4.cif")]
    logger.warning(f"Found {len(pathlst)} structures.")

    env = lmdb.open(output_file, map_size=1024**4)
    metadata = {
        "sizes": [],
        "names": [],
        "comment": f"Created time: {str(datetime.datetime.now())}\n"
        f"Base directory: {str(directory)}\n" + comment,
    }

    pbar = tqdm(total=len(pathlst) // 1000 + 1, ncols=80, desc="Processing chunks (1k)")
    for path_chunk in chunks(pathlst, 1000):
        res_chunk = Parallel(n_jobs=num_workers)(
            delayed(process_item)(p.parent, p.name.split("-model_v4.cif")[0], angles)
            for p in tqdm(path_chunk, ncols=80, leave=False, desc="Processing data")
        )
        txn = env.begin(write=True)
        for name, size, result in res_chunk:
            key = f"{name}".encode()
            txn.put(key, result)
            metadata["sizes"].append(size)
            metadata["names"].append(name)
        txn.commit()
        pbar.update(1)
    pbar.close()
    with env.begin(write=True) as txn:
        txn.put("metadata".encode(), obj2bstr(metadata))
    env.close()


@cli.command()
@click.argument("lmdb_files", nargs=-1, type=click.Path(dir_okay=True, readable=True))
@click.argument("output_file", type=click.Path(writable=True))
def merge(lmdb_files: List[str], output_file: str):
    logger.warning(f"Merging {len(lmdb_files)} lmdb files into {output_file}")
    metadata_dicts = []
    env = lmdb.open(output_file, map_size=1024**4)
    for lmdb_file in tqdm(lmdb_files, ncols=80, desc="# LMDBs merged"):
        txn = env.begin(write=True)
        with lmdb.open(lmdb_file, readonly=True).begin() as txn2:
            for key, value in txn2.cursor():
                txn.put(key, value)
            metadata_dicts.append(bstr2obj(txn2.get("metadata".encode())))
        txn.commit()

    txn = env.begin(write=True)
    metadata = {"sizes": [], "names": [], "comment": ""}
    for md in metadata_dicts:
        metadata["sizes"].extend(md["sizes"])
        metadata["names"].extend(md["names"])
    metadata["comment"] = (
        f"Merged time: {(datetime.datetime.now())}\n"
        + "Merged LMDBs: "
        + "----------\n".join([md["comment"] for md in metadata_dicts])
    )
    txn.put("metadata".encode(), obj2bstr(metadata))
    txn.commit()
    env.close()


if __name__ == "__main__":
    cli()
