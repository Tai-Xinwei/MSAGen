# -*- coding: utf-8 -*-
import datetime
from pathlib import Path
from typing import List, Union

import click
import lmdb
from joblib import Parallel, delayed
from tqdm import tqdm

from commons import bstr2obj, obj2bstr, Protein
import logging
import commons
from io import StringIO
import gzip
import json
import numpy as np

from process_pdb_complex import process_pdb_complex


logger = logging.getLogger(__name__)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_item(input_path: Path):
    prot = Protein.from_file(input_path, input_path.stem)
    d = prot.to_dict()
    confidence_path = input_path.parent / (input_path.name.rsplit("-", 1)[0] + '-confidence_v4.json.gz')
    pae_path = input_path.parent / (input_path.name.rsplit("-", 1)[0] + '-predicted_aligned_error_v4.json.gz')
    if confidence_path.exists():
        d["confidence"] = np.array(json.loads(gzip.open(confidence_path, "rt").read())['confidenceScore'])
    # if pae_path.exists():
    #     try:
    #         d["pae"] = np.array(json.loads(gzip.open(pae_path, "rt").read())[0]['predicted_aligned_error'])
    #     except:
    #         print(pae_path)
    bstr = obj2bstr(d)
    return prot.name, len(prot), bstr


def list_keys(env: lmdb.Environment):
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    return keys


@click.group()
def cli():
    pass


@cli.command()
@click.argument("mmcifdir", type=click.Path(exists=True))
@click.argument("chemcompdir", type=click.Path(exists=True))
@click.argument("inplist", type=click.Path(readable=True))
@click.argument("outfile", type=click.Path(writable=True))
@click.option("--num-workers", type=int, default=-1, help="Number of workers.")
@click.option("--comment", type=str, default="", help="Comments for output.")
def processpdb(mmcifdir: str, chemcompdir: str, inplist: str, outfile: str,
               num_workers: int, comment: str):
    """Process training data from mmCIF files and save to lmdb."""
    logger.warning(f"Processing struture for list {inplist}.")
    pdbids = []
    with open(inplist, 'r') as fp:
        for line in fp:
            pdbids.append(line.strip())
    assert all(4==len(_) for _ in pdbids), (
        f"ERROR: PDBID should be 4 characters long in {inplist}.")
    logger.warning(f"Found {len(pdbids)} structures in {mmcifdir}.")

    if Path(outfile).exists():
        logger.warning(f"Output file {outfile} exists. Stop.")
        return
    logger.warning(f"Save to {outfile}")

    def _process_one(pdbid: str, mmcif_dir: str, chem_comp_dir: str):
        mmcif_path = str( Path(mmcif_dir) / f"{pdbid}.cif" )
        data = process_pdb_complex(mmcif_path, chem_comp_dir)
        if not data:
            return pdbid, None
        return pdbid, obj2bstr(data)

    env = lmdb.open(outfile, map_size=1024**4)
    metadata = {'keys': [],
                'comment': (f'Created time: {datetime.datetime.now()}\n'
                            f'Structure directory: {mmcifdir}\n'
                            f'Chem_comp directory: {chemcompdir}\n'
                            f'Input list: {inplist}\n'
                            f'Comments: {comment}\n')}
    pbar = tqdm(total=len(pdbids)//1000+1, desc='Processing chunks (1k)')
    for path_chunk in chunks(pdbids, 1000):
        res_chunk = Parallel(n_jobs=num_workers)(
            delayed(_process_one)(i, str(mmcifdir), str(chemcompdir))
            for i in tqdm(path_chunk, leave=False, desc="Processing...")
        )
        with env.begin(write=True) as txn:
            for name, res in res_chunk:
                if not res:
                    txn.put(name.encode(), res)
                    metadata['keys'].append(name)
        pbar.update(1)
    pbar.close()
    with env.begin(write=True) as txn:
        txn.put('__metadata__'.encode(), obj2bstr(metadata))
    env.close()


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(writable=True))
@click.option("--num-workers", type=int, default=-1)
@click.option("--comment", type=str, default="")
@click.option("--glob", type=str, default="*.cif")
def processafdb(
    directory: str, output_file: str, num_workers: int, glob: str, comment: str
):
    logger.warning(f"Processing {glob} in {directory}")
    logger.warning(f"Save to {output_file}")
    directory = Path(directory)
    pathlst = [i for i in directory.glob(f"{glob}")]
    logger.warning(f"Found {len(pathlst)} structures.")
    if Path(output_file).exists():
        logger.warning(f"Output file {output_file} exists. Stop.")
        return
    env = lmdb.open(output_file, map_size=8*1024**4) # 8TB max size
    metadata = {
        "sizes": [],
        "keys": [],
        "comment": f"Created time: {str(datetime.datetime.now())}\n"
        f"Base directory: {str(directory)}\n" + comment,
    }

    pbar = tqdm(total=len(pathlst) // 1000 + 1, ncols=80, desc="Processing chunks (1k)")
    for path_chunk in chunks(pathlst, 1000):
        res_chunk = Parallel(n_jobs=num_workers)(
            delayed(process_item)(p)
            for p in tqdm(path_chunk, ncols=80, leave=False, desc="Processing data")
        )
        txn = env.begin(write=True)
        for name, size, result in res_chunk:
            key = name.encode()
            txn.put(key, result)
            metadata["sizes"].append(size)
            metadata["keys"].append(name)
        txn.commit()
        pbar.update(1)
    pbar.close()
    with env.begin(write=True) as txn:
        txn.put("__metadata__".encode(), obj2bstr(metadata))
    env.close()


@cli.command()
@click.argument("lmdb_files", nargs=-1, type=click.Path(dir_okay=True, readable=True))
@click.argument("output_file", type=click.Path(writable=True))
def merge(lmdb_files: List[str], output_file: str):
    logger.warning(f"Merging {len(lmdb_files)} lmdb files into {output_file}")
    metadata_dicts = []
    metadata = {"sizes": [], "keys": [], "comment": ""}
    env = lmdb.open(output_file, map_size=8*1024**4) # 8TB max size
    for lmdb_file in tqdm(lmdb_files, ncols=80, desc="# LMDBs merged"):
        txn = env.begin(write=True)
        exist_keys = set(list(txn.cursor().iternext(values=False)))
        env2 = lmdb.open(lmdb_file, readonly=True, map_size=8*1024**4) # 8TB max size
        keys = list(env2.begin(write=False).cursor().iternext(values=False))
        # progress bar
        with tqdm(total=len(keys), ncols=150, desc=f"Merging {Path(lmdb_file).stem}", leave=False) as pbar:
            with env2.begin() as txn2:
                for key, value in txn2.cursor():
                    if key == b"__metadata__":
                        continue
                    # check for duplicates
                    if key in exist_keys:
                        logger.error(f"Duplicate key {key} found in {lmdb_file}, skipped.Please modify the key and 'keys' in __metadata__")
                        continue
                    txn.put(key, value)
                    metadata["sizes"].append(bstr2obj(value)["size"])
                    metadata["keys"].append(key.decode())
                    pbar.update(1)
                metadata_dicts.append(bstr2obj(txn2.get(b"__metadata__")))
        txn.commit()
        env2.close()
    txn = env.begin(write=True)
    metadata["comment"] = (
        f"Merged time: {(datetime.datetime.now())}\n"
        + "Merged LMDBs: "
        + "----------\n".join([md["comment"] for md in metadata_dicts])
    )
    txn.put("__metadata__".encode(), obj2bstr(metadata))
    txn.commit()
    env.close()




def fix_item(pdb_file: Path, format: str, indir: Path, outdir: Path):
    dst_dir = outdir / pdb_file.relative_to(indir).parent
    dst_file = dst_dir / f"fixed_{pdb_file.stem}.pdb"
    if dst_file.exists():
        return
    if pdb_file.name.endswith(".gz"):
        with gzip.open(pdb_file, "rt") as f:
            pdb_str = f.read()
    else:
        with open(pdb_file, "r") as f:
            pdb_str = f.read()
    try:
        fixed_str = commons.fix_structure(StringIO(pdb_str), format=format).getvalue()
    except Exception as e:
        print(f"Error in {pdb_file}: {e} {e.args}, skipping.")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_file, "w") as fout:
        fout.write(fixed_str)
    return True

@cli.command()
@click.argument("indir", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path(exists=False))
@click.option("--glob", type=str, default="*.pdb")
@click.option("--format", type=str, default="pdb")
@click.option("--num-workers", type=int, default=-1)
def fix_pdb(indir: Union[Path, str], outdir: Union[Path, str], glob: str, format: str, num_workers: int):
    indir, outdir = Path(indir), Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=num_workers)(
        delayed(fix_item)(pdb_file, format, indir, outdir)
        for pdb_file in tqdm(list(indir.glob(glob)), ncols=80, desc="Fixing")
    )


if __name__ == "__main__":
    cli()
