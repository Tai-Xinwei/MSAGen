# -*- coding: utf-8 -*-
import datetime
import gzip
import json
import logging
from io import StringIO
from pathlib import Path
from typing import List
from typing import Union

import click
import lmdb
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from commons import bstr2obj
from commons import fix_structure
from commons import obj2bstr
from commons import Protein
from process_pdb_complex import process_pdb_complex


#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(level=logging.INFO)
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
@click.option("--mmcif-dir",
              type=click.Path(exists=True),
              help="Input directory of mmCIF files rsync from RCSB.")
@click.option("--chem-comp-file",
              type=click.Path(exists=True),
              default="components.cif",
              help="Input mmCIF file of all chemical components.")
@click.option("--output-lmdb",
              type=click.Path(exists=False),
              default="output.lmdb",
              help="Output lmdb file.")
@click.option("--remove-hydrogens",
              type=bool,
              default=True,
              help="Remove hydrogen atoms.")
@click.option("--num-workers",
              type=int,
              default=-1,
              help="Number of workers.")
@click.option("--data-comment",
              type=str,
              default="PDB snapshot from https://snapshots.pdbj.org/20240101/.",
              help="Comments for output.")
def processpdb(mmcif_dir: str, chem_comp_file: str, output_lmdb: str,
               remove_hydrogens: bool, num_workers: int, data_comment: str):
    """Process training data from mmCIF files and save to lmdb."""
    mmcif_dir = Path(mmcif_dir).resolve()
    mmcif_paths = [_ for _ in Path(mmcif_dir).rglob("*.cif.gz")]
    assert mmcif_paths and all(11==len(_.name) for _ in mmcif_paths), (
        f"PDBID should be 4 characters long in {mmcif_dir}.")
    logger.info(f"Processing {len(mmcif_paths)} structures in {mmcif_dir}.")

    chem_comp_path = Path(chem_comp_file).resolve()
    logger.info(f"Chemical components information is in {chem_comp_path}")

    if Path(output_lmdb).exists():
        logger.error(f"Output file {output_lmdb} exists. Stop.")
        return
    logger.info(f"Save processed data to {output_lmdb}")

    def _process_one_pdb(mmcif_path):
        pdbid, header, data = process_pdb_complex(
            str(mmcif_path), str(chem_comp_path), remove_hydrogens)
        return pdbid, header, data

    env = lmdb.open(output_lmdb, map_size=1024**4) # 1TB max size
    metadata = {'keys': [],
                'structure_methods': [],
                'release_dates': [],
                'resolutions': []}
    metadata['comment'] = (
        f'Created time: {datetime.datetime.now()}\n'
        f'Structure directory: {mmcif_dir}\n'
        f'Chemical components: {chem_comp_path}\n'
        f'Output lmdb: {output_lmdb}\n'
        f'Remove hydrogens: {remove_hydrogens}\n'
        f'Number of workers: {num_workers}\n'
        f'Comments: {data_comment}\n'
        )

    pbar = tqdm(total=len(mmcif_paths)//10000+1, desc='Processing chunks (10k)')
    for path_chunk in chunks(mmcif_paths, 10000):
        res_chunk = Parallel(n_jobs=num_workers)(
            delayed(_process_one_pdb)(p) for p in tqdm(path_chunk)
            )
        with env.begin(write=True) as txn:
            for pdbid, header, result in res_chunk:
                if not result: continue # skip empty data
                txn.put(pdbid.encode(), obj2bstr(result))
                metadata['keys'].append(pdbid)
                metadata['structure_methods'].append(header['structure_method'])
                metadata['release_dates'].append(header['release_date'])
                metadata['resolutions'].append(header['resolution'])
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
        fixed_str = fix_structure(StringIO(pdb_str), format=format).getvalue()
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
