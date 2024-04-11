# -*- coding: utf-8 -*-
from shutil import rmtree
from threading import local
import pandas as pd
import click
import os
from subprocess import run, PIPE, DEVNULL
import lmdb
from tqdm import trange, tqdm
from pathlib import Path
from joblib import Parallel, delayed
import zlib, pickle
import datetime



try:
    BLOB_ACCOUNT_NAME=os.environ['BLOB_ACCOUNT_NAME']
    BLOB_CONTAINER_NAME=os.environ['BLOB_CONTAINER_NAME']
    # BLOB_PATH=os.environ['BLOB_PATH']
    SAS_TOKEN=os.environ['SAS_TOKEN']
except KeyError:
    print("Environment variables not set. Please set the following environment variables:")
    print("BLOB_ACCOUNT_NAME, BLOB_CONTAINER_NAME, BLOB_PATH, SAS_TOKEN")
    exit(1)


def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))


def load_accession_id(accession_id_file):
    df = pd.read_csv(accession_id_file, header=None, names=['accession_id', 'start', 'end', 'afdb_id', 'version'])
    return df

def azcopy_download(azcopy, local_path, remote_folder):
    cmd = [azcopy, 'copy', f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{remote_folder}{SAS_TOKEN}', str(local_path), '--recursive']
    ret = run(cmd, stdout=PIPE, stderr=PIPE,)
    if ret.returncode != 0:
        raise Exception(f"Azcopy failed: STDERR: {ret.stderr.decode('utf-8')}\nSTDOUT: {ret.stdout.decode('utf-8')}")


def azcopy_upload(azcopy, local_path, remote_folder):
    cmd = [azcopy, 'copy', str(local_path), f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{remote_folder}{SAS_TOKEN}', '--recursive']
    ret = run(cmd, stdout=PIPE, stderr=PIPE,)
    if ret.returncode != 0:
        raise Exception(f"Azcopy failed: STDERR: {ret.stderr.decode('utf-8')}\nSTDOUT: {ret.stdout.decode('utf-8')}")


def helper(azcopy, index, local_path):
    azcopy_download(azcopy, local_path, f"AFDBv4-processed/{index}")
    env = lmdb.open(str(local_path / f"{index}" / f'{index}.lmdb'), map_size=1024**4)
    txn = env.begin(write=False)
    exist_keys = [i.decode("utf-8").replace('-model_v4.cif', '') for i in txn.cursor().iternext(values=False)]
    exist_keys.remove('__metadata__')
    # remove local_path / f"{index}"
    rmtree(local_path / f"{index}")
    return exist_keys


@click.command()
@click.argument('local_path', type=click.Path(exists=True))
@click.option('--azcopy', type=click.Path(exists=False), default='azcopy')
@click.option('--start', type=int, default=0)
@click.option('--end', type=int, default=200)
def list_keys(local_path, azcopy, start, end):
    local_path = Path(local_path)
    exist_keys = Parallel(n_jobs=8)(delayed(helper)(azcopy, index, local_path) for index in trange(start, end))
    with open(f'exist_keys_{start}_{end}.txt', 'w') as f:
        for keys in exist_keys:
            for key in keys:
                f.write(key + '\n')




def plddt_filter_helper(azcopy, index, local_path, output_file, plddt_lb):
    if Path(f'/sfmdata/protein/AFDBv4-plddt{plddt_lb}/{index}/{index}_{plddt_lb}.lmdb').is_dir():
        print(f"Shard {index} already processed, skipping")
        return
    print("Downloading shard", index)
    azcopy_download(azcopy, local_path, f"AFDBv4-processed/{index}")
    env = lmdb.open(str(output_file), map_size=1024**4)
    txn = env.begin(write=True)
    env2 = lmdb.open(str(local_path / f"{index}" / f'{index}.lmdb'), map_size=1024**4, readonly=True)
    txn2 = env2.begin(write=False)
    metadata = {"sizes": [], "keys": [], "comment": ""}
    nbefore, nafter = 0, 0
    for key, value in txn2.cursor():
        if key == b"__metadata__":
            continue
        nbefore += 1
        data = bstr2obj(value)
        try:
            if data['confidence'].mean() < plddt_lb:
                continue
        except KeyError as err:
            print(f"ERROR: {err}")
            print(f"ERROR: No confidence data found, {key}, {index} shard, {output_file}")
            exit(1)
        nafter += 1
        txn.put(key, value)
        metadata["sizes"].append(data["size"])
        metadata["keys"].append(key.decode())
    txn.commit()
    env2.close()
    print(f"\nShard {index}: {nafter}/{nbefore} ({nafter / nbefore *100:.2f}%) structures passed the filter")
    txn = env.begin(write=True)
    metadata["comment"] = (
        f"Created time: {(datetime.datetime.now())}\n"
        + f"pLDDT threshold: {plddt_lb}\n"
    )
    txn.put("__metadata__".encode(), obj2bstr(metadata))
    txn.commit()
    env.close()

    azcopy_upload(azcopy, output_file, f"AFDBv4-plddt{plddt_lb}/{index}")
    rmtree(local_path / f"{index}")
    rmtree(output_file)


def filter_cluster_helper(azcopy, index, local_path, output_file, si, key_set):
    azcopy_download(azcopy, local_path, f"AFDBv4-processed/{index}")

    env = lmdb.open(str(output_file), map_size=1024**4)
    txn = env.begin(write=True)

    env2 = lmdb.open(str(local_path / f"{index}" / f'{index}.lmdb'), map_size=1024**4, readonly=True)
    txn2 = env2.begin(write=False)

    metadata = {"sizes": [], "keys": [], "comment": ""}
    # datadict = dict()
    nbefore, nafter = 0, 0
    for key, value in tqdm(txn2.cursor(), desc=f"{index}", leave=False):
        if key == b"__metadata__":
            continue
        nbefore += 1
        if key not in key_set:
            continue
        nafter += 1
        data = bstr2obj(value)
        txn.put(key, value)
        metadata["sizes"].append(data["size"])
        metadata["keys"].append(key.decode())

    txn.commit()
    env2.close()


    print(f"\nShard {index}: {nafter}/{nbefore} ({nafter / nbefore *100:.2f}%) structures passed the filter")
    txn = env.begin(write=True)
    metadata["comment"] = (
        f"Created time: {(datetime.datetime.now())}\n"
        f"Sequence similarity threshold: {si}"
    )
    txn.put("__metadata__".encode(), obj2bstr(metadata))
    txn.commit()
    env.close()

    azcopy_upload(azcopy, output_file, f"AFDBv4-cluster/AFDB{si}/{index}/")
    rmtree(local_path / f"{index}")
    rmtree(output_file)


@click.command()
@click.argument("plddt_lb", type=int)
@click.argument('local_path', type=click.Path(exists=True))
# @click.argument("output_file", type=click.Path(writable=True))
@click.option('--azcopy', type=click.Path(exists=False), default='azcopy')
def plddt_filter(plddt_lb: int, local_path: str, azcopy: str):
    local_path = Path(local_path)
    Parallel(n_jobs=4)(delayed(plddt_filter_helper)(azcopy, index, local_path, local_path/f"{index}_{plddt_lb}.lmdb", plddt_lb) for index in trange(1000, ncols=150, desc="LMDB chunk"))


@click.command()
@click.argument("cluster_lst", type=click.Path(exists=True))
@click.argument('local_path', type=click.Path(exists=True))
@click.argument("si", type=int)
@click.option('--azcopy', type=click.Path(exists=False), default='azcopy')
@click.option('--njobs', type=int, default=4)
@click.option('--start', type=int, default=0)
@click.option('--end', type=int, default=1000)
def clust(cluster_lst: str, local_path: str, si: int, azcopy: str, njobs: int, start: int, end: int):
    local_path = Path(local_path)
    with open(cluster_lst, 'r') as f:
        key_set = set([(i + '-model_v4.cif').encode() for i in f.read().splitlines()])
    print(f"Number of keys in cluster list: {len(key_set)}")
    fulllist, idxlist = list(range(start, end)), []
    for i in fulllist:
        if not Path(f'/sfmdata/protein/AFDBv4-cluster/AFDB{si}/{i}/{i}.lmdb').is_dir():
            idxlist.append(i)
        else:
            print(f"Shard {i} already processed, skipping")
    Parallel(n_jobs=njobs)(delayed(filter_cluster_helper)(azcopy, index, local_path, local_path/f"{index}.lmdb", si, key_set) for index in tqdm(idxlist, ncols=150, desc="LMDB chunk"))# trange(start, end, ncols=150, desc="LMDB chunk"))

@click.group()
def cli():
    pass



if __name__ == '__main__':
    cli()
