# -*- coding: utf-8 -*-
from subprocess import run, PIPE, DEVNULL
from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
from time import time
from shutil import rmtree



try:
    BLOB_ACCOUNT_NAME=os.environ['BLOB_ACCOUNT_NAME']
    BLOB_CONTAINER_NAME=os.environ['BLOB_CONTAINER_NAME']
    BLOB_PATH=os.environ['BLOB_PATH']
    SAS_TOKEN=os.environ['SAS_TOKEN']
except KeyError:
    print("Environment variables not set. Please set the following environment variables:")
    print("BLOB_ACCOUNT_NAME, BLOB_CONTAINER_NAME, BLOB_PATH, SAS_TOKEN")
    exit(1)


# format bytes to human readable format
def format_bytes(num):
    """
    this function will convert bytes to MB... GB... etc
    """
    for x in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if num < 1024.0:
            return f"{num:.2f} {x}"
        num /= 1024.0


def gcloud_download(urls, local_path):
    cmd = ['gcloud', 'storage', 'cp'] + urls + [local_path]
    ret = run(cmd, stdout=PIPE, stderr=PIPE,)
    # log the output to file
    with open(local_path / 'gcloud_log.txt', 'w') as f:
        f.write(f"RETURN CODE: {str(ret.returncode)}\n")
        f.write("STDOUT log:\n")
        f.write(ret.stdout.decode('utf-8'))
        f.write("STDERR log:\n")
        f.write(ret.stderr.decode('utf-8'))
    if ret.returncode != 0:
        raise Exception(f"Failed to download to {local_path}")


def azcopy_upload(local_path):
    cmd = ['azcopy', 'copy', str(local_path), f'https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{BLOB_PATH}{SAS_TOKEN}', '--recursive']
    ret = run(cmd, stdout=PIPE, stderr=PIPE,)
    if ret.returncode != 0:
        raise Exception(f"Azcopy failed: {ret.stderr.decode('utf-8')}")



def main(args):
    file_list = pd.read_csv(args.file_list, delim_whitespace=True, header=None, names=['size', 'date', 'url'])
    print("Total number of files: ", len(file_list))
    print("Total size of files: ", format_bytes(file_list['size'].sum()))
    base_path = Path(args.local_path)
    splits = np.array_split(file_list, args.num_splits)
    for idx, split in enumerate(splits):
        if idx < args.start_idx:
            print(f"Skipping split {idx}/{args.num_splits}")
            continue
        t1 = time()
        split_path = base_path / f"{idx}"
        split_path.mkdir(parents=True, exist_ok=False)
        print(f"Downloading split {idx}/{args.num_splits}, # files: {len(split)}, size: {format_bytes(sum(split['size']))} to {str(split_path)}")
        urls = split['url'].tolist()
        gcloud_download(urls, split_path)
        azcopy_upload(split_path)
        rmtree(split_path)
        t2 = time()
        print(f"Done in {t2-t1:.2f}s")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download AFDB data from google cloud and upload to azure blob storage')
    parser.add_argument('file_list', type=str, help='File containing list of files to download')
    parser.add_argument('num_splits', type=int, help='Number of splits to download')
    parser.add_argument('local_path', type=str, help='Local path to download data to')
    parser.add_argument('start_idx', type=int, help='Overwrite existing files')
    args = parser.parse_args()
    main(args)
