# -*- coding: utf-8 -*-
import json
import logging
import logging.handlers
import os
import pickle
import zlib
from dataclasses import dataclass
from shutil import rmtree
from typing import Optional

import click
import datasets
import datasets.config
import lmdb
from datasets import Dataset, DatasetInfo
from datasets.utils.py_utils import asdict, convert_file_size_to_int
from datasets.utils.tqdm import disable_progress_bars, enable_progress_bars
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Confirm, Prompt
from rich.table import Table
from tqdm import tqdm

from sfm.experimental.datasets import utils
from sfm.experimental.datasets.builder import SFMDatasetBuilder, SFMDatasetBuilderConfig
from sfm.experimental.datasets.logging import logger


@dataclass
class CliContext:
    cache_dir: str
    num_proc: int


@click.group()
@click.option(
    "--cache-dir",
    type=click.Path(exists=False),
    default="/data/cache",
    show_default=True,
    help="Cache directory",
)
@click.option(
    "--num-proc",
    type=int,
    default=os.cpu_count(),
    show_default=True,
    help="Number of processes to use",
)
@click.pass_context
def main(ctx: click.Context, cache_dir: str, num_proc: int):
    """
    [Experimental] SFM Datasets CLI.
    """
    datasets_logger = logging.getLogger("datasets")
    datasets_logger.setLevel(logging.INFO)
    datasets_logger.handlers = []
    logging.getLogger("datasets.arrow_dataset").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").handlers = []
    ctx.ensure_object(dict)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    ctx.obj["cache_dir"] = cache_dir
    ctx.obj["num_proc"] = num_proc


@main.command("list")
@click.pass_context
def list(ctx: click.Context):
    """
    List available datasets
    """
    logger.fatal("[red]Not implemented yet")


@main.command("show")
@click.argument("path", type=click.Path(exists=True), metavar="DATASET_PATH")
@click.pass_context
def show(ctx: click.Context, path: str):
    """
    Show dataset info
    """
    splits = utils.load_dataset_infos_dict(path)
    logger.info(f"{len(splits)} splits found: {', '.join(splits)}")

    console = Console()
    for split, info in splits.items():
        console.rule(f"[bold red]{split} split")
        pprint(info, max_length=20, console=console)


@main.command("download")
@click.pass_context
def download(ctx: click.Context):
    """
    Download a dataset
    """
    logger.fatal("[red]Not implemented yet")


@main.command("build")
@click.argument(
    "script", type=click.Path(exists=True), metavar="DATASET_LOADING_SCRIPT"
)
@click.option("--config", "-c", type=str, help="BuilderConfig name")
@click.option("--root-dir", "-d", type=str, help="Input data root directory")
@click.option("--save-dir", "-o", type=str, help="Output data directory")
@click.option("--num-shards", type=int, help="Number of shards")
@click.option(
    "--force", is_flag=True, help="Force rebuild if the dataset already exists"
)
@click.pass_context
def build(
    ctx: click.Context,
    script: str,
    config: Optional[str] = None,
    root_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    num_shards: Optional[int] = None,
    force: bool = False,
):
    """
    Build a dataset from a Huggingface dataset loading script
    """
    Console().rule(
        f"[cyan]Building dataset from [bold green]{os.path.basename(script)}/{'default' if not config else config}"
    )

    config_kwargs = {}
    if root_dir:
        config_kwargs["data_root_dir"] = root_dir
    if save_dir:
        config_kwargs["data_save_dir"] = save_dir

    builder: SFMDatasetBuilder = datasets.load_dataset_builder(
        script, name=config, cache_dir=ctx.obj["cache_dir"], **config_kwargs
    )
    config: SFMDatasetBuilderConfig = builder.config

    save_dir = config.data_save_dir
    if save_dir and builder._fs.exists(
        os.path.join(save_dir, datasets.config.DATASETDICT_JSON_FILENAME)
    ):
        if not force:
            logger.warning(f"[yellow]Dataset already exists at {save_dir}")
            return
        else:
            logger.warning(
                f"[red]Dataset already exists at {save_dir}. Forcing rebuild"
            )
            if not Confirm.ask(
                f"[red]{save_dir} will be removed. Are you sure you want to continue?"
            ):
                return
            rmtree(save_dir)

    builder.download_and_prepare(
        download_mode="force_redownload",
        num_proc=ctx.obj["num_proc"],
        verification_mode=datasets.VerificationMode.NO_CHECKS,
    )

    if save_dir:
        splits = builder.as_dataset()
        logger.info(f"Saving dataset to {save_dir}")
        num_shards = {k: num_shards or 1 for k in splits}
        for k, ds in splits.items():
            dataset_nbytes = ds._estimate_nbytes()
            max_shard_size = convert_file_size_to_int(datasets.config.MAX_SHARD_SIZE)
            min_num_shards = int(dataset_nbytes / max_shard_size) + 1
            num_shards[k] = max(num_shards[k], min_num_shards)
        splits.save_to_disk(
            save_dir, num_proc=ctx.obj["num_proc"], num_shards=num_shards
        )

        for k, ds in splits.items():
            logger.info(f"Collecting metadata for {k} split")
            metadata = builder.get_metadata(ds, num_proc=ctx.obj["num_proc"])
            if metadata:
                path = os.path.join(save_dir, k, "metadata.pickle.gz")
                logger.info(
                    f"Saving metadata ({','.join(metadata.keys())}) for {k} split to {path}"
                )
                with open(path, "wb") as f:
                    f.write(zlib.compress(pickle.dumps(metadata)))
            else:
                logger.warning(f"No metadata was collected for {k} split")


@main.command("lmdb")
@click.argument("path", type=click.Path(exists=True), metavar="DATASET_PATH")
@click.pass_context
def to_lmdb(ctx: click.Context, path: str):
    """Convert a dataset to LMDB format"""

    if os.path.exists(os.path.join(path, "lmdb")):
        if not Confirm.ask(
            f"[red]LMDB already exists at {path}/lmdb. Do you want to continue?"
        ):
            return
        rmtree(os.path.join(path, "lmdb"))

    def _write_examples(examples, idx, lmdb_path):
        batch = []
        for i, key in enumerate(idx):
            batch.append(
                (str(key).encode(), pickle.dumps({k: examples[k][i] for k in examples}))
            )

        with lmdb.open(lmdb_path, map_size=1024**4, lock=True) as env:
            with env.begin(write=True) as txn:
                for k, v in batch:
                    txn.put(k, v)

    dataset = datasets.load_from_disk(path)
    for split, ds in dataset.items():
        lmdb_path = os.path.join(path, f"lmdb/{split}")
        os.makedirs(lmdb_path)
        _ = ds.map(
            lambda x, idx: _write_examples(x, idx, lmdb_path),
            with_indices=True,
            batched=True,
            batch_size=10000,
            num_proc=ctx.obj["num_proc"],
            desc=f"Writing {split} split to LMDB",
        )


@main.command("parquet")
@click.argument("path", type=click.Path(exists=True), metavar="DATASET_PATH")
@click.option("--save-dir", "-o", type=str, help="Output data directory")
@click.option("--columns", type=str, help="Columns to save")
@click.option(
    "--num-shards", type=int, default=1, show_default=True, help="Number of shards"
)
@click.pass_context
def to_parquet(
    ctx: click.Context,
    path: str,
    save_dir: Optional[str] = None,
    columns: Optional[str] = None,
    num_shards: int = 1,
):
    """Convert a dataset to Parquet format"""
    save_dir = save_dir or os.path.join(path, "parquet")
    if os.path.exists(save_dir):
        if not Confirm.ask(
            f"[red]Parquet already exists at {save_dir}. Do you want to continue?"
        ):
            return
        rmtree(save_dir)

    dataset = datasets.load_from_disk(path)
    columns = columns.split(",") if columns else None
    for split, ds in dataset.items():
        utils.save_dataset_to_parquet(
            ds,
            os.path.join(save_dir, split),
            columns=columns,
            num_shards=num_shards,
            num_proc=ctx.obj["num_proc"],
        )


@main.command("publish")
@click.pass_context
def publish(ctx: click.Context):
    """
    Publish a dataset
    """
    logger.fatal("[red]Not implemented yet")
