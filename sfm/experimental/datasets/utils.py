# -*- coding: utf-8 -*-
import json
import os
from typing import List, Optional

import datasets.config
from datasets import Dataset, DatasetInfo
from datasets.info import DatasetInfosDict
from datasets.utils.tqdm import disable_progress_bars

from sfm.experimental.datasets.logging import logger


def load_dataset_infos_dict(path: str) -> DatasetInfosDict:
    dataset_dict_json_path = os.path.join(
        path, datasets.config.DATASETDICT_JSON_FILENAME
    )
    with open(dataset_dict_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    return DatasetInfosDict(
        {k: DatasetInfo.from_directory(os.path.join(path, k)) for k in splits}
    )


def save_dataset_to_parquet(
    dataset: Dataset,
    save_dir: str,
    columns: Optional[List[str]] = None,
    num_shards: int = 1,
    num_proc: Optional[int] = None,
):
    ds = dataset
    if columns:
        ds = ds.select_columns(columns)

    def _shard_to_parquet(idx: int):
        shard_path = os.path.join(
            save_dir, f"data-{idx:05d}-of-{num_shards:05d}.parquet"
        )
        shard = ds.shard(num_shards=num_shards, index=idx, contiguous=True)
        disable_progress_bars()
        shard.to_parquet(shard_path)

    shards = Dataset.from_dict({"shards": range(num_shards)})
    _ = shards.map(
        lambda _, idx: _shard_to_parquet(idx),
        with_indices=True,
        num_proc=num_proc,
        desc=f"Saving dataset as Parquet to {save_dir}",
    )
