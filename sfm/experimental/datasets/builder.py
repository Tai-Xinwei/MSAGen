# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, Optional, Union

import datasets
import datasets.config
from datasets import BuilderConfig

from sfm.experimental.datasets.logging import logger


class SFMDatasetBuilderConfig(BuilderConfig):
    def __init__(
        self,
        data_root_dir: Optional[str] = None,
        data_extra_dirs: Optional[Dict[str, str]] = None,
        data_save_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root_dir = data_root_dir
        self.data_extra_dirs = data_extra_dirs or {}
        self.data_save_dir = data_save_dir or os.path.join(self.name, str(self.version))

        self.data_root_dir = data_root_dir
        if data_root_dir:
            if self.data_dir:
                self.data_dir = os.path.join(data_root_dir, self.data_dir)
            if self.data_extra_dirs:
                self.data_extra_dirs = {
                    k: os.path.join(data_root_dir, v)
                    for k, v in self.data_extra_dirs.items()
                }
            self.data_save_dir = os.path.join(data_root_dir, self.data_save_dir)


class SFMDatasetBuilder(datasets.GeneratorBasedBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.config, SFMDatasetBuilderConfig)
        self.parents = {}

    def download_and_prepare(self, **kwargs):
        config: SFMDatasetBuilderConfig = self.config
        for name, path in config.data_extra_dirs.items():
            logger.info(
                f"[green]{self.name}[/green] requires [green]{name}[/green] at {path}"
            )
            self.parents[name] = datasets.load_from_disk(path)

        super().download_and_prepare(**kwargs)

    def get_metadata(
        self, dataset: datasets.Dataset, num_proc: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        return None


class MolecularDatasetBuilderConfig(SFMDatasetBuilderConfig):
    def __init__(
        self,
        theory,
        units: Optional[Dict[str, str]] = None,
        infer_smiles_from_bonds: bool = True,
        infer_smiles_from_coords: bool = False,
        ogb_feature_smiles_column: Optional[str] = "smiles_from_bonds",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theory = theory
        self.units = units or {"energy": "eV", "coords": "angstrom"}
        self.infer_smiles_from_bonds = infer_smiles_from_bonds
        self.infer_smiles_from_coords = infer_smiles_from_coords
        self.ogb_feature_smiles_column = ogb_feature_smiles_column
