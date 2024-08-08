# -*- coding: utf-8 -*-
import glob
import json
import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, Optional

import datasets
import numpy as np
import qcelemental as qcel
import rdkit.Chem as Chem
from datasets import Features
from rdkit import RDLogger
from tqdm import tqdm

from sfm.experimental.datasets.builder import (
    MolecularDatasetBuilderConfig,
    SFMDatasetBuilder,
)
from sfm.experimental.datasets.logging import logger
from sfm.experimental.datasets.molecule import OGB_FEATURES, Molecule, mol2graph

_CITATION = """\
@article{nakata2023pubchemqc,
  title={PubChemQC B3LYP/6-31G*//PM6 Data Set: The Electronic Structures of 86 Million Molecules Using B3LYP/6-31G* Calculations},
  author={Nakata, Maho and Maeda, Toshiyuki},
  journal={Journal of Chemical Information and Modeling},
  volume={63},
  number={18},
  pages={5734--5754},
  year={2023},
  publisher={ACS Publications}
}
"""

_HATREE_TO_EV = qcel.constants.conversion_factor("hartree", "eV")
_BOHR_TO_ANGSTROM = qcel.constants.conversion_factor("bohr", "angstrom")


class PubChemQCB3LYPPM6(SFMDatasetBuilder):
    """PubChemQC B3LYP/6-31G*//PM6 dataset."""

    DEFAULT_CONFIG_NAME = "wb97xd3"
    BUILDER_CONFIG_CLASS = MolecularDatasetBuilderConfig

    DISABLE_RDKIT_LOG: bool = True

    DEFAULT_DATA_ROOT_DIR = "/blob/data/PubChemQC-B3LYP-PM6"

    BUILDER_CONFIGS = [
        MolecularDatasetBuilderConfig(
            name="raw-b3lyp",
            version=datasets.Version("1.0.0"),
            description="Oringal PubChemQC B3LYP/6-31G*//PM6 dataset",
            theory="B3LYP/6-31G*",
            data_root_dir=DEFAULT_DATA_ROOT_DIR,
            data_dir="raw/Compounds",
            data_save_dir="raw/b3lyp/1.0.0",
        ),
        MolecularDatasetBuilderConfig(
            name="raw-wb97xd3",
            version=datasets.Version("1.0.0"),
            description="LightAIMD re-labeled dataset with WB97X-D3/def2-SVP energy and force",
            theory="WB97X-D3/def2-SVP",
            data_root_dir=DEFAULT_DATA_ROOT_DIR,
            data_dir="raw/lightaimd/wb97x-d3",
            # lightaimd dataset also requires raw b3lyp dataset to get smiles and mol features
            data_extra_dirs={"b3lyp": "raw/b3lyp/1.0.0"},
            data_save_dir="raw/wb97xd3/1.0.0",
        ),
        MolecularDatasetBuilderConfig(
            name="b3lyp",
            version=datasets.Version("1.0.0"),
            description="PubChemQC B3LYP/6-31G*//PM6 dataset for SFM",
            theory="B3LYP/6-31G*",
            data_root_dir=DEFAULT_DATA_ROOT_DIR,
            data_dir="raw/b3lyp/1.0.0",
        ),
        MolecularDatasetBuilderConfig(
            name="wb97xd3",
            version=datasets.Version("1.0.0"),
            description="PubChemQC WB97X-D3/def2-SVP//PM6 dataset for SFM",
            theory="WB97X-D3/def2-SVP",
            data_root_dir=DEFAULT_DATA_ROOT_DIR,
            data_dir="raw/wb97xd3/1.0.0",
        ),
    ]

    def __init__(self, **kwargs):
        dataset_name = kwargs.pop("dataset_name", None) or "pubchemqc-b3lyp-pm6"
        super().__init__(dataset_name=dataset_name, **kwargs)
        if self.DISABLE_RDKIT_LOG:
            RDLogger.DisableLog("rdApp.*")
        assert isinstance(self.config, MolecularDatasetBuilderConfig)

    def _info(self):
        config: MolecularDatasetBuilderConfig = self.config
        features = Molecule.features()
        if config.infer_smiles_from_bonds:
            features["smiles_from_bonds"] = datasets.Value("string")
        if config.infer_smiles_from_coords:
            features["smiles_from_xyz"] = datasets.Value("string")
        if config.ogb_feature_smiles_column:
            features.update(OGB_FEATURES)
        return datasets.DatasetInfo(
            description=config.description,
            features=Features(features),
            homepage="https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_pm6_datasets.html",
            license="https://creativecommons.org/licenses/by/4.0/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name in ["b3lyp", "wb97xd3"]:
            self.raw_dataset = datasets.load_from_disk(
                dataset_path=self.config.data_dir
            )["train"]
            total_size = len(self.raw_dataset)
            num_proc = dl_manager.download_config.num_proc
            chunk_size = (total_size + num_proc - 1) // num_proc
            chucks = [
                (i, min(i + chunk_size, total_size))
                for i in range(0, total_size, chunk_size)
            ]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"shards": chucks},
                )
            ]
        else:
            if self.config.name == "raw-wb97xd3":
                logger.info(
                    "[yellow]It may take a while for preparing cid2idx mapping to start generating examples"
                )
                self.b3lyp = self.parents["b3lyp"]["train"]

            tars = sorted(glob.glob(os.path.join(self.config.data_dir, "*.tar.*")))
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"shards": tars},
                )
            ]

    def _generate_examples(self, shards):
        gen_examples_fn = (
            self._generate_raw_examples
            if self.config.name.startswith("raw-")
            else self._generate_sfm_examples
        )
        for key, exmaples in gen_examples_fn(shards):
            yield key, exmaples

    def _generate_sfm_examples(self, shards):
        config: MolecularDatasetBuilderConfig = self.config

        def _valid_example(example, smiles_match=False, allow_multi_frags=True):
            if (
                example["error"]
                or example["energy"] is None
                or example["energy"] >= 0.0
            ):
                return False
            if any([example[feat] is None for feat in OGB_FEATURES]):
                return False

            try:
                if smiles_match or not allow_multi_frags:
                    mol = Chem.RemoveHs(
                        Chem.MolFromSmiles(example[config.ogb_feature_smiles_column])
                    )

                if smiles_match:
                    ref_mol = Chem.RemoveHs(
                        Chem.MolFromSmiles(example["smiles_isomeric"])
                    )
                    if Chem.MolToSmiles(
                        ref_mol, isomericSmiles=False
                    ) != Chem.MolToSmiles(mol, isomericSmiles=False):
                        return False

                if not allow_multi_frags and len(Chem.GetMolFrags(mol)) > 1:
                    return False

                return True
            except:
                return False

        # match pubchem canonical smiles and don't allow multi-fragment molecules
        def filter_fn(x):
            return _valid_example(x, smiles_match=True, allow_multi_frags=False)

        for shard in shards:
            for i in range(shard[0], shard[1]):
                mol = self.raw_dataset[i]
                if filter_fn(mol):
                    yield mol["id"], mol

    def _generate_raw_examples(self, shards):
        config: MolecularDatasetBuilderConfig = self.config
        if config.name == "raw-wb97xd3":
            logger.debug("Preparing cid2idx mapping...")
            self.b3lyp_id2idx = {id: idx for idx, id in enumerate(self.b3lyp["id"])}

        def _b3lyp_example(jsonpath):
            mol = self.load_b3lyp_mol(jsonpath)
            example = asdict(mol)
            if mol.error:
                return mol.id, example

            try:
                infered_mols = {}
                if config.infer_smiles_from_bonds:
                    mw = mol.to_rdkit_mol(
                        use_bond=True, removeHs=False, raise_error=True
                    )
                    infered_mols["smiles_from_bonds"] = mw
                    example["smiles_from_bonds"] = Chem.MolToSmiles(mw)
                if config.infer_smiles_from_coords:
                    mw = mol.to_rdkit_mol(
                        use_bond=False, removeHs=False, raise_error=True
                    )
                    infered_mols["smiles_from_xyz"] = mw
                    example["smiles_from_xyz"] = Chem.MolToSmiles(mw)
            except Exception as ex:
                example["error"] = f"infer_smiles_error: {ex}"

            if config.ogb_feature_smiles_column:
                mw = infered_mols.get(config.ogb_feature_smiles_column)
                if mw:
                    try:
                        ogb_feat = mol2graph(mw)
                        assert ogb_feat["num_nodes"] == mol.num_atoms
                        example.update(ogb_feat)
                    except Exception as ex:
                        example["error"] = f"ogb_feature_error: {ex}"
            return mol.id, example

        def _lightaimd_example(jsonpath):
            mol = self.load_lightaimd_mol(jsonpath)
            ref = self.b3lyp[self.b3lyp_id2idx[mol.id]]

            mol.formula = ref["formula"] or mol.formula
            mol.smiles_isomeric = ref["smiles_isomeric"]
            mol.bonds = ref["bonds"]

            if not ref["error"]:
                try:
                    assert np.all(ref["atoms"] == mol.atoms)
                    assert ref["charge"] == mol.charge
                    assert ref["multiplicity"] == mol.multiplicity
                except Exception as ex:
                    mol.error = f"ref_data_mismatch: {ex}"

            example = asdict(mol)

            if config.infer_smiles_from_bonds:
                example["smiles_from_bonds"] = ref["smiles_from_bonds"]
            if config.infer_smiles_from_coords:
                example["smiles_from_xyz"] = ref["smiles_from_xyz"]
            if config.ogb_feature_smiles_column:
                for feat in OGB_FEATURES:
                    example[feat] = ref[feat]
            if not mol.error and ref["error"]:
                example["error"] = f"ref_data_error: {ref['error']}"
            return mol.id, example

        for tarpath in shards:
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
                name = os.path.basename(tarpath).split(".")[0]
                os.system(f"tar -xf {tarpath} -C {tmpdir}")
                for jsonpath in glob.glob(
                    os.path.join(tmpdir, name, "**/*.json"), recursive=True
                ):
                    if self.config.name == "raw-b3lyp":
                        yield _b3lyp_example(jsonpath)
                    else:
                        yield _lightaimd_example(jsonpath)

    def get_metadata(
        self, dataset: datasets.Dataset, num_proc: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        config: MolecularDatasetBuilderConfig = self.config
        if config.name not in ["b3lyp", "wb97xd3"]:
            return None

        def _batch_metadata(examples, idx, max_atomic_num: int = 128):
            XX, XY = 0, 0
            for atoms, energy in zip(examples["atoms"], examples["energy"]):
                elements, counts = np.unique(atoms, return_counts=True)
                if np.any(elements > max_atomic_num):
                    continue
                x = np.zeros(max_atomic_num)
                x[elements - 1] = counts
                x = x.reshape(-1, 1)
                XX += x @ x.T
                XY += x @ np.array([[energy]])

            return {
                "index": [idx],
                "id": [examples["id"]],
                "size": [examples["num_atoms"]],
                "XX": [XX],
                "XY": [XY],
            }

        max_atomic_num = 36 if config.name == "b3lyp" else 128
        batches = dataset.map(
            lambda examples, idx: _batch_metadata(
                examples, idx, max_atomic_num=max_atomic_num
            ),
            batched=True,
            with_indices=True,
            batch_size=10000,
            remove_columns=dataset.column_names,
            num_proc=num_proc or os.cpu_count(),
            keep_in_memory=True,
            desc="Calculating atomic reference energies",
        )

        logger.info("Ordering molecules by id and getting index and size")
        index, ids, size, XX, XY = [], [], [], 0, 0
        for batch in batches:
            index.append(batch["index"])
            ids.append(batch["id"])
            size.append(batch["size"])
            XX += np.array(batch["XX"])
            XY += np.array(batch["XY"])

        atomic_energies = (np.linalg.pinv(XX) @ XY).reshape(-1)
        # sort by molecule id and save (loc, id) as index
        index_reordered = np.argsort(np.concatenate(ids))
        index = np.concatenate(index)[index_reordered]
        size = np.concatenate(size)[index_reordered]

        return dict(index=index, size=size, atomic_energies=atomic_energies)

    @staticmethod
    def load_b3lyp_mol(jsonpath):
        cid = os.path.basename(os.path.dirname(jsonpath))
        with open(jsonpath) as f:
            try:
                js = json.load(f)
            except Exception as ex:
                return Molecule(id=cid, error=f"load_json_error: {ex}")

            try:
                assert int(cid) == js["pubchem"]["cid"]
                # fmt: off
                elements = np.array(js["pubchem"]["B3LYP@PM6"]["atoms"]["elements"]["number"])
                coords = np.array(js["pubchem"]["B3LYP@PM6"]["atoms"]["coords"]["3d"]).reshape(len(elements), 3).reshape(-1)
                props = js["pubchem"]["B3LYP@PM6"]["properties"]
                bonds = js["pubchem"]["B3LYP@PM6"].get("bonds")
                if bonds:
                    connections = np.array(js["pubchem"]["B3LYP@PM6"]["bonds"]["connections"]["index"]).reshape(-1, 2) - 1
                    orders = np.array(js["pubchem"]["B3LYP@PM6"]["bonds"]["order"]).reshape(-1, 1)
                    bonds = np.hstack([connections, orders]).reshape(-1)

                pubchem_mol = Chem.MolFromSmiles(js["pubchem"]["Isomeric SMILES"])
                return Molecule(
                    id=cid,
                    formula=js["pubchem"]["molecular formula"],
                    smiles_isomeric=Chem.MolToSmiles(pubchem_mol) if pubchem_mol else None,
                    atoms=elements,
                    num_atoms=len(elements),
                    charge=props["charge"],
                    multiplicity=props["multiplicity"],
                    bonds=bonds,
                    coords=coords,
                    energy=props["energy"]["total"],
                )
                # fmt: on
            except Exception as ex:
                return Molecule(id=cid, error=f"parse_data_error: {ex}")

    @staticmethod
    def load_lightaimd_mol(jsonpath):
        cid = os.path.basename(jsonpath).split(".")[0]
        with open(jsonpath) as f:
            try:
                js = json.load(f)
            except Exception as ex:
                return Molecule(id=cid, error=f"load_json_error: {ex}")

            try:
                if js["success"]:
                    inp, props = js["molecule"], js["properties"]
                    mol = dict(
                        energy=props["scf_total_energy"] * _HATREE_TO_EV,
                        forces=np.array(props["scf_total_gradient"])
                        * (-_HATREE_TO_EV)
                        / _BOHR_TO_ANGSTROM,
                    )
                else:
                    inp, error = js["input_data"]["molecule"], js["error"]
                    mol = dict(
                        error=f"{error['error_type']}: {error.get('error_message')}"
                    )

                elements = np.array(
                    [Chem.GetPeriodicTable().GetAtomicNumber(e) for e in inp["symbols"]]
                )
                mol.update(
                    dict(
                        id=cid,
                        formula=inp["name"],
                        atoms=elements,
                        num_atoms=len(elements),
                        charge=inp["molecular_charge"],
                        multiplicity=inp["molecular_multiplicity"],
                        coords=np.array(inp["geometry"]) * _BOHR_TO_ANGSTROM,
                    )
                )
                return Molecule(**mol)
            except Exception as ex:
                return Molecule(id=cid, error=f"parse_data_error: {ex}")
