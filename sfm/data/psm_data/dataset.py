# -*- coding: utf-8 -*-
import zlib
from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import bisect
import glob
import os
import pickle as pkl
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Union

import lmdb
import torch
from ase.io import read as ase_read
from numpy import dot
from numpy.linalg import norm
from rdkit import Chem
from sympy.utilities.iterables import multiset_permutations
from torch.utils.data import Subset
from torch_geometric.data import Data
from tqdm import tqdm

from sfm.data.data_utils import _filter_by_size_dynamic
from sfm.data.dataset import FoundationModelDataset, LMDBFoundationModelDataset
from sfm.data.mol_data import algos
from sfm.data.prot_data.util import bstr2obj
from sfm.data.psm_data.collator import collate_fn
from sfm.data.psm_data.crop import spatial_crop_psm
from sfm.data.psm_data.utils import (
    PM6_ATOM_ENERGY_OUTLIER_LIST,
    PM6_ATOM_REFERENCE_LIST,
    VOCAB,
    convert_to_single_emb,
    get_conv_variable_lin,
    get_data_defult_config,
    matrixtoblock_lin,
)
from sfm.logging import logger
from sfm.models.psm.psm_config import PSMConfig

_PT = Chem.GetPeriodicTable()


class MoleculeLMDBDataset(FoundationModelDataset):
    energy_mean: float = 0.0
    energy_std: float = 1.0
    energy_per_atom_mean: float = 0.0
    energy_per_atom_std: float = 1.0
    force_mean: float = 0.0  # force mean should always be 0.0 to keep equivariance
    force_std: float = 1.0

    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: str,
        keys: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
    ) -> None:
        assert lmdb_path, "LMDB path must be provided"
        self.lmdb_path = lmdb_path

        self.args = args
        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None

        atomic_ref_energy = self.atomic_ref_energies()
        ref_energy = np.ones(130) * 1e7
        ref_energy[np.arange(len(atomic_ref_energy)) + 1] = atomic_ref_energy
        ref_energy[np.abs(ref_energy) < 1e-6] = 1e7

        outliers = self.outlier_energy_atoms()
        if len(outliers) > 0:
            ref_energy[outliers] = 1e7
        atoms_kept = np.argwhere(ref_energy < 1e7).reshape(-1)
        logger.info(
            "Keep reference energy for {} atoms: [{}]",
            len(atoms_kept),
            ",".join(_PT.GetElementSymbol(int(a)) for a in atoms_kept if 0 < a < 119),
        )
        self.atomic_ref_energy_tensor = torch.tensor(ref_energy, dtype=torch.float64)

        if keys is not None:
            assert sizes is not None, "sizes must be provided with keys"
            self._env = lmdb.open(
                str(self.lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self._txn = self._env.begin(write=False)
            self._keys = keys
            self._sizes = sizes
        # else:
        #     self.filter_indices_by_size(
        #         indices=np.array(range(len(self.keys))),
        #         max_sizes=self.args.max_length - 2,
        #     )

        self.energy_per_atom_scale = getattr(
            self.args, "energy_per_atom_label_scale", None
        )
        if self.energy_per_atom_scale is not None:
            # Should not be used in general unless you know what you are doing
            logger.warning(
                "=== N O T E === Scaling energy_per_atom label by {}",
                self.energy_per_atom_scale,
            )

    @lru_cache(maxsize=1)
    def atomic_ref_energies(self):
        if not self.args.molecule_ref_energy_source:
            logger.warning("=== N O T E === Using DEPRECATED PM6_ATOM_REFERENCE_LIST")
            return np.array(PM6_ATOM_REFERENCE_LIST)

        with open(
            os.path.join(
                self.args.data_path,
                self.args.molecule_ref_energy_source,
                "metadata.pickle.gz",
            ),
            "rb",
        ) as f:
            metadata = pkl.loads(zlib.decompress(f.read()))
        return metadata["atomic_energies"]

    @lru_cache(maxsize=1)
    def outlier_energy_atoms(self):
        outliers = self.args.molecule_outlier_energy_atoms
        if not self.args.molecule_outlier_energy_atoms:
            return []

        if outliers == "DEPRECATED_PM6_ATOM_ENERGY_OUTLIER_LIST":
            logger.warning(
                "=== N O T E === Using DEPRECATED PM6_ATOM_ENERGY_OUTLIER_LIST"
            )
            return PM6_ATOM_ENERGY_OUTLIER_LIST
        else:
            return [int(a) for a in outliers.split(",")]

    def _ensure_init_db(self):
        if self._env is not None:
            return
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self._env.begin(write=False)
        metadata = bstr2obj(self._txn.get("metadata".encode()))
        self._keys = [str(key) for key in metadata["keys"]]
        self._sizes = metadata["size"] if "size" in metadata else metadata["sizes"]

    def _close_db(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._txn = None

    @property
    def env(self):
        self._ensure_init_db()
        return self._env

    @property
    def txn(self):
        self._ensure_init_db()
        return self._txn

    @property
    def sizes(self):
        self._ensure_init_db()
        return self._sizes

    @property
    def keys(self):
        self._ensure_init_db()
        return self._keys

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        max_validation_samples = getattr(self.args, "max_validation_samples", None)
        if max_validation_samples is not None:
            num_validation_samples = min(num_validation_samples, max_validation_samples)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in training_indices],
            sizes=[self._sizes[idx] for idx in training_indices],
        )
        dataset_val = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in validation_indices],
            sizes=[self._sizes[idx] for idx in validation_indices],
        )

        return dataset_train, dataset_val

    def raw(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        return pkl.loads(value)

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        data = self.raw(idx)

        # node features conversion for embedding, [6, 1, 0, 2] -> [6, 1 + 512, 0 + 512 x 2, 2 + 512 x 3]
        # note that in node_feat from ogb smiles2graph, hydrogen is represented by number 0 in the first dimension of node features
        # see https://github.com/snap-stanford/ogb/blob/f631af76359c9687b2fe60905557bbb241916258/ogb/utils/features.py#L60
        # +1 for the atomic number here to be consistent with other datasets
        data["node_feat"][:, 0] += 1
        data["node_feat"] = convert_to_single_emb(
            torch.tensor(data["node_feat"], dtype=torch.long)
        )
        if "pos" in data:
            coords = torch.tensor(data["pos"], dtype=torch.float64)
        elif "coords" in data:
            coords = torch.tensor(data["coords"], dtype=torch.float64)
        else:
            coords = torch.zeros((data["node_feat"].size()[0], 3), dtype=torch.float64)

        x = data["node_feat"]

        data["sample_type"] = 0
        data["token_type"] = data["node_feat"][
            :, 0
        ]  # token type only records the atomic numbers
        data["idx"] = idx

        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64)

        if "energy" in data or "total_energy" in data:
            total_energy = data["energy"] if "energy" in data else data["total_energy"]

            reference_energy = (
                torch.gather(self.atomic_ref_energy_tensor, 0, data["token_type"] - 1)
                .sum()
                .unsqueeze(0)
            )
            data["energy"] = torch.tensor(total_energy) - reference_energy
            data["energy_per_atom"] = (
                torch.tensor(total_energy) - reference_energy
            ) / data["num_atoms"]

            if self.energy_per_atom_scale is not None:
                data["energy_per_atom"] *= self.energy_per_atom_scale
        else:
            data["energy"] = torch.tensor([0.0], dtype=torch.float64)
            data["energy_per_atom"] = torch.tensor([0.0], dtype=torch.float64)

        has_forces = data.get("forces") is not None
        if has_forces:
            data["forces"] = torch.tensor(data["forces"], dtype=torch.float64)
        else:
            data["forces"] = torch.zeros((x.size()[0], 3), dtype=torch.float64)

        data["has_energy"] = torch.tensor([1], dtype=torch.bool)
        data["has_forces"] = torch.tensor([has_forces], dtype=torch.bool)
        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False

        return data

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data["edge_feat"], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(
            edge_attr
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)
        adj[edge_index[1, :], edge_index[0, :]] = True

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["node_feat"]

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            max_dist = np.amax(shortest_path_result)
            edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = torch.tensor(edge_input, dtype=torch.long)
            data["spatial_pos"] = spatial_pos

        return data

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "_sizes") and isinstance(self._sizes, np.ndarray):
                ignored = indices[self._sizes[indices] > max_sizes].tolist()
                indices = indices[self._sizes[indices] <= max_sizes]
            elif hasattr(self, "_sizes") and isinstance(self._sizes, list):
                sizes = np.array(self._sizes)
                ignored = indices[np.array(sizes[indices]) > max_sizes].tolist()
                indices = indices[np.array(sizes[indices]) <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self._sizes, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(indices, self._sizes, max_sizes)

        logger.warning(
            f"Removed {len(ignored)} examples from the {self.lmdb_path} because they are longer than {max_sizes}."
        )
        self._sizes = [self._sizes[idx] for idx in indices]
        self._keys = [self._keys[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.keys)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]

    def __getstate__(self):
        state = self.__dict__.copy()
        if state["_env"] is not None:
            state["_env"].close()
            del state["_env"]
            state["_env"] = None
        if state["_txn"] is not None:
            del state["_txn"]
            state["_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self._env.begin(write=False)


class PubChemQCB3lypPM6Dataset(MoleculeLMDBDataset):
    latest_version = "wb97xd3/1.0.0"

    def __init__(self, args, path, version=None, keys=None, sizes=None):
        path = os.path.normpath(path)
        if path.endswith("PubChemQC-B3LYP-PM6"):
            path = os.path.join(path, version or self.latest_version, "lmdb", "train")
        if not path.endswith("/lmdb/train"):
            # b3lyp/1.0.0 -> b3lyp/1.0.0/lmdb/train
            path = os.path.join(path, "lmdb", "train")
        assert os.path.exists(path)

        self.dataset_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
        super().__init__(args, path, keys=keys, sizes=sizes)

    def _ensure_init_db(self):
        if self._env is not None:
            return
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self._env.begin(write=False)
        with open(
            os.path.join(self.dataset_dir, "train", "metadata.pickle.gz"), "rb"
        ) as f:
            metadata = pkl.loads(zlib.decompress(f.read()))
        self._keys = [str(key) for key in metadata["index"]]
        self._sizes = metadata["size"]

    def raw(self, idx: Union[int, np.integer]) -> Data:
        x = super().raw(idx)
        x["node_feat"] = np.array(x["node_feat"]).reshape(-1, 9)
        x["coords"] = np.array(x["coords"]).reshape(-1, 3)
        assert x["num_nodes"] == len(x["node_feat"]) == len(x["coords"])

        forces = x.get("forces")
        if forces is not None:
            x["forces"] = np.array(forces).reshape(-1, 3)
            assert x["num_nodes"] == len(x["forces"])

        x["edge_index"] = np.array(x["edge_index"]).reshape(2, -1)
        x["edge_feat"] = np.array(x["edge_feat"]).reshape(-1, 3)
        assert x["edge_index"].shape[1] == x["edge_feat"].shape[0]

        return x


class PM6FullLMDBDataset(MoleculeLMDBDataset):
    latest_version = "20240527.1"
    energy_mean: float = -42774.16038176129
    energy_std: float = 25029.68158883449
    energy_per_atom_mean: float = -994.0920019593214
    energy_per_atom_std: float = 770.7496116135809

    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: str,
        version: Optional[str] = None,
        keys: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
    ):
        path = os.path.normpath(lmdb_path)
        if path.endswith("PubChemQC-B3LYP-PM6"):
            path = os.path.join(
                path, version or PM6FullLMDBDataset.latest_version, "full"
            )
        logger.warning("=== N O T E === Using DEPRECATED PM6FullLMDBDataset")
        super().__init__(args, path, keys=keys, sizes=sizes)


class PlainPM6FullLMDBDataset(PM6FullLMDBDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data["edge_feat"], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(
            edge_attr
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["node_feat"]

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        data["adj"] = adj
        data["attn_edge_type"] = attn_edge_type

        return data


class SmallMolDataset(FoundationModelDataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """

    energy = "energy"
    forces = "forces"

    def __init__(
        self,
        args: PSMConfig,
        path,
        data_name="pubchem5w",
        transforms=[],
        enable_hami=False,
        remove_init=False,
        remove_atomref_energy=True,
        Htoblock_otf=True,  ## on save H matrix, H to block is process in collate unifined for memory saving.
        basis="def2-tzvp",
        ## 1 kcal/mol = 0.0433634 eV, transform to eV by default here
    ):
        super(SmallMolDataset, self).__init__()
        if data_name.lower() == "pubchem5w":
            if basis != "def2-tzvp":
                raise ValueError(
                    "sorry, when using pubchem the basis should be def2-tzvp"
                )
        (
            self.atom_reference,
            self.system_ref,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
            self.has_energy,
            self.has_forces,
            self.is_pbc,
            self.unit,
        ) = get_data_defult_config(data_name)
        db_paths = []
        if isinstance(path, str):
            if path.endswith("lmdb"):
                db_paths.append(path)
            else:
                db_paths.extend(glob.glob(path + "/*.lmdb"))

        elif isinstance(path, list):
            for p in path:
                if p.endswith("lmdb"):
                    db_paths.append(p)
                else:
                    db_paths.extend(glob.glob(p + "/*.lmdb"))
        # print(db_paths)
        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        self.enable_hami = enable_hami
        self._keys, self.envs = [], []
        self.db_paths = sorted(db_paths)
        self.open_db()
        self.transforms = transforms  # unused
        self.remove_init = remove_init
        self.remove_atomref_energy = remove_atomref_energy
        self.conv, self.orbitals_ref, self.mask, self.chemical_symbols = (
            None,
            None,
            None,
            None,
        )
        self.Htoblock_otf = Htoblock_otf
        if self.enable_hami:
            self.conv, _, self.mask, _ = get_conv_variable_lin(basis)
        self.args = args

    def open_db(self):
        for db_path in self.db_paths:
            self.envs.append(self.connect_db(db_path))
            length = self.envs[-1].begin().get("length".encode("ascii"))
            if length is not None:
                length = pkl.loads(length)
            else:
                length = self.envs[-1].stat()["entries"]

            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        datapoint_pickled = (
            self.envs[db_idx]
            .begin()
            .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        )
        data_object = pkl.loads(datapoint_pickled)
        data_object.id = el_idx  # f"{db_idx}_{el_idx}"

        energy = data_object.energy
        # out["pyscf_energy"] = copy.deepcopy(energy.astype(np.float32))  # this is pyscf energy ground truth
        if self.remove_atomref_energy:
            unique, counts = np.unique(
                data_object.atomic_numbers.int().numpy(), return_counts=True
            )
            energy = energy - np.sum(self.atom_reference[unique] * counts)
            energy = torch.Tensor([energy - self.system_ref])

        out = {
            "sample_type": 0,
            "coords": data_object.pos,
            "forces": data_object.forces * self.unit,
            "num_atoms": data_object.pos.shape[0],
            "token_type": data_object.atomic_numbers.int().reshape(-1),
            "idx": idx,
            "edge_index": data_object.edge_index,
            "energy": energy.reshape(-1)
            * self.unit,  # this is used from model training, mean/ref is removed.
            "has_energy": torch.tensor([self.has_energy], dtype=torch.bool),
            "has_forces": torch.tensor([self.has_forces], dtype=torch.bool),
        }
        if self.is_pbc:
            cell_corner_pos_matrix = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
            out["token_type"] = torch.cat(
                [out["token_type"], torch.full([8], 128)], dim=-1
            )  # convert_to_single_emb(x)
            cell_corner_pos = torch.matmul(
                cell_corner_pos_matrix, data_object.cell.squeeze(dim=0).float()
            )
            out["coords"] = torch.cat(
                [out["coords"], cell_corner_pos], dim=0
            )  # expand pos with cell corners
            out["forces"] = torch.cat(
                [
                    torch.tensor(out["forces"].clone().detach(), dtype=torch.float32),
                    torch.zeros([8, 3], dtype=torch.float32),
                ],
                dim=0,
            )  # expand forces for cell corners

            out["cell"] = data_object.cell.squeeze(dim=0)
            out["pbc"] = torch.ones(3, dtype=torch.float32).bool()
            out["stress"] = torch.zeros(
                (3, 3), dtype=torch.float32, device=energy.device
            )
            out["cell_offsets"] = data_object.cell_offsets.numpy()

            out["energy_per_atom"] = out["energy"] / out["num_atoms"]

        else:
            out["cell"] = torch.zeros((3, 3), dtype=torch.float32)
            out["pbc"] = torch.zeros(3, dtype=torch.float32).bool()
            out["stress"] = torch.zeros(
                (3, 3), dtype=torch.float32, device=energy.device
            )

            out["energy_per_atom"] = out["energy"] / out["num_atoms"]

        if self.enable_hami:
            # out.update({"init_fock":data_object.init_fock.astype(np.float32)})
            if self.remove_init:
                data_object.fock = data_object.fock - data_object.init_fock
            if self.Htoblock_otf is True:
                out.update(
                    {
                        "buildblock_mask": self.mask,
                        "max_block_size": self.conv.max_block_size,
                        "fock": data_object.fock * self.unit,
                    }
                )
            else:
                diag, non_diag, diag_mask, non_diag_mask = None, None, None, None
                diag, non_diag, diag_mask, non_diag_mask = matrixtoblock_lin(
                    data_object.fock,
                    data_object.atomic_numbers,
                    self.mask,
                    self.conv.max_block_size,
                )
                out.update(
                    {
                        "diag_hamiltonian": diag * self.unit,
                        "non_diag_hamiltonian": non_diag * self.unit,
                        "diag_mask": diag_mask,
                        "non_diag_mask": non_diag_mask,
                    }
                )
            out.update({"init_fock": data_object.init_fock * self.unit})
            out.update({"s1e": data_object.s1e * self.unit})

        for key in out.keys():
            if key not in [
                "num_atoms",
                "token_type",
                "idx",
                "edge_index",
                "has_energy",
                "has_forces",
                "sample_type",
            ]:
                out[key] = torch.tensor(out[key], dtype=torch.float32)

        out = self.generate_2dgraphfeat(out)

        return out

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"].clone().detach(), dtype=torch.long)
        edge_attr = torch.ones((data["edge_index"].shape[1], 1), dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr + 1
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["token_type"].reshape(-1, 1)

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            max_dist = np.amax(shortest_path_result)
            edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = torch.tensor(edge_input, dtype=torch.long)
            data["spatial_pos"] = spatial_pos

        return data

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=32,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
            self.envs = []
        else:
            self.env.close()
            self.env = None

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = self.num_samples
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        dataset_train = Subset(self, training_indices)
        dataset_val = Subset(self, validation_indices)
        return dataset_train, dataset_val

    def split_train_valid_test(self, ratio_list: list, sort=False, shuffle=True):
        num_samples = self.num_samples

        indices = list(range(num_samples))
        # Shuffle the indices and split them into training and validation sets
        if shuffle:
            random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * ratio_list[1])
        num_test_samples = int(num_samples * ratio_list[2])
        num_training_samples = num_samples - num_validation_samples - num_test_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[
            num_training_samples : num_training_samples + num_validation_samples
        ]
        test_indices = indices[num_training_samples + num_validation_samples :]

        dataset_train = Subset(self, training_indices)
        dataset_val = Subset(self, validation_indices)
        dataset_test = Subset(self, test_indices)
        return dataset_train, dataset_val, dataset_test


class MatterSimDataset:
    def __init__(self, args: PSMConfig, data_path, split, atoms_list=None):
        self.data_lmdb = None
        self.data_txn = None
        self.index_to_key_name = []
        self.data_path = data_path
        self.add_unit_cell_virtual_node = args.add_unit_cell_virtual_node
        self.split = split
        if not os.path.exists(self.data_path):
            logger.warning(f"Path {self.data_path} does not exists.")
        if atoms_list is not None or (not os.path.isdir(self.data_path)):
            self.dataset_type = "single structure file"
            if atoms_list is not None:
                self.atoms_list = atoms_list
                # logger.info(
                #     "atoms_list provided, will use this for inference and not load data from disk. Setting split to None."
                # )
                self.split = None
            else:
                # strip trailing slashes
                self.data_path = self.data_path.rstrip("/")
                assert (
                    self.data_path.endswith("xyz")
                    or self.data_path.endswith("cif")
                    or self.data_path.endswith("POSCAR")
                    or self.data_path.endswith("CONTCAR")
                ), "Structure format not supported"
                logger.info(
                    "Assuming you are using this functionality for inference only, setting split to None."
                )
                self.split = None
                self.atoms_list = ase_read(self.data_path, index=":")
        else:
            self.dataset_type = "lmdb"
            lmdb_path = f"{self.data_path}/{self.split}"
            self.data_lmdb = lmdb.open(
                lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.data_txn = self.data_lmdb.begin(write=False)
            if self.data_txn.get("index_to_key_name".encode()) is None:
                self.index_to_key_name = []
                for key, val in self.data_txn.cursor():
                    self.index_to_key_name.append(key.decode())
            else:
                self.index_to_key_name = bstr2obj(
                    self.data_txn.get("index_to_key_name".encode())
                )
        self.args = args

        if args.psm_validation_mode and hasattr(args, "max_validation_samples"):
            if self.dataset_type == "single structure file":
                self.atoms_list = self.atoms_list[: args.max_validation_samples]
            else:
                self.index_to_key_name = self.index_to_key_name[
                    : self.args.max_validation_samples
                ]

    def switch_lattice_vectors(self, pbc, cell):
        # simple algorithm to switch lattice vectors so that they are more aligned with the initial lattice vectors
        initial_lattice_vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        best_permutation = None
        best_lattice_flip_sign = None
        max_cosine_sum = 0.0
        for permutation in multiset_permutations(np.arange(3)):
            cosine = 0.0
            lattice_flip_sign = []
            for i in range(3):
                index = permutation[i]
                original_lattice_vector = cell[index]
                initial_lattice_vector = initial_lattice_vectors[i]
                cosine_similarity = dot(
                    original_lattice_vector, initial_lattice_vector
                ) / (norm(original_lattice_vector) * norm(initial_lattice_vector))
                cosine += np.abs(cosine_similarity)
                lattice_flip_sign.append(-1.0 if cosine_similarity < 0.0 else 1.0)
            if cosine > max_cosine_sum:
                best_permutation = permutation
                max_cosine_sum = cosine
                best_lattice_flip_sign = lattice_flip_sign
        pbc = pbc[best_permutation]
        cell = cell[best_permutation] * np.array(best_lattice_flip_sign)[:, None]
        return pbc, cell

    # energy and std calculated over training part of the dataset
    @property
    def energy_mean(self):
        return -66.0996156928496

    @property
    def energy_std(self):
        return 102.91694201560776

    @property
    def energy_per_atom_mean(self):
        return -4.707989414326259

    @property
    def energy_per_atom_std(self):
        return 3.7324579639110653

    @property
    def force_mean(self):  # force mean should always be 0.0 to keep equivariance
        return 0.0

    @property
    def force_std(self):
        return 2.155674863803223

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.dataset_type == "single structure file":
            data = self.atoms_list[idx].todict()
        else:
            key = self.index_to_key_name[idx]
            data = pkl.loads(self.data_txn.get(key.encode()))

        if self.data_path.find("force-filtered") != -1:
            is_stable_periodic = True
        else:
            is_stable_periodic = False | self.add_unit_cell_virtual_node

        data["is_stable_periodic"] = is_stable_periodic

        numbers = data.pop(
            "numbers"
        )  # atomic numbers, starting from 1 for hydrogen atoms
        x = torch.tensor(numbers, dtype=torch.long)

        data["num_atoms"] = int(x.size()[0])

        if is_stable_periodic:
            # for structure generation task, we add virtual nodes for unit cell
            x = torch.cat([x, torch.full([8], 128)], dim=-1)

        positions = data.pop("positions")

        data["sample_type"] = 1
        data["coords"] = torch.tensor(positions, dtype=torch.float64)
        data["token_type"] = x
        data["idx"] = idx

        if is_stable_periodic:
            data["pbc"], data["cell"] = self.switch_lattice_vectors(
                data["pbc"], data["cell"]
            )

        data["cell"] = torch.tensor(data["cell"], dtype=torch.float64)
        data["pbc"] = torch.tensor(data["pbc"], dtype=torch.bool)

        if is_stable_periodic:
            cell_corner_pos_matrix = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            )

            cell_corner_pos = torch.matmul(cell_corner_pos_matrix, data["cell"])
            data["coords"] = torch.cat(
                [data["coords"], cell_corner_pos], dim=0
            )  # expand pos with cell corners

        if "forces" not in data and "energy" not in data and "stress" not in data:
            data["energy"] = torch.tensor([0.0], dtype=torch.float64)
            data["energy_per_atom"] = torch.tensor([0.0], dtype=torch.float64)
            data["stress"] = torch.zeros([3, 3], dtype=torch.float64)
            if is_stable_periodic:
                data["forces"] = torch.zeros(
                    [data["num_atoms"] + 8, 3], dtype=torch.float64
                )
            else:
                data["forces"] = torch.zeros(
                    [data["num_atoms"], 3], dtype=torch.float64
                )
        else:
            if is_stable_periodic:
                data["forces"] = torch.cat(
                    [
                        (
                            torch.tensor(data["forces"], dtype=torch.float64)
                            - self.force_mean
                        ),
                        torch.zeros([8, 3], dtype=torch.float64),
                    ],
                    dim=0,
                )  # expand forces for cell corners
            else:
                data["forces"] = torch.tensor(
                    data["forces"] - self.force_mean, dtype=torch.float64
                )

            data["energy"] = torch.tensor([(data["info"]["energy"] - self.energy_mean)])
            data["energy_per_atom"] = torch.tensor(
                [
                    (
                        (data["info"]["energy"] / float(data["num_atoms"]))
                        - self.energy_per_atom_mean
                    )
                ]
            )
            data["stress"] = torch.tensor(data["info"]["stress"], dtype=torch.float64)

        if self.args.rescale_loss_with_std:
            data["energy"] = data["energy"] / self.energy_std
            data["energy_per_atom"] = data["energy_per_atom"] / self.energy_per_atom_std
            data["forces"] = data["forces"] / self.force_std

        data["has_energy"] = torch.tensor([1], dtype=torch.bool)
        data["has_forces"] = torch.tensor([1], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        return data

    def generate_2dgraphfeat(self, data):
        N = data["token_type"].size()[-1]
        adj = torch.ones([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        in_degree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"].unsqueeze(-1),
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = in_degree

        if self.args.preprocess_2d_bond_features_with_cuda:
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result = (
                torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
            )
            edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = edge_input
            data["spatial_pos"] = spatial_pos

        return data

    def __len__(self):
        if self.dataset_type == "single structure file":
            return len(self.atoms_list)
        else:
            return len(self.index_to_key_name)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.dataset_type == "lmdb":
            if state["data_lmdb"] is not None:
                state["data_lmdb"].close()
                del state["data_lmdb"]
                state["data_lmdb"] = None
            if state["data_txn"] is not None:
                del state["data_txn"]
                state["data_txn"] = None
            return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.dataset_type == "lmdb":
            self.lmdb_path = f"{self.data_path}/{self.split}"
            self.data_lmdb = lmdb.open(
                self.lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.data_txn = self.data_lmdb.begin(write=False)


class AFDBLMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
        keys: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
    ):
        self.lmdb_path = lmdb_path
        self.args = args

        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None

        if keys is not None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self._txn = self._env.begin(write=False)
            self._keys = keys
            self._sizes = sizes

    def _init_db(self):
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self._sizes, self._keys = metadata["sizes"], metadata["keys"]

    def _close_db(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._txn = None

    @property
    def env(self):
        if self._env is None:
            self._init_db()
        return self._env

    @property
    def txn(self):
        if self._txn is None:
            self._init_db()
        return self._txn

    @property
    def sizes(self):
        if self._sizes is None:
            self._init_db()
        return self._sizes

    @property
    def keys(self):
        if self._keys is None:
            self._init_db()
        return self._keys

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)

        # random cut off the sequence data["aa"] to self.max_length
        if len(data["aa"]) > self.args.max_length:
            random_start = random.randint(0, len(data["aa"]) - self.args.max_length)
            data["aa"] = data["aa"][random_start : random_start + self.args.max_length]
            coords = data["pos"][
                random_start : random_start + self.args.max_length, 1, :
            ]
        else:
            # CA atom positions, assume all values are valid.
            coords = data["pos"][:, 1, :]

        # minus 1 due to add padding index=0 in collator
        x = torch.tensor([VOCAB[tok] - 1 for tok in data["aa"]], dtype=torch.int64)

        data["sample_type"] = 2
        data["token_type"] = x
        data["idx"] = idx

        coords = torch.tensor(coords, dtype=torch.float64)
        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)
        data["energy_per_atom"] = torch.tensor(
            [0.0], dtype=torch.float64, device=x.device
        )

        data["has_energy"] = torch.tensor([0], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False

        return data

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in training_indices],
            sizes=[self._sizes[idx] for idx in training_indices],
        )

        dataset_val = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in validation_indices],
            sizes=[self._sizes[idx] for idx in validation_indices],
        )

        return dataset_train, dataset_val

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "_sizes") and isinstance(self._sizes, np.ndarray):
                ignored = indices[self._sizes[indices] > max_sizes].tolist()
                indices = indices[self._sizes[indices] <= max_sizes]
            elif hasattr(self, "_sizes") and isinstance(self._sizes, list):
                sizes = np.array(self._sizes)
                ignored = indices[np.array(sizes[indices]) > max_sizes].tolist()
                indices = indices[np.array(sizes[indices]) <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self._sizes, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(indices, self._sizes, max_sizes)

        logger.warning(
            f"Removed {len(ignored)} examples from the AFDBLMDBDataset because they are longer than {max_sizes}."
        )
        self._sizes = [self._sizes[idx] for idx in indices]
        self._keys = [self._keys[idx] for idx in indices]

    # protein does not have 2dgraph, create one for mixing data
    def generate_2dgraphfeat(self, data):
        N = data["token_type"].shape[0]
        adj = torch.ones([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"].unsqueeze(-1),
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result = (
                torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
            )
            edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = edge_input
            data["spatial_pos"] = spatial_pos

        return data

    def __len__(self) -> int:
        return len(self.keys)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]

    def __getstate__(self):
        state = self.__dict__.copy()
        if state["_env"] is not None:
            state["_env"].close()
            del state["_env"]
            state["_env"] = None
        if state["_txn"] is not None:
            del state["_txn"]
            state["_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self._env.begin(write=False)


class PDBDataset(AFDBLMDBDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
        dataset_name: Optional[str] = None,
        keys: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
    ):
        # version = "20240101_snapshot.20240630_8fe6fe4b.subset_release_date_before_20200430.protein_chain.lmdb"
        version = "20240630_snapshot.20240711_dd3e1b69.subset_release_date_before_20200430.protein_chain.lmdb"
        testflag = "ProteinTest"
        if lmdb_path.find(version) == -1 and lmdb_path.find(testflag) == -1:
            lmdb_path = os.path.join(lmdb_path, version)
        super().__init__(args, lmdb_path, keys=keys, sizes=sizes)

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)

        # random cut off the sequence data["aa"] to self.max_length
        if len(data["aa"]) > self.args.max_length:
            random_start = random.randint(0, len(data["aa"]) - self.args.max_length)
            data["aa"] = data["aa"][random_start : random_start + self.args.max_length]
            coords = data["pos"][random_start : random_start + self.args.max_length, :]
        else:
            # CA atom positions, assume all values are valid.
            coords = data["pos"][:, :]

        # minus 1 due to add padding index=0 in collator
        x = torch.tensor([VOCAB[tok] - 1 for tok in data["aa"]], dtype=torch.int64)

        data["sample_type"] = 2
        data["token_type"] = x
        data["idx"] = idx
        data["key"] = key

        coords = torch.tensor(coords, dtype=torch.float64)

        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)
        data["energy_per_atom"] = torch.tensor(
            [0.0], dtype=torch.float64, device=x.device
        )

        data["has_energy"] = torch.tensor([0], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False

        return data


class UR50LMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
        keys: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
    ):
        self.lmdb_path = lmdb_path
        self.args = args

        self.vacab_mapping_dict = {
            0: 158,  # maps '<cls>' from vocab to self.vocab
            1: 0,  # maps '<pad>' from vocab to self.vocab
            2: 159,  # maps '<eos>' from vocab to self.vocab
            # 3: None,  # there is no equivalent of '<unk>' in self.vocab
            4: 130,  # maps 'L' from vocab to self.vocab
            5: 131,  # maps 'A' from vocab to self.vocab
            6: 132,  # maps 'G' from vocab to self.vocab
            7: 133,  # maps 'V' from vocab to self.vocab
            8: 134,  # maps 'S' from vocab to self.vocab
            9: 135,  # maps 'E' from vocab to self.vocab
            10: 136,  # maps 'R' from vocab to self.vocab
            11: 137,  # maps 'T' from vocab to self.vocab
            12: 138,  # maps 'I' from vocab to self.vocab
            13: 139,  # maps 'D' from vocab to self.vocab
            14: 140,  # maps 'P' from vocab to self.vocab
            15: 141,  # maps 'K' from vocab to self.vocab
            16: 142,  # maps 'Q' from vocab to self.vocab
            17: 143,  # maps 'N' from vocab to self.vocab
            18: 144,  # maps 'F' from vocab to self.vocab
            19: 145,  # maps 'Y' from vocab to self.vocab
            20: 146,  # maps 'M' from vocab to self.vocab
            21: 147,  # maps 'H' from vocab to self.vocab
            22: 148,  # maps 'W' from vocab to self.vocab
            23: 149,  # maps 'C' from vocab to self.vocab
            24: 150,  # maps 'X' from vocab to self.vocab
            25: 151,  # maps 'B' from vocab to self.vocab
            26: 152,  # maps 'U' from vocab to self.vocab
            27: 153,  # maps 'Z' from vocab to self.vocab
            28: 154,  # maps 'O' from vocab to self.vocab
            29: 156,  # maps '.' from vocab to self.vocab
            30: 155,  # maps '-' from vocab to self.vocab
            31: 157,  # maps '<mask>' from vocab to self.vocab
        }

        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None

        if keys is not None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self._txn = self._env.begin(write=False)
            self._keys = keys
            self._sizes = sizes

        # self.filter_indices_by_size(
        #     indices=np.array(range(len(self.keys))), max_sizes=self.args.max_length
        # )

    def _init_db(self):
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self._sizes, self._keys = metadata["lengths"], metadata["prot_accessions"]

    def _close_db(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._txn = None

    @property
    def env(self):
        if self._env is None:
            self._init_db()
        return self._env

    @property
    def txn(self):
        if self._txn is None:
            self._init_db()
        return self._txn

    @property
    def sizes(self):
        if self._sizes is None:
            self._init_db()
        return self._sizes

    @property
    def keys(self):
        if self._keys is None:
            self._init_db()
        return self._keys

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(f"{key}".encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = pkl.loads(value)
        data["aa"] = list(data["aa_seq"])

        # # random cut off the sequence data["aa"] to self.max_length
        # if len(data["aa"]) > self.args.max_length:
        #     random_start = random.randint(0, len(data["aa"]) - self.args.max_length)
        #     data["aa"] = data["aa"][random_start : random_start + self.args.max_length]

        x = torch.tensor(
            [self.vacab_mapping_dict[tok] - 1 for tok in data["aa"]], dtype=torch.int64
        )

        data["sample_type"] = 5
        data["token_type"] = x
        data["idx"] = idx

        data["coords"] = torch.zeros(
            (data["token_type"].size()[0], 3), dtype=torch.float64
        )
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.bool)
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)
        data["energy_per_atom"] = torch.tensor(
            [0.0], dtype=torch.float64, device=x.device
        )

        data["has_energy"] = torch.tensor([0], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False

        return data

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in training_indices],
            sizes=[self._sizes[idx] for idx in training_indices],
        )

        dataset_val = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in validation_indices],
            sizes=[self._sizes[idx] for idx in validation_indices],
        )

        return dataset_train, dataset_val

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "_sizes") and isinstance(self._sizes, np.ndarray):
                ignored = indices[self._sizes[indices] > max_sizes].tolist()
                indices = indices[self._sizes[indices] <= max_sizes]
            elif hasattr(self, "_sizes") and isinstance(self._sizes, list):
                sizes = np.array(self._sizes)
                ignored = indices[np.array(sizes[indices]) > max_sizes].tolist()
                indices = indices[np.array(sizes[indices]) <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self._sizes, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(indices, self._sizes, max_sizes)

        logger.warning(
            f"Removed {len(ignored)} examples from the AFDBLMDBDataset because they are longer than {max_sizes}."
        )
        self._sizes = [self._sizes[idx] for idx in indices]
        self._keys = [self._keys[idx] for idx in indices]

    # protein does not have 2dgraph, create one for mixing data
    def generate_2dgraphfeat(self, data):
        N = data["token_type"].shape[0]
        adj = torch.ones([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"].unsqueeze(-1),
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result = (
                torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
            )
            edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = edge_input
            data["spatial_pos"] = spatial_pos

        return data

    def __len__(self) -> int:
        return len(self.keys)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]

    def __getstate__(self):
        state = self.__dict__.copy()
        if state["_env"] is not None:
            state["_env"].close()
            del state["_env"]
            state["_env"] = None
        if state["_txn"] is not None:
            del state["_txn"]
            state["_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self._env.begin(write=False)


class PDBComplexDataset(AFDBLMDBDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
        keys: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
    ):
        # version = "20240630_snapshot.20240711_dd3e1b69.subset_release_date_before_20200430.ligand_protein_filteredNan.lmdb"
        version = "20240630_snapshot.20240714_2753ddc5.subset_release_date_before_20200430.ligand_protein.excludeNAs.removeHs.lmdb"
        # version = "posebusters-428structures-20240725-406c71b2.lmdb"
        self.crop_radius = args.crop_radius
        self.max_residue_num = args.max_residue_num

        self.iter_flag = True

        if lmdb_path.find(version) == -1:
            lmdb_path = os.path.join(lmdb_path, version)
        super().__init__(args, lmdb_path, keys=keys, sizes=sizes)

    def _init_db(self):
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        if self._keys is None:
            self._keys = metadata["keys"]

    def _crop_and_reconstruct_graph(self, data):
        polymer_chains = data["polymer_chains"]
        non_polymers = data["nonpoly_graphs"]

        polymer_chains_idxes = []
        for key in polymer_chains.keys():
            # # TODO: filter DNA/RNA chains, needs to be considered in the future
            # if np.any(polymer_chains[key]["restype"] == "N"):
            #     continue

            # some croped polymer has all Nan coords, so we need to avoid it
            if np.any(~np.isnan(polymer_chains[key]["center_coord"])):
                polymer_chains_idxes.append(key)
            # else:
            #     print(len(non_polymers), data["pdbid"], polymer_chains[key])

        # random generate crop center
        if len(non_polymers) > 0:
            if self.iter_flag:
                polymer_chains_idx = -1
                # pick random ligand from non-polymer to choose crop center
                center_ligand_idx = random.choice(range(len(non_polymers)))
                crop_center_ligand = non_polymers[center_ligand_idx]["node_coord"]
                keep_num = self.max_residue_num - len(crop_center_ligand)
                # pick random atom from ligand as crop center, there is nan in the ligand node_coord, avoid it
                crop_center_ligand = crop_center_ligand[
                    np.any(~np.isnan(crop_center_ligand), axis=-1)
                ]
                crop_center = random.choice(crop_center_ligand)
            else:
                center_ligand_idx = -1
                keep_num = self.max_residue_num
                # pick random polymer from non-polymer to choose crop center
                polymer_chains_idx = random.choice(range(len(polymer_chains_idxes)))
                chain_name = polymer_chains_idxes[polymer_chains_idx]
                crop_center_polymer = polymer_chains[chain_name]["center_coord"]
                # pick random atom from polymer as crop center, there is nan in the ligand node_coord, avoid it
                crop_center_polymer = crop_center_polymer[
                    np.any(~np.isnan(crop_center_polymer), axis=-1)
                ]
                crop_center = random.choice(crop_center_polymer)
            self.iter_flag = not self.iter_flag
        elif len(polymer_chains_idxes) > 0:
            center_ligand_idx = -1
            keep_num = self.max_residue_num
            # pick random polymer from non-polymer to choose crop center
            polymer_chains_idx = random.choice(range(len(polymer_chains_idxes)))
            chain_name = polymer_chains_idxes[polymer_chains_idx]
            crop_center_polymer = polymer_chains[chain_name]["center_coord"]
            # pick random atom from polymer as crop center, there is nan in the ligand node_coord, avoid it
            crop_center_polymer = crop_center_polymer[
                np.any(~np.isnan(crop_center_polymer), axis=-1)
            ]
            crop_center = random.choice(crop_center_polymer)
        else:
            raise ValueError(
                "No polymer or ligand in the complex, our model can't handle it"
            )

        # crop the complex and multimers
        cropped_chain_idxes_list, center_ligand_idx = spatial_crop_psm(
            polymer_chains,
            non_polymers,
            polymer_chains_idxes,
            self.crop_radius,
            center_ligand_idx,
            crop_center,
            keep_num=keep_num,
        )

        # reconstruct the graph
        data = self._reconstruct_graph(
            cropped_chain_idxes_list,
            center_ligand_idx,
            polymer_chains,
            non_polymers,
        )

        return data

    def _reconstruct_graph(
        self,
        cropped_chain_idxes_list,
        center_ligand_idx,
        polymer_chains,
        non_polymers,
    ):
        """
        Reconstruct the graph from the cropped complex and multimers
        """
        token_type = []
        coords = []
        position_ids = []
        start_position_ids = 0
        polymer_len = 0
        # reconstruct the polymer chains
        for idx, chain in enumerate(cropped_chain_idxes_list):
            chain_name = chain["chain_name"]
            cropped_chain_idxes = chain["cropped_chain_idxes"]
            chain = polymer_chains[chain_name]

            # rescontruct the residue sequence
            crop_chain = chain["seqres"][cropped_chain_idxes].tolist()
            if idx == len(cropped_chain_idxes_list) - 1:
                token_type.extend(crop_chain)
            else:
                token_type.extend(crop_chain + ["."])

            # rescontruct the coords
            if idx == len(cropped_chain_idxes_list) - 1:
                crop_coords = chain["center_coord"][cropped_chain_idxes]
            else:
                crop_coords = np.concatenate(
                    [
                        chain["center_coord"][cropped_chain_idxes],
                        np.zeros((1, 3)),
                    ],
                    axis=0,
                )
            coords.append(crop_coords)

            # build discontinuous position ids for rope
            if idx == len(cropped_chain_idxes_list) - 1:
                position_ids.extend(
                    range(start_position_ids, start_position_ids + len(crop_chain))
                )
                start_position_ids = start_position_ids + len(crop_chain) + 1000
                polymer_len += len(crop_chain)
            else:
                position_ids.extend(
                    range(start_position_ids, start_position_ids + len(crop_chain) + 1)
                )
                start_position_ids = start_position_ids + len(crop_chain) + 1 + 1000
                polymer_len += len(crop_chain) + 1

        if polymer_len > 0:
            x = [VOCAB[tok] - 1 for tok in token_type]
            node_feature = torch.zeros((len(x), 9), dtype=torch.int32)
        else:
            x = []

        # reconstruct the ligand
        if center_ligand_idx != -1:
            ligand = non_polymers[center_ligand_idx]

            # rescontruct the atom type of the ligand
            atom_ids = (ligand["node_feat"][:, 0] + 1).tolist()
            if len(x) > 0:
                x.extend([VOCAB["."] - 1] + atom_ids)
                pos = np.concatenate([np.zeros((1, 3)), ligand["node_coord"]], axis=0)
                # build position ids for ligand, but this may not used in the attention, just for length alignment
                position_ids.extend(
                    range(start_position_ids, start_position_ids + len(atom_ids) + 1)
                )
            else:
                x.extend(atom_ids)
                pos = ligand["node_coord"]
                # build position ids for ligand, but this may not used in the attention, just for length alignment
                position_ids.extend(
                    range(start_position_ids, start_position_ids + len(atom_ids))
                )

            # rescontruct the coords of the ligand
            coords.append(pos)

            if polymer_len == 0:
                node_feature = torch.from_numpy(ligand["node_feat"])
            else:
                node_feature = torch.cat(
                    [
                        node_feature,
                        torch.zeros((1, 9), dtype=torch.int32),
                        torch.from_numpy(ligand["node_feat"]),
                    ],
                    dim=0,
                )
                polymer_len += 1

            edge_index = ligand["edge_index"]
            # edge_attr = ligand["edge_feat"]
        else:
            edge_index = None

        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.int32)
        coords = torch.tensor(np.concatenate(coords, axis=0), dtype=torch.float64)
        position_ids = torch.tensor(position_ids, dtype=torch.int32)

        data = {
            "token_type": x,
            "coords": coords,
            "polymer_len": polymer_len,
            "position_ids": position_ids,
            "node_feature": node_feature,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

        return data

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        ori_data = bstr2obj(value)

        # crop and reconstruct the graph
        data = self._crop_and_reconstruct_graph(ori_data)

        data["idx"] = index
        N = data["token_type"].shape[0]
        data["num_atoms"] = N

        polymer_len = data["polymer_len"]

        data["x"] = data["token_type"]
        assert (
            data["node_feature"][:, 0].shape[0] == data["token_type"].shape[0]
        ), f"{data['node_feature'][:, 0].shape[0]} != {data['token_type'].shape[0]}"
        data["node_feature"][:, 0] = data["token_type"]
        data["node_attr"] = convert_to_single_emb(data["node_feature"].long())

        if data["edge_index"] is not None:
            # complex
            data["sample_type"] = 6
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[
                data["edge_index"][0, :] + polymer_len,
                data["edge_index"][1, :] + polymer_len,
            ] = True
            # allow interaction between protein and ligand, and protein and protein
            polymer_ligand_adj = torch.zeros([N, N], dtype=torch.bool)
            polymer_ligand_adj[:polymer_len] = True
            polymer_ligand_adj |= (
                polymer_ligand_adj.clone().T
            )  # torch disallow inplace operationS
            adj |= polymer_ligand_adj
        else:
            # multimers
            data["sample_type"] = 7
            adj = torch.ones([N, N], dtype=torch.bool)

        data["adj"] = adj

        # redundant, but for compatibility
        attn_edge_type = torch.zeros(
            [N, N, data["edge_attr"].size(-1)], dtype=torch.long
        )
        data["attn_edge_type"] = attn_edge_type
        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.bool)
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64)
        data["forces"] = torch.zeros((N, 3), dtype=torch.float64)
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["has_energy"] = torch.tensor([0], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)
        data["energy_per_atom"] = torch.tensor([0.0], dtype=torch.float64)
        data["energy"] = torch.tensor([0.0], dtype=torch.float64)
        data["in_degree"] = adj.long().sum(dim=1).view(-1)
        data["is_stable_periodic"] = False
        return data

    def __len__(self) -> int:
        return len(self.keys)

    def collate(self, batch: List[Data]) -> Data:
        result = collate_fn(batch)
        polymer_len = torch.tensor([i["polymer_len"] for i in batch])
        result.update(dict(polymer_len=polymer_len))
        return result

    def split_dataset(self, validation_ratio=0.01, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]
        # training_indices = indices
        # validation_indices = indices

        # Create training and validation datasets
        dataset_train = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in training_indices],
        )
        # dataset_train._sizes = [self._sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(
            self.args,
            self.lmdb_path,
            keys=[self._keys[idx] for idx in validation_indices],
        )
        # dataset_val._sizes = [self._sizes[idx] for idx in validation_indices]

        return dataset_train, dataset_val

    def filter_AllNan_indices(self):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        """
        indices = []
        for idx in range(len(self.keys)):
            key = self.keys[idx]
            value = self.txn.get(key.encode())
            if value is None:
                raise IndexError(f"Name {key} has no data in the dataset")
            ori_data = bstr2obj(value)
            polymer_chains = ori_data["polymer_chains"]
            polymer_chains_idxes = []
            for key in polymer_chains.keys():
                # # TODO: filter DNA/RNA chains, needs to be considered in the future
                # if np.any(polymer_chains[key]["restype"] == "N"):
                #     continue

                # some croped polymer has all Nan coords, so we need to avoid it
                if np.any(~np.isnan(polymer_chains[key]["center_coord"])):
                    polymer_chains_idxes.append(key)

            if len(polymer_chains_idxes) > 0:
                indices.append(idx)

        self._keys = [self.keys[idx] for idx in indices]
