# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdDetermineBonds as rdDetermineBonds
from datasets import Sequence, Value


@dataclass
class Molecule:
    id: str
    formula: Optional[str] = None
    smiles_isomeric: Optional[str] = None
    atoms: Optional[np.ndarray] = None
    num_atoms: Optional[int] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    bonds: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    energy: Optional[float] = None
    forces: Optional[np.ndarray] = None
    error: Optional[str] = None

    @staticmethod
    def features() -> Dict[str, Any]:
        return {
            "id": Value("string"),
            "formula": Value("string"),
            "smiles_isomeric": Value("string"),
            "atoms": Sequence(Value("int8")),
            "num_atoms": Value("int32"),
            "charge": Value("int8"),
            "multiplicity": Value("int8"),
            "bonds": Sequence(Value("int32")),
            "coords": Sequence(Value("float32")),
            "energy": Value("float"),
            "forces": Sequence(Value("float32")),
            "error": Value("string"),
        }

    def to_rdkit_mol(
        self,
        use_bond: bool = True,
        check_charge: bool = True,
        raise_error: bool = False,
        removeHs: bool = False,
    ) -> Optional[Chem.Mol]:
        # NOTE: This function cannot differentiate isotopes
        try:
            assert self.coords is None or len(self.atoms) * 3 == len(self.coords)

            with Chem.RWMol() as mw:
                conf = Chem.Conformer()
                for i, atm in enumerate(self.atoms):
                    mw.AddAtom(Chem.Atom(int(atm)))
                    if self.coords is not None:
                        conf.SetAtomPosition(i, self.coords[i * 3 : i * 3 + 3])
                if conf:
                    mw.AddConformer(conf)

                if use_bond:
                    bonds = self.bonds.reshape(-1, 3) if self.bonds is not None else []
                    for i, j, order in bonds:
                        i, j = int(i), int(j)
                        if order == 1:
                            mw.AddBond(i, j, Chem.BondType.SINGLE)
                        elif order == 2:
                            mw.AddBond(i, j, Chem.BondType.DOUBLE)
                        elif order == 3:
                            mw.AddBond(i, j, Chem.BondType.TRIPLE)
                        elif order == 4:
                            mw.AddBond(i, j, Chem.BondType.AROMATIC)
                        else:
                            raise ValueError(
                                f"order={order}, which should be >=1 and <=4"
                            )
                    Chem.SanitizeMol(mw)
                else:
                    assert self.coords is not None
                    rdDetermineBonds.DetermineBonds(mw, charge=int(self.charge))

            assert not check_charge or Chem.GetFormalCharge(mw) == self.charge
            return Chem.RemoveHs(mw) if removeHs else mw
        except Exception:
            if raise_error:
                raise
            return None


OGB_FEATURES = {
    "num_nodes": Value("int32"),
    "node_feat": Sequence(Value("int8")),  # (num_nodes*9,)
    "edge_index": Sequence(Value("int16")),  # (2*num_edges,)
    "edge_feat": Sequence(Value("int8")),  # (num_edges*3,)
}


from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def mol2graph(mol: Chem.Mol) -> Dict[str, Any]:
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    g = dict()
    g["edge_index"] = edge_index
    g["edge_feat"] = edge_attr
    g["node_feat"] = x
    g["num_nodes"] = len(x)

    if (
        np.any(g["edge_index"] >= 65536)
        or np.any(g["edge_feat"] >= 256)
        or np.any(g["node_feat"] >= 256)
    ):
        raise ValueError("[GraphTooBigError]")

    g["node_feat"] = g["node_feat"].reshape(-1).astype(np.int8)
    g["edge_index"] = g["edge_index"].reshape(-1).astype(np.int16)
    g["edge_feat"] = g["edge_feat"].reshape(-1).astype(np.int8)

    return g
