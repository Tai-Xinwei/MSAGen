# -*- coding: utf-8 -*-
import json
import os
import pickle
import tempfile
from glob import glob
from multiprocessing import Pool

import lmdb
import numpy as np
import rdkit.Chem as Chem
from tqdm import tqdm

from sfm.data.mol_data.utils.lmdb import obj2bstr
from sfm.data.mol_data.utils.molecule import mol2graph, xyz2mol
from sfm.logging import logger


def convert_b3lyp_json_to_mol(js, use_bond_orders=False, check_charge=True):
    error = None
    props = js["properties"]
    pubchem_isomeric, pubchem_canonical = js["smiles"], None
    pubchem_mol = Chem.MolFromSmiles(js["smiles"])
    if pubchem_mol is not None:
        pubchem_isomeric = Chem.MolToSmiles(pubchem_mol)
        pubchem_canonical = Chem.MolToSmiles(pubchem_mol, isomericSmiles=False)
    mol = dict(
        cid=js["pubchemcid"],
        formula=js["formula"],
        isomeric=None,
        canonical=None,
        pubchem_isomeric=pubchem_isomeric,
        pubchem_canonical=pubchem_canonical,
        atoms=[atm for atm in js["atoms"]["elements"]["number"]],
        coords=np.array(js["atoms"]["coords"]["3d"]).reshape(-1, 3),
        charge=props["charge"],
        multiplicity=props["multiplicity"],
        fragments=None,
        xc="B3LYP",
        basis="6-31g*",
        nao=props["orbitals"]["basis number"],
        energy=props["energy"]["total"],
        homo=props["energy"]["alpha"]["homo"],
        lumo=props["energy"]["alpha"]["lumo"],
        gap=props["energy"]["alpha"]["gap"],
    )

    try:
        if mol["energy"] >= 0 or mol["gap"] <= 0:
            error = ("EnergyError", props["energy"])
            return mol

        if len(mol["coords"]) != len(mol["atoms"]):
            error = ("CoordsError", f"len(coords)!={len(mol['atoms'])}")
            return mol

        try:
            bond_orders = None
            if use_bond_orders:
                bond_orders = []
                connections = np.array(js["bonds"]["connections"]["index"]).reshape(
                    -1, 2
                )
                for conn, order in zip(connections, js["bonds"]["order"]):
                    bond_orders.append((int(conn[0]), int(conn[1]), order))

            m = xyz2mol(
                mol["atoms"],
                mol["coords"],
                charge=mol["charge"],
                bond_orders=bond_orders,
                check_charge=check_charge,
            )
        except Exception as ex:
            error = ("xyz2molError", str(ex))
            return mol

        m_noHs = Chem.RemoveHs(m)
        mol["isomeric"] = Chem.MolToSmiles(m_noHs)
        mol["canonical"] = Chem.MolToSmiles(m_noHs, isomericSmiles=False)

        mol["fragments"] = len(Chem.GetMolFrags(m))
        m.RemoveAllConformers()
        mol.update(mol2graph(m))
        return mol
    except Exception as ex:
        error = (f"Exception[{type(ex).__name__}]", str(ex))
        return mol
    finally:
        if error:
            mol["error"] = error


def load_mols_from_tars(tar_files, work_dir, use_bond_orders=False):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    keys, sizes, mols, errors = [], [], [], []

    for tar_file in tqdm(tar_files):
        with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir:
            os.system(f"tar -xf {tar_file} -C {tmpdir}")
            for path in sorted(glob(f"{tmpdir}/*.json")):
                with open(path) as f:
                    mol = convert_b3lyp_json_to_mol(
                        json.load(f), use_bond_orders=use_bond_orders
                    )

                error = mol.get("error", None)
                if not error:
                    mols.append(mol)
                    keys.append(mol["cid"])
                    sizes.append(mol["num_nodes"])
                else:
                    errors.append({"cid": mol["cid"], "error": error})

    with lmdb.open(output_dir, map_size=1024**4).begin(write=True) as txn:
        for mol in mols:
            txn.put(str(mol["cid"]).encode(), pickle.dumps(mol))

    return keys, sizes, mols, errors


if __name__ == "__main__":
    data_dir = "/data/psm/PubChemQC-B3LYP/raw/json/"
    work_dir = "/mnt/b3lyp/"
    version = "20240618.1"
    use_bond_orders = False

    if use_bond_orders:
        logger.critical("=== N O T E === Using bond orders to build molecules")
        version += ".bond_orders"

    output_dir = f"/data/psm/PubChemQC-B3LYP/{version}/"
    n_procs = 20

    tars = glob(os.path.join(data_dir, "*.tar.gz"))
    chunck_size = (len(tars) + n_procs - 1) // n_procs
    chuncks = [tars[i : i + chunck_size] for i in range(0, len(tars), chunck_size)]

    def _load_mols(tar_files):
        return load_mols_from_tars(tar_files, work_dir, use_bond_orders=use_bond_orders)

    logger.info(f"Building dataset from {data_dir}")
    with Pool(n_procs) as pool:
        res = pool.map(_load_mols, chuncks, chunksize=1)

    keys, sizes, mols, errors = [], [], [], []
    for _keys, _sizes, _mols, _errors in res:
        keys.extend(_keys)
        sizes.extend(_sizes)
        mols.extend(_mols)
        errors.extend(_errors)

    counts = dict(
        total=len(mols) + len(errors),
        valid=len(mols),
        isomeric_match=0,
        canonical_match=0,
        charged=0,
        multi_fragments=0,
        errors={},
    )
    for mol in mols:
        if mol["isomeric"] == mol["pubchem_isomeric"]:
            counts["isomeric_match"] += 1
        if mol["canonical"] == mol["pubchem_canonical"]:
            counts["canonical_match"] += 1
        if mol["charge"] != 0:
            counts["charged"] += 1
        if mol["fragments"] > 1:
            counts["multi_fragments"] += 1
    for err in errors:
        err_type, _ = err["error"]
        counts["errors"][err_type] = counts["errors"].get(err_type, 0) + 1

    metadata = dict(version=version, keys=keys, sizes=sizes, counts=counts)
    with lmdb.open(output_dir, map_size=1024**4).begin(write=True) as txn:
        txn.put("metadata".encode(), obj2bstr(metadata))

    if len(errors) > 0:
        with open(os.path.join(output_dir, "errors.json"), "w") as f:
            json.dump(errors, f)

    logger.info(f"Dataset was built and saved in {output_dir}")
    valid, total = counts["valid"], counts["total"]
    logger.info(f"{valid}/{total} ({valid*100/total:.2f}%) molecules were extracted")
    print(
        json.dumps(
            {k: v for k, v in metadata.items() if k not in ["keys", "sizes"]}, indent=2
        )
    )
