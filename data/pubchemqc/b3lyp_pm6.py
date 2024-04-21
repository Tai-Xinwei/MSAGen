# -*- coding: utf-8 -*-
import json
import logging
import os
import pickle
import re
import tempfile
from glob import glob
from typing import Any, Dict, List, Tuple

import lmdb
import numpy as np
import rdkit.Chem as Chem
from joblib import Parallel, delayed

from data.utils.lmdb import obj2bstr
from data.utils.molecule import mol2graph

logger = logging.getLogger(__name__)


def convert_b3lpy_pm6_mol(
    dirpath: str,
    remove_smiles_mismatch: bool = True,
    ignore_isomeric_mismatch: bool = True,
    remove_multi_frags: bool = True,
    compress_graph: bool = True,
) -> Dict[str, Any]:
    """Process a molecule in PubChemQC-B3LYP//PM6 dataset:
    xyz+bond_orders -> rdkit mol -> sanitization [-> check smiles] [-> check fragments]
    -> generate graph features -> return features and 3D coordinates.

    Args:
        - dirpath (str): path to the folder of the molecule
        - remove_smiles_mismatch (bool, optional): raise if the smiles of the molecule
          does not match the one from PubChem. This can ensure that the bond structure
          of the molecule is correct. Defaults to True.
        - ignore_isomeric_mismatch (bool, optional): ignore the smile mismatch due to
          the isomeric difference. Defaults to True.
        - remove_multi_frags (bool, optional): raise if there're multiple fragments.
          Defaults to True.
        - compress_graph (bool, optional): whether to compress the graph features.
          Defaults to True.

    Raises:
        - ValueError: raise ValueError if
          - charge != 0 (ChargeError)
          - multiplicity != 1 (MultiplicityError)
          - failed to build molecule from xyz file (XYZError)
          - failed to sanitize the molecule (SanitizeError)
          - (optional) mismatch PubChem simile (SmilesMatchError)
          - (optional) having multiple fragments (MultiFragsError)
          - (optional) feature graph is too big (GraphTooBig raised by `mol2graph`)

    Returns:
        Dict[str, Any]: a dictionary of features and labels
    """
    cid = dirpath.rstrip("/").split("/")[-1]
    with open(os.path.join(dirpath, f"{cid}.B3LYP@PM6.S0.json")) as f:
        js = json.load(f)

    props = js["pubchem"]["B3LYP@PM6"]["properties"]
    if props["charge"] != 0:
        raise ValueError(f"[ChargeError] actual charge={props['charge']}")
    if props["multiplicity"] != 1:
        raise ValueError(
            f"[MultiplicityError] actual multiplicity={props['multiplicity']}"
        )

    xyz = Chem.MolFromXYZFile(os.path.join(dirpath, f"{cid}.B3LYP@PM6.S0.xyz"))
    if xyz is None:
        raise ValueError("[XYZError] Failed to build mol from xyz file")

    bonds = js["pubchem"]["B3LYP@PM6"]["bonds"]
    connections = np.array(bonds["connections"]["index"]).reshape(-1, 2)

    with Chem.RWMol(xyz) as mol:
        for conn, order in zip(connections, bonds["order"]):
            mol.AddBond(int(conn[0]) - 1, int(conn[1]) - 1, Chem.BondType.values[order])

    try:
        Chem.SanitizeMol(mol)
    except Exception as ex:
        raise ValueError(f"[SanitizeError-{type(ex).__name__}] {ex}")

    if Chem.GetFormalCharge(mol) != props["charge"]:
        raise ValueError(
            f"[ChargeError] actual/expected charge: {Chem.GetFormalCharge(mol)} != {props['charge']}"
        )

    mol_noHs, pubchem_mol = Chem.RemoveHs(mol), Chem.MolFromSmiles(
        js["pubchem"]["Isomeric SMILES"]
    )
    smiles, pubchem_smiles = Chem.MolToSmiles(mol_noHs), Chem.MolToSmiles(pubchem_mol)
    if remove_smiles_mismatch and smiles != pubchem_smiles:
        isomeric_mismatch = Chem.MolToSmiles(
            mol_noHs, isomericSmiles=False
        ) == Chem.MolToSmiles(pubchem_mol, isomericSmiles=False)
        if not ignore_isomeric_mismatch or not isomeric_mismatch:
            reason = "IsomericMismatch" if isomeric_mismatch else "Unknown"
            raise ValueError(
                f"[SmilesMatchError-{reason}] actual/pubchem_smiles: {smiles} != {pubchem_smiles}"
            )

    frags = Chem.GetMolFrags(mol)
    if remove_multi_frags and len(frags) > 1:
        raise ValueError(f"[MultiFragsError] n_frags={len(frags)}, smiles={smiles}")

    energy = props["energy"]
    ret = dict(
        formula=js["pubchem"]["molecular formula"],
        smiles=smiles,
        pubchem_smiles=pubchem_smiles,
        coords=mol.GetConformer().GetPositions(),
        charge=int(props["charge"]),
        multiplicity=props["multiplicity"],
        xc="B3LYP",
        basis="6-31g*",
        nao=props["orbitals"]["basis number"],
        energy=energy["total"],
        alpha_homo=energy["alpha"]["homo"],
        alpha_lumo=energy["alpha"]["lumo"],
        alpha_gap=energy["alpha"]["gap"],
        beta_homo=energy["beta"]["homo"],
        beta_lumo=energy["beta"]["lumo"],
        beta_gap=energy["beta"]["gap"],
    )

    # remove conformers before generating graph features
    mol.RemoveAllConformers()
    ret.update(mol2graph(mol, compress_graph_or_raise=compress_graph))
    return ret


def process_b3lyp_pm6_tar(
    path: str, work_dir: str, process_mol_kwargs: Dict[str, Any]
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Dict[str, Any]]]:
    """Unzip a tar file and process all molecules in it.

    Args:
        - path (str): path to the tar file
        - work_dir (str): temporary work directory for unzip
        - process_mol_kwargs (Dict[str, Any]):
          kwargs for molecule processing. See `convert_b3lpy_pm6_mol`

    Returns:
        Tuple[List[Tuple[int, Dict[str, Any]]], List[Dict[str, Any]]]:
        list of molecules (tuples of cid and molecule) and list of errors
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    mols, errors = [], []
    with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir:
        os.system(f"tar -xf {path} -C {tmpdir}")
        unzip_dir = os.path.join(tmpdir, os.path.basename(path).split(".")[0])
        for dirpath in sorted(glob(f"{unzip_dir}/*")):
            cid = int(dirpath.rstrip("/").split("/")[-1])
            try:
                mol = convert_b3lpy_pm6_mol(dirpath, **process_mol_kwargs)
                mols.append((cid, mol))
            except Exception as ex:
                m = re.search(r"^\[([-\w]+)\]", str(ex))
                error_type = "UnknownError" if not m else m[1]
                errors.append(
                    {"cid": cid, "error_type": error_type, "message": str(ex)}
                )
    return mols, errors


def build_dataset(
    data_dir: str,
    output_dir: str,
    version: str,
    work_dir: str,
    process_mol_kwargs: Dict[str, Any],
    num_cpus: int = -1,
    verbose: int = 10,
) -> None:
    """Build the dataset from PubChemQC-B3LYP//PM6 data.

    Args:
        - data_dir (str): path to PubChemQC-B3LYP//PM6 data
        - output_dir (str): path to the output directory
        - version (str): version of the generated dataset
        - process_mol_kwargs (Dict[str, Any]):
          kwargs for molecule processing. See `convert_b3lpy_pm6_mol`
        - work_dir (str): temporary work directory for unzip the original data
        - num_cpus (int): number of cpus to use. Default to -1 (use all cpus)
        - verbose (int): verbose level for parallel jobs. Default to 10 (0 to disable)

    Raises:
        - ValueError: raise if the output directory is existing
    """
    if os.path.exists(output_dir):
        raise ValueError(
            f"Output directory {output_dir} has been alreay existed. "
            "Delete the existing one or specify a new output dir to avoid to corrupt your data."
        )
    os.makedirs(output_dir)

    logger.info(f"Building dataset from {data_dir}")

    errors, keys, sizes, counts = [], [], [], {}
    env = lmdb.open(output_dir, map_size=1024**4)
    for mols, errs in Parallel(n_jobs=num_cpus, return_as="generator", verbose=verbose)(
        delayed(process_b3lyp_pm6_tar)(
            path=path, work_dir=work_dir, process_mol_kwargs=process_mol_kwargs
        )
        for path in sorted(glob(os.path.join(data_dir, "*")))
    ):
        with env.begin(write=True) as txn:
            for cid, mol in mols:
                txn.put(str(cid).encode(), pickle.dumps(mol))
                keys.append(cid)
                sizes.append(mol["num_nodes"])
        for e in errs:
            err_type = e["error_type"]
            counts[err_type] = counts.setdefault(err_type, 0) + 1
        errors += errs

    counts["valid"] = len(keys)
    metadata = dict(
        version=version,
        config=process_mol_kwargs,
        keys=keys,
        sizes=sizes,
        counts=counts,
    )
    with env.begin(write=True) as txn:
        txn.put("metadata".encode(), obj2bstr(metadata))

    if len(errors) > 0:
        with open(os.path.join(output_dir, "errors.json"), "w") as f:
            json.dump(errors, f)

    logger.info(f"Dataset was built and saved in {output_dir}")
    total, valid = sum(counts.values()), counts["valid"]
    logger.info(f"{valid}/{total} ({valid*100/total:.2f}%) molecules were extracted")
    print(
        json.dumps(
            {k: v for k, v in metadata.items() if k not in ["keys", "sizes"]}, indent=2
        )
    )


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    arg_parser = ArgumentParser(
        description="Build dataset from PubChemQC-B3LYP//PM data",
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="Example: python data/pubchemqc/b3lyp_pm6.py"
        " --data-dir /data/psm/PubChemQC-B3LYP-PM6/raw/Compounds"
        " --output-dir /data/psm/PubChemQC-B3LYP-PM6"
        " --version 20240419.1"
        " --workdir /mnt/pm6",
    )
    # fmt: off
    arg_parser.add_argument("--data-dir", type=str, required=True, help="Path to data, e.g. /data/psm/PubChemQC-B3LYP-PM6/raw/Compounds")
    arg_parser.add_argument("--output-dir", type=str, required=True,
                            help="Output dir for the built dataset, e.g. /data/psm/PubChemQC-B3LYP-PM6. The string of '{version}/full' will be appended to the output path.")
    arg_parser.add_argument("--version", type=str, required=True, help="Version of the built dataset, e.g. 20240419.1")
    arg_parser.add_argument("--workdir", type=str, required=True, help="Temporary workdir for unzip the raw data.")
    arg_parser.add_argument("--no-remove-smiles-mismatch", action="store_true", help="Remove molecule if smiles does not match")
    arg_parser.add_argument("--no-ignore-isomeric-mismatch", action="store_true", help="Keep a molecule if smiles mismatch only due to isomeric difference")
    arg_parser.add_argument("--no-remove-multi-frags", action="store_true", help="Remove a molecule if has multiple fragments")
    arg_parser.add_argument("--no-compress-graph", action="store_true", help="Compress graph features. Remove if the graph is too big.")
    arg_parser.add_argument("--cpus", type=int, default=-1, help="Number of cpus to use. -1 to use all")
    # fmt: on

    args = arg_parser.parse_args()
    config = dict(
        remove_smiles_mismatch=not args.no_remove_smiles_mismatch,
        ignore_isomeric_mismatch=not args.no_ignore_isomeric_mismatch,
        remove_multi_frags=not args.no_remove_multi_frags,
        compress_graph=not args.no_compress_graph,
    )

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] - %(message)s", level="INFO"
    )

    build_dataset(
        data_dir=args.data_dir,
        output_dir=os.path.join(args.output_dir, args.version, "full"),
        version=args.version,
        work_dir=args.workdir,
        process_mol_kwargs=config,
        num_cpus=args.cpus,
    )
