# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import os
import pickle as pkl
import tempfile
from datetime import timedelta
from glob import glob
from typing import Any, AsyncGenerator, Optional

import aiofiles
import aiofiles.os
from ai4s.jobq import WorkSpecification
from qcelemental.models import Molecule
from qcelemental.util.serialization import JSONArrayEncoder
from rdkit.Chem import rdmolops
from rich.console import Console

from sfm.data.mol_data.utils.jobq import enqueue, launch_workers, setup_logging
from sfm.data.mol_data.utils.lightaimd import run_qcschema

logger = logging.getLogger("lightaimd.pcqcb3lyppm6")


class GEOMDFTWorkSpec(WorkSpecification):
    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        sku: str = "8xV100-IB",
        require_wfn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sku = sku
        self.processed = 0
        self.succeeded = 0
        self.require_wfn = require_wfn
        # self.cnt = 0

    async def list_tasks(self, path=None, force=False):
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".pickle"):
                    # self.cnt += 1
                    yield dict(path=os.path.join(root, file))

    def mol_to_xyz(self, mol):
        conf = mol.GetConformer()
        xyz_str = f"{mol.GetNumAtoms()}\n"
        xyz_str += "XYZ file generated from RDKit molecule\n"
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            xyz_str += f"{symbol} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"
        return xyz_str

    async def __call__(
        self,
        path: str,  # The full path of the *.pickle file
    ):
        print(f"Processing {path}")
        try:
            data = pkl.load(open(path, "rb"))
        except Exception as e:
            print(e)
            logger.error(f"Failed to load {path}")
            raise e

        multiplicity = 1  # NOTE: This is a hard-coded value
        mol_confs = [i["rd_mol"] for i in data.get("conformers", [])]
        for idx, mol in enumerate(mol_confs):
            try:
                charge = rdmolops.GetFormalCharge(mol)
                logger.info(
                    f"Processing {path}, frame: {idx}, charge: {charge}, multiplicity: {multiplicity}"
                )
                xyz = self.mol_to_xyz(mol)
                out = run_qcschema(
                    mol=Molecule.from_data(
                        xyz,
                        dtype="xyz",
                        molecular_charge=charge,
                        molecular_multiplicity=multiplicity,
                    ),
                    driver="gradient",
                    basis="def2-svp",
                    method="wb97x-d3",
                    disp="d3zero",
                    extras=None,
                    verbose=0,
                    sku=self.sku,
                    require_wfn=self.require_wfn,
                )

                dirpath = os.path.join(
                    self.output_dir, os.path.basename(path).replace(".pickle", "")
                )
                await aiofiles.os.makedirs(dirpath, exist_ok=True)
                async with aiofiles.open(
                    os.path.join(dirpath, f"{idx:03d}.json"), "w"
                ) as f:
                    await f.write(json.dumps(out, cls=JSONArrayEncoder, indent=2))
                error = getattr(out, "error", None)
            except Exception as e:
                error = e
                raise e
            finally:
                self.processed += 1
                if error:
                    logger.info(
                        "[{}/{}] {} {}".format(
                            self.succeeded, self.processed, path + "_" + str(idx), error
                        )
                    )
                else:
                    self.succeeded += 1
                    logger.info(
                        "[{}/{}] {} {:4} {:5} {:8.1f} {:6} {:15.6f} {}".format(
                            self.succeeded,
                            self.processed,
                            path + ":" + str(idx),
                            len(out.molecule.symbols),
                            out.properties.calcinfo_nbasis,
                            out.extras["program_time_seconds"],
                            out.properties.scf_iterations,
                            out.properties.return_energy,
                            out.molecule.name,
                        )
                    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # fmt:off
    parser.add_argument("--storage-account", default="sfmdataeastus2")
    parser.add_argument("--queue", default="geom-queue")
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--input-dir", default="/blob/data/geom_origin/rdkit_folder/drugs")
    parser.add_argument("--output-dir", default="/blob/data/geom_madft/drugs")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--sku", choices=["H100","A100","V100","4xA100","8xA100-IB","8xV100-IB"], default="A100")
    parser.add_argument("--require-wfn", action="store_true")
    # fmt:on
    args = parser.parse_args()
    print(args)

    logger.setLevel(logging.INFO)
    # setup_logging(
    #     # disable ai4s.jobq when run lightaimd
    #     skip_ai4s_jobq_trace=not args.enqueue,
    #     rich_tracebacks=True,
    #     console=Console(width=200),
    # )

    setup_logging(
        # disable ai4s.jobq when run lightaimd
        skip_ai4s_jobq_trace=False,
        skip_ai4s_task_trace=False,
        rich_tracebacks=True,
        console=Console(width=200),
    )
    workspec = GEOMDFTWorkSpec(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sku=args.sku,
        require_wfn=args.require_wfn,
    )

    if args.enqueue:
        asyncio.run(enqueue(args.storage_account, args.queue, workspec, dry_run=False))
    else:
        asyncio.run(
            launch_workers(
                args.storage_account,
                args.queue,
                workspec,
                time_limit=timedelta(days=100),
                visibility_timeout=timedelta(minutes=30),
                with_heartbeat=False,
                num_workers=1,
            )
        )
