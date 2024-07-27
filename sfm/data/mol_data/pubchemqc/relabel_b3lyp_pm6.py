# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import os
import tempfile
from datetime import timedelta
from glob import glob
from typing import Any, AsyncGenerator, Optional

import aiofiles
import aiofiles.os
from ai4s.jobq import WorkSpecification
from qcelemental.models import Molecule
from qcelemental.util.serialization import JSONArrayEncoder
from rich.console import Console

from sfm.data.mol_data.utils.jobq import enqueue, launch_workers, setup_logging
from sfm.data.mol_data.utils.lightaimd import run_qcschema

logger = logging.getLogger("lightaimd.pcqcb3lyppm6")


class PubChemQCB3LYPPM6WorkSpec(WorkSpecification):
    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        enqueue_worldsize: int = 1,
        enqueue_rank: int = 0,
        sku: str = "8xV100-IB",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dir = input_dir
        self.enqueue_worldsize = enqueue_worldsize
        self.enqueue_rank = enqueue_rank
        self.output_dir = output_dir
        self.sku = sku
        self.processed = 0
        self.succeeded = 0

    async def task_seeds(self) -> AsyncGenerator[Any, None]:
        assert self.input_dir is not None, "input_dir is required"
        for i, path in enumerate(sorted(glob(os.path.join(self.input_dir, "*")))):
            if i % self.enqueue_worldsize == self.enqueue_rank:
                yield path

    async def list_tasks(self, path=None, force=False):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
            dirname = os.path.basename(path).split(".")[0]
            logger.info(f"Extracting {dirname} to {tmpdir}")
            proc = await asyncio.create_subprocess_exec(
                *f"tar -xf {path} -C {tmpdir}".split()
            )
            retcode = await proc.wait()
            assert retcode == 0, f"Failed to extract {dirname}"

            errors = []
            for dirpath in sorted(glob(os.path.join(tmpdir, dirname, "*"))):
                cid = dirpath.rstrip("/").split("/")[-1]
                try:
                    kwargs = {"extras": {"cid": cid}}
                    xyz_path = os.path.join(dirpath, f"{cid}.B3LYP@PM6.S0.xyz")
                    async with aiofiles.open(xyz_path) as f:
                        kwargs["xyz"] = await f.read()
                    # NOTE: charge and multiplicity are always 0 and 1
                    kwargs["charge"] = 0
                    kwargs["multiplicity"] = 1
                    yield kwargs
                except:
                    errors.append(cid)
            if errors:
                logger.warning(f"Found {len(errors)} errors in {path}: {errors}")

    async def __call__(
        self,
        xyz: str,
        charge: int,
        multiplicity: int,
        extras: Optional[dict] = None,
    ):
        assert self.output_dir is not None, "output_dir is required"
        cid = extras["cid"]
        try:
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
                extras=extras,
                verbose=0,
                sku=self.sku,
            )

            part = int(cid) // 25000
            dirpath = os.path.join(
                self.output_dir, f"Compound_{part*25000:09}_{(part+1)*25000:09}"
            )
            await aiofiles.os.makedirs(dirpath, exist_ok=True)
            async with aiofiles.open(os.path.join(dirpath, f"{cid}.json"), "w") as f:
                await f.write(json.dumps(out, cls=JSONArrayEncoder))
            error = getattr(out, "error", None)
        except Exception as e:
            error = e
            raise e
        finally:
            self.processed += 1
            if error:
                logger.info(
                    "[{}/{}] {} {}".format(self.succeeded, self.processed, cid, error)
                )
            else:
                self.succeeded += 1
                logger.info(
                    "[{}/{}] {} {:4} {:5} {:8.1f} {:6} {:15.6f} {}".format(
                        self.succeeded,
                        self.processed,
                        cid,
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
    parser.add_argument("--queue", default="test-queue")
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--enqueue-worldsize", type=int, default=1)
    parser.add_argument("--enqueue-rank", type=int, default=0)
    parser.add_argument("--input-dir", default="/blob/psm/PubChemQC-B3LYP-PM6/raw/Compounds")
    parser.add_argument("--output-dir", default="/blob/psm/PubChemQC-B3LYP-PM6/lightaimd/test")
    parser.add_argument("--sku", choices=["A100","V100","4xA100","8xA100-IB","8xV100-IB"], default="8xV100-IB")
    # fmt:on
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    setup_logging(
        # disable ai4s.jobq when run lightaimd
        skip_ai4s_jobq_trace=not args.enqueue,
        rich_tracebacks=True,
        console=Console(width=200),
    )

    workspec = PubChemQCB3LYPPM6WorkSpec(
        input_dir=args.input_dir,
        enqueue_worldsize=args.enqueue_worldsize,
        enqueue_rank=args.enqueue_rank,
        output_dir=args.output_dir,
        sku=args.sku,
    )

    if args.enqueue:
        asyncio.run(enqueue(args.storage_account, args.queue, workspec, dry_run=False))
    else:
        asyncio.run(
            launch_workers(
                args.storage_account,
                args.queue,
                workspec,
                time_limit=timedelta(days=7),
                visibility_timeout=timedelta(minutes=30),
                with_heartbeat=False,
                num_workers=1,
            )
        )
