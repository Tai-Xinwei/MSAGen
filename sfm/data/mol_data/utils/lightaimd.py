# -*- coding: utf-8 -*-
import tempfile
from typing import Optional

import qcengine as qcng
from madft.lightaimd import LightAIMDHarness
from madft.schema.job import run_qcschema as madft_run_qcschema
from qcelemental.models import AtomicInput, Molecule

COMPUTE = {
    "A100": {"gpus": 1, "cpus": 24, "numa_node": 1, "infiniband": False},
    "V100": {"gpus": 1, "cpus": 6, "numa_node": 1, "infiniband": False},
    "4xA100": {"gpus": 4, "cpus": 96, "numa_node": 4, "infiniband": False},
    "8xA100-IB": {"gpus": 8, "cpus": 96, "numa_node": 4, "infiniband": True},
    "8xV100-IB": {"gpus": 8, "cpus": 40, "numa_node": 2, "infiniband": True},
}


def ensure_lightaimd(
    sku: str,
    gpus: Optional[int] = None,
    force: bool = False,
):
    if "lightaimd" in qcng.list_available_programs():
        if not force:
            return qcng.get_program("lightaimd")
        qcng.unregister_program("lightaimd")

    compute = COMPUTE[sku]
    IB, gpu_count, cpu_count, numa_count = (
        compute["infiniband"],
        compute["gpus"],
        compute["cpus"],
        compute["numa_node"],
    )

    n_proc = min(gpus or gpu_count, gpu_count)
    mpirun = [
        "/opt/openmpi-4.1.4/bin/mpirun" if IB else "/opt/openmpi-4.0.3/bin/mpirun",
        "--allow-run-as-root",
        f"-np {n_proc}",
        f"--map-by ppr:{min(n_proc, gpu_count//numa_count)}:numa:pe={cpu_count//gpu_count}",
        "--bind-to core",
    ]
    qcng.register_program(LightAIMDHarness(dict(mpirun=" ".join(mpirun))))
    return qcng.get_program("lightaimd")


def run_qcschema(
    mol: Molecule,
    driver: str = "gradient",
    basis: str = "def2-svp",
    method: str = "b3lyp",
    disp: Optional[str] = None,
    grid_level: int = 3,
    scf_method: str = "rks",
    max_steps: int = 50,
    converge_threshold: float = 1e-8,
    eri_tolerance: float = 1e-12,
    require_wfn: bool = False,
    extras: Optional[dict] = None,
    workdir: Optional[str] = None,
    verbose: int = 4,
    gpus: Optional[int] = None,
    sku: str = "8xV100-IB",
    force_reinit: bool = False,
):
    lightaimd = ensure_lightaimd(sku, gpus, force=force_reinit)
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
        _workdir, _verbose = lightaimd.workdir, lightaimd.verbose
        lightaimd.workdir = workdir or tmpdir
        lightaimd.verbose = verbose
        try:
            return madft_run_qcschema(
                AtomicInput(
                    molecule=mol,
                    driver=driver,
                    model={"basis": basis, "method": method},
                    keywords={
                        "xcFunctional": {"gridLevel": grid_level},
                        "scf": {
                            "method": scf_method,
                            "maxSteps": max_steps,
                            "dispersion": disp,
                            "convergeThreshold": converge_threshold,
                            "eriTolerance": eri_tolerance,
                        },
                    },
                    protocols={"wavefunction": "all" if require_wfn else "none"},
                    extras=extras or {},
                )
            )
        finally:
            lightaimd.workdir = _workdir
            lightaimd.verbose = _verbose
