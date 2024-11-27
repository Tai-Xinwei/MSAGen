# -*- coding: utf-8 -*-
import os
import sys
from typing import Any, Dict

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])
from dataclasses import dataclass, field

from ase import units
from ase.build import make_supercell
from ase.constraints import ExpCellFilter
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import FIRE

from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import seed_everything
from sfm.tasks.psm.ase_deep_calculator import DeepCalculator
from sfm.tasks.psm.ft_modules import PSM_FT_REGISTER
from sfm.utils import env_init


@dataclass
class Config(DistributedTrainConfig, PSMConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    ase_task: str = "NVT"  # NVT,NVE,relax
    work_dir: str = "./temp"
    ase_steps: int = 8000
    # relaxtion
    initial_cell_matrix: list = field(default_factory=lambda: [1, 1, 1])
    lower_deformation: int = 0
    upper_deformation: int = 0
    deformation_step: int = 1
    fmax: float = 0.01

    # MD
    dynamics_type: str = "Andersen"  # choices=["Langevin", "Andersen", "Nose-Hoover", "Berendsen", "Parrinello-Rahman"]
    timestep: float = 1.0  # fs
    temperature: float = 500.0  # K
    friction: float = 0.002  # for Langevin
    andersen_prob: float = 0.01  # for Andersen
    taut: float = 100.0  # for Nose-Hoover and Berendsen (fs)
    taup: float = 500.0  # for Berendsen (fs)
    pressure: float = 1.01325  # for Berendsen or Parrinello-Rahman (bar)
    pfactor: float = 1.0  # for Parrinello-Rahman(GPa)
    fixcm: bool = True
    logfile: str = "md.log"
    loginterval: int = 5
    trajectory: str = "md.traj"
    append_trajectory: bool = True


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


def ase_task(
    model: PSMModel,
    args: Config,
    extra_collate_fn=None,
):
    deep_calculator = DeepCalculator(
        model,
        args=args,
        atoms_type="periodic",
        extra_collate_fn=extra_collate_fn,
        device="cuda",
    )

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    task_atoms = ase_read(args.data_path)
    task_atoms = make_supercell(
        task_atoms,
        [
            [args.initial_cell_matrix[0], 0, 0],
            [0, args.initial_cell_matrix[1], 0],
            [0, 0, args.initial_cell_matrix[2]],
        ],
    )
    if args.ase_task == "relax":
        relaxed_atoms = task_atoms.copy()
        prev_atoms = relaxed_atoms.copy()
        for frac in tqdm(
            np.arange(
                args.lower_deformation,
                args.upper_deformation + args.deformation_step,
                args.deformation_step,
            )
        ):
            _atoms = prev_atoms
            pres = frac / 160.2176621

            _atoms.calc = deep_calculator
            ecf = ExpCellFilter(_atoms, scalar_pressure=pres)
            logger.info(f"before relaxation: {_atoms.get_positions()}")
            FIRE_logfile = os.path.join(args.work_dir, f"relaxation-{frac}.log")
            FIRE_trajfile = os.path.join(args.work_dir, f"trajectory-{frac}.traj")
            traj = Trajectory(FIRE_trajfile, "w", _atoms)
            qn = FIRE(ecf, logfile=FIRE_logfile)
            qn.attach(traj)
            qn.run(fmax=args.fmax, steps=args.ase_steps)

            logger.info(f"after relaxation: {_atoms.get_positions()}")

        prev_atoms = _atoms.copy()
    elif args.ase_task == "NVE":
        nve_atoms = task_atoms.copy()
        nve_atoms.calc = deep_calculator
        from ase.md.verlet import VelocityVerlet

        MaxwellBoltzmannDistribution(nve_atoms, temperature_K=args.temperature)
        ZeroRotation(nve_atoms)
        Stationary(nve_atoms)

        trajectory_path = os.path.join(args.work_dir, args.trajectory)
        logfile_path = os.path.join(args.work_dir, args.logfile)
        dyn = VelocityVerlet(
            nve_atoms,
            timestep=args.timestep * units.fs,
            trajectory=trajectory_path,
            logfile=logfile_path,
            loginterval=args.loginterval,
            append_trajectory=args.append_trajectory,
        )
        dyn.run(args.ase_steps)
    elif args.ase_task == "NVT":
        atoms = task_atoms.copy()
        atoms.calc = deep_calculator
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
        ZeroRotation(atoms)
        Stationary(atoms)
        dynamics_type = args.dynamics_type

        trajectory_path = os.path.join(args.work_dir, args.trajectory)
        logfile_path = os.path.join(args.work_dir, args.logfile)
        if dynamics_type == "Langevin":
            from ase.md.langevin import Langevin

            dyn = Langevin(
                atoms,
                timestep=args.timestep * units.fs,
                temperature_K=args.temperature,
                friction=args.friction / units.fs,
                fixcm=args.fixcm,
                trajectory=trajectory_path,
                logfile=logfile_path,
                loginterval=args.loginterval,
                append_trajectory=args.append_trajectory,
            )
        if dynamics_type == "Andersen":
            from ase.md.andersen import Andersen

            dyn = Andersen(
                atoms,
                timestep=args.timestep * units.fs,
                temperature_K=args.temperature,
                andersen_prob=args.andersen_prob,
                fixcm=args.fixcm,
                trajectory=trajectory_path,
                logfile=logfile_path,
                loginterval=args.loginterval,
                append_trajectory=args.append_trajectory,
            )
        elif dynamics_type == "Nose-Hoover":  # Nose-Hoover NVT is NPT with no pressure
            from ase.md.npt import NPT

            if np.all(atoms.cell == 0):  # if cell is not set
                atoms.set_cell(np.eye(3) * 100.0)
            dyn = NPT(
                atoms,
                timestep=args.timestep * units.fs,
                temperature_K=args.temperature,
                externalstress=0.0,
                ttime=args.taut * units.fs,
                trajectory=trajectory_path,
                logfile=logfile_path,
                loginterval=args.loginterval,
                append_trajectory=args.append_trajectory,
            )
        elif dynamics_type == "Berendsen":
            from ase.md.nvtberendsen import NVTBerendsen

            dyn = NVTBerendsen(
                atoms,
                timestep=args.timestep * units.fs,
                temperature_K=args.temperature,
                taut=args.taut * units.fs,
                fixcm=args.fixcm,
                trajectory=trajectory_path,
                logfile=logfile_path,
                loginterval=args.loginterval,
                append_trajectory=args.append_trajectory,
            )
        else:
            raise NotImplementedError(f"Unknown dynamics_type: {dynamics_type}")
        dyn.run(args.ase_steps)
    else:
        raise NotImplementedError(f"Unknown ase_task: {args.ase_task}")


@hydra.main(
    version_base=None, config_path="../../../config_file", config_name="config_psm"
)
def main(args: DictConfig) -> None:
    args = OmegaConf.to_object(args)
    assert isinstance(
        args, Config
    ), f"args must be an instance of Config! But it is {type(args)}"

    seed_everything(args.seed)
    env_init.set_env(args)

    finetune_module = None
    extra_collate_fn = None
    if args.psm_finetune_mode:
        finetune_module = PSM_FT_REGISTER[args.finetune_module](args)
        extra_collate_fn = finetune_module.update_batched_data

    model = PSMModel(
        args, loss_fn=DiffMAE3dCriterions, psm_finetune_head=finetune_module
    ).cuda()
    model.eval()

    ase_task(model, args, extra_collate_fn)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
