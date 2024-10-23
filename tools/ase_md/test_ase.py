# -*- coding: utf-8 -*-
import numpy as np
import time
from ase import Atoms
from ase.build import molecule, make_supercell
from ase.calculators.emt import EMT
from ase.calculators.psi4 import Psi4
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units
from ase.io import write, read
from ase import visualize
from tools.ase_md.calculator import PSMCalculator
from tools.ase_md.pyscf_calculator import PySCFCalculator
from pyscf.dft.rks import RKS
from pyscf.dft.uks import UKS
from pyscf.dft import KS
from pyscf import gto
import matplotlib.pyplot as plt

from sfm.logging.loggers import get_logger

from argparse import ArgumentParser
import debugpy

parser = ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("--config-path", type=str, default="./")
parser.add_argument("--config-name", type=str, default="final_config")
parser.add_argument("--calculator", type=str, default="psm")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--name", type=str, default="water_md")

supported_calculators = ["psm", "pyscf", "emt"]

logger = get_logger()

def main(args):
    if args.debug:
        debugpy.listen(("0.0.0.0", 5678))
        logger.info("Waiting for debugger attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached!")

    if args.calculator not in supported_calculators:
        raise ValueError(f"Unsupported calculator: {args.calculator}. Supported calculators are: {supported_calculators}")

    # water = molecule('H2O')
    # water.set_cell([3, 3, 3])
    # water_box = water.repeat((4, 4, 4))

    # box_length = (len(water_box) / 64) ** (1/3) * 10
    # water_box.set_cell([box_length, box_length, box_length])
    # water_box.center()
    # write("water_box.pdb", water_box)

    atoms = read("tools/ase_md/asprin.xyz")

    if args.calculator == "psm":
        calc = PSMCalculator(config_path=args.config_path, config_name=args.config_name)
    elif args.calculator == "pyscf":
        calc = PySCFCalculator(use_gpu=True)
    elif args.calculator == "emt":
        calc = EMT()
    else:
        raise ValueError(f"Unsupported calculator: {args.calculator}")
    # calc = Psi4(atoms = atoms, method = 'b3lyp', memory = '500MB', basis = 'def2-svp')
    atoms.calc = calc
    # Set the momenta corresponding to T=300K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    time_step = 1 * units.fs  # 1 fs
    n_steps = args.steps
    dyn = VelocityVerlet(atoms, time_step)

    start_time = time.time()
    def printenergy(a=atoms, d=dyn):
        nonlocal start_time
        elapsed_time = time.time() - start_time
        epot = a.get_potential_energy() / len(a)  # eV/atom
        ekin = a.get_kinetic_energy() / len(a)  # eV/atom
        logger.info(f'Steps: {dyn.get_number_of_steps():04d}, Elapsed: {elapsed_time:.3f}, Energy per atom: Epot = {epot:.3f} eV, Ekin = {ekin:.3f} eV, '
            f'Etot = {epot + ekin:.3f} eV')
        start_time = time.time()
    dyn.attach(printenergy, interval=5)

    trajectory = Trajectory(f'{args.name}.traj', 'w', atoms)
    dyn.attach(trajectory.write, interval=1)

    logger.info("Starting MD simulation...")
    dyn.run(n_steps)
    logger.info("MD simulation completed.")

    traj = read(f'{args.name}.traj', index=':')
    energies = np.array([atoms.get_total_energy() for atoms in traj])
    logger.info(f"Mean energy: {np.mean(energies)}, std: {np.std(energies)}, relative std: {np.std(energies) / np.mean(energies)}")

    plt.plot(energies)
    plt.xlabel('Step')
    plt.ylabel('Total Energy (eV)')
    plt.title('Total Energy during MD Simulation')
    plt.savefig(f'{args.name}.png')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
