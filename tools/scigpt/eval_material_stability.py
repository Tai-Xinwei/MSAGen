# -*- coding: utf-8 -*-
import os
import numpy as np
from ase import Atoms
from ase.io import read, write
from kinetics.forcefield.potential import Potential
from kinetics.forcefield.potential import DeepCalculator
from ase.optimize import FIRE, LBFGS, BFGSLineSearch
from ase.constraints import ExpCellFilter
from numpy import mean, sqrt, square
from numpy.linalg import norm
from pymatgen.entries.compatibility import (
    MaterialsProject2020Compatibility,
    Compatibility,
)
from pymatgen.entries.computed_entries import ComputedStructureEntry, ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import bulk
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
from glob import glob
from tqdm import tqdm
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--num_gpus", type=int, default=None)
args = parser.parse_args()


if args.num_gpus is None:
    args.num_gpus = torch.cuda.device_count()
if args.num_gpus == 1:
    device = "cuda:0"
else:
    device = f"cuda:{args.split}"

print(f"{args.num_gpus} GPUs found, using {device}")


def get_stability_score_simple(atoms, calc, fname):
    atoms_original = atoms.copy()
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    optimizer = FIRE(ecf, logfile="/dev/null")

    optimizer.run(fmax=0.1, steps=500)

    write(fname, atoms)

    atoms.get_positions() - atoms_original.get_positions()

    rmsd = norm(atoms.get_positions() - atoms_original.get_positions(), ord=2)

    pymatgen_structure = AseAtomsAdaptor.get_structure(atoms)
    potential_energy = atoms.get_potential_energy()
    computed_entry = ComputedStructureEntry(
        pymatgen_structure,
        potential_energy,
        parameters={
            "hubbards": {
                "Co": 3.32,
                "Cr": 3.7,
                "Fe": 5.3,
                "Mn": 3.9,
                "Mo": 4.38,
                "Ni": 6.2,
                "V": 3.25,
                "W": 6.2,
            },
            "run_type": "GGA+U",
        },
    )  # parameters= MPRelaxSet(pymatgen_structure).CONFIG["INCAR"]["LDAUU"])
    processed = MaterialsProject2020Compatibility(check_potcar=False).process_entry(
        computed_entry, verbose=False, clean=True
    )
    if processed == None:
        my_entry = computed_entry
        potential_energy = potential_energy
    else:
        my_entry = processed
        potential_energy = processed.energy

    # Extract elements from the entry's composition
    elements = [str(element) for element in my_entry.composition.elements]

    # Obtain all entries in the relevant chemical system using the Materials Project API
    with MPRester("ffxHEifcif9pfKRF7DRn70iPcQDAvG0k") as mpr:
        # Get all entries in the chemical system of the entry
        entries = mpr.get_entries_in_chemsys(elements)

    # Construct the phase diagram
    pd = PhaseDiagram(entries)

    # Get the energy above the hull for your material
    try:
        e_above_hull = pd.get_e_above_hull(my_entry)
    except:
        e_above_hull = 0

    return atoms, rmsd, e_above_hull




#potential = Potential.load(load_path=r"/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/mattersim/model_200epochs_3.5M.pth", device=device)
potential = Potential.load(load_path=r"/blob/renqian/SFM/threedimargen/data/materials_data/mattersim/model_200epochs_3.5M.pth", device=device)
calculator = DeepCalculator(potential, stress_weight=1 / 160, device=device)

#path = r"/msralaphilly2/ml-la/renqian/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/instructv1_mat_sample/*.cif"
path = r"/blob/renqian/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/instructv1_mat_sample/*.cif"
files = glob(path)


#with open(f"/msralaphilly2/ml-la/renqian/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/instructv1_mat_sample/stability_results_{args.split}.txt", "w") as f:
with open(f"/blob/renqian/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/instructv1_mat_sample/stability_results_{args.split}.txt", "w") as f:
    for i, fname in enumerate(tqdm(files)):
        if i % 4 == args.split:
            try:
                atoms = read(fname)
                new_f = fname.split(".cif")[0] + "_relaxed.cif"
                _, rmsd, e_above_hull = get_stability_score_simple(atoms=atoms, calc=calculator, fname=new_f)
            except:
                rmsd = "failed"
                e_above_hull = 0

            # Write the results to the text file
            f.write(f"For {fname}, RMSD: {rmsd} Angstrom, e_hull: {e_above_hull} eV/atom.\n")
            print(f"For {fname}, RMSD: {rmsd} Angstrom, e_hull: {e_above_hull} eV/atom.")
