# -*- coding: utf-8 -*-
from argparse import Namespace

import numpy as np
import torch
from pyscf import dft, gto, scf

HATREE_TO_KCAL = 627.5096


def transform_h_into_pyscf(hamiltonian: np.ndarray, mol: gto.Mole) -> np.ndarray:
    """
    Transforms the given Hamiltonian matrix into the PySCF format based on the atomic orbital (AO) type order.

    Args:
        hamiltonian (np.ndarray): The input Hamiltonian matrix.
        mol (gto.Mole): The PySCF Mole object representing the molecular system.

    Returns:
        np.ndarray: The transformed Hamiltonian matrix in the PySCF format.
    """
    # Get ao type list, format [atom_idx, atom_type, ao_type]
    ao_type_list = mol.ao_labels()
    order_list = []
    for idx, (_, _, ao_type) in enumerate(ao_type_list):
        # for p orbitals, the order is px, py, pz, which means the order should transform
        # from [0, 1, 2], to [2, 0, 1], thus [+2, -1, -1]
        if "px" in ao_type:
            order_list.append(idx + 2)
        elif "py" in ao_type:
            order_list.append(idx - 1)
        elif "pz" in ao_type:
            order_list.append(idx - 1)
        else:
            order_list.append(idx)

    # Transform hamiltonian
    hamiltonian_pyscf = hamiltonian[..., order_list, :]
    hamiltonian_pyscf = hamiltonian_pyscf[..., :, order_list]

    return hamiltonian_pyscf


def get_pyscf_obj_from_dataset(
    pos, atomic_numbers, basis: str = "def2-svp", xc: str = "b3lyp5", gpu=False
):
    """
    Get the PySCF Mole and KS objects from a dataset.

    Args:
        data (dict): The dataset containing the molecular data.
        idx (int): The index of the molecular data to retrieve.
        basis (str, optional): The basis set to use. Defaults to "def2-svp".
        xc (str, optional): The exchange-correlation functional to use. Defaults to "b3lyp5".
        gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.

    Returns:
        tuple: A tuple containing the PySCF Mole and KS objects.

    """

    mol = gto.Mole()
    mol.atom = "".join(
        [
            f"{atomic_numbers[i]} {pos[i][0]} {pos[i][1]} {pos[i][2]}\n"
            for i in range(len(atomic_numbers))
        ]
    )
    mol.basis = basis
    mol.verbose = 1
    mol.build()
    mf = dft.KS(mol, xc=xc)
    if gpu:
        try:
            from madft.cuda_factory import CUDAFactory

            factory = CUDAFactory()
            mf = factory.generate_cuda_instance(mf)
        except:
            print("CUDA is not available, falling back to CPU")
    return mol, mf


def get_psycf_obj_from_xyz(
    file_name: str, basis: str = "def2-svp", xc: str = "b3lyp5", gpu=False
):
    """
    Create a PySCF Mole and DFT object from an XYZ file.

    Args:
        file_name (str): The path to the XYZ file.
        basis (str, optional): The basis set to use. Defaults to 'def2-svp'.
        xc (str, optional): The exchange-correlation functional to use. Defaults to 'b3lyp5'.
        gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.

    Returns:
        tuple: A tuple containing the PySCF Mole object and the DFT object.
    """
    with open(file_name, "r") as f:
        lines = f.readlines()

    num_atoms = int(lines[0])
    charge, multiplicity = map(int, lines[1].split())

    atom_lines = lines[2 : 2 + num_atoms]
    atom_info = [line.split() for line in atom_lines]

    mol = gto.Mole()
    mol.atom = "".join(
        [f"{info[0]} {info[1]} {info[2]} {info[3]}\n" for info in atom_info]
    )
    mol.basis = basis
    mol.verbose = 4
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.build()
    mf = dft.KS(mol, xc=xc)
    if gpu:
        try:
            from madft.cuda_factory import CUDAFactory

            factory = CUDAFactory()
            mf = factory.generate_cuda_instance(mf)
        except:
            print("CUDA is not available, falling back to CPU")
    return mol, mf


def get_hamiltonian_by_model(
    mf: scf.RHF, model: torch.nn.Module, data: dict = None, transform_h=False
):
    """
    Calculates the Hamiltonian matrix for a given mean-field object and model.

    Args:
        mf (scf.RHF): The mean-field object.
        model (torch.nn.Module): The model used for prediction.
        data (dict, optional): Additional data for the model. Defaults to None.
        transform_h (bool, optional): Whether to transform the Hamiltonian matrix into pyscf order.
            Defaults to False.

    Returns:
        numpy.ndarray: The Hamiltonian matrix.
    """
    dm0 = mf.init_guess_by_minao()
    h = mf.get_fock(dm=dm0)
    if not data:
        mol = mf.mol
        coordinates = mol.atom_coords(unit="Angstrom")
        atom_nums = mol.atom_charges()
        data = Namespace(
            pos=coordinates.reshape(1, -1, 3), atomic_numbers=atom_nums.reshape(1, -1)
        )
    pred_h = (
        model.forward(data, full_hami=True)["pred_hamiltonian"].detach().cpu().numpy()
    )
    if transform_h:
        pred_h = transform_h_into_pyscf(pred_h)
    h += pred_h

    return h


class SCFCallback:
    def __init__(self):
        self.iter_count = 0

    def __call__(self, envs):
        self.iter_count += 1

    def count(self):
        return self.iter_count


def fock_to_dm(mf: scf.RHF, fock: np.ndarray, s1e: np.ndarray = None):
    if s1e is None:
        s1e = mf.get_ovlp()
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return dm


def run_scf(mf: scf.RHF, model: torch.nn.Module = None, data: dict = None):
    """
    Calculates the energy for a given mean-field object and model.

    Args:
        mf (scf.RHF): The mean-field object.
        model (torch.nn.Module optional): The model used for prediction.
        data (dict, optional): Additional data for the model. Defaults to None.

    Returns:
        (float, int): The energy and the iteration step.
    """
    if model:
        h = get_hamiltonian_by_model(mf, model, data)
        mf.init_guess = fock_to_dm(mf, h)
    mf.callback = SCFCallback()
    mf.conv_tol = 1e-8
    mf.kernel()
    return mf.e_tot, mf.callback.count()


def get_energy_from_h(mf: scf.RHF, h: np.ndarray):
    """
    Calculates the energy for a given mean-field object and Hamiltonian matrix.

    Args:
        mf (scf.RHF): The mean-field object.
        h (np.ndarray): The Hamiltonian matrix.

    Returns:
        float: The energy.
    """
    dm = fock_to_dm(mf, h)
    e_tot = mf.energy_tot(dm=dm)
    return e_tot


def get_energy_by_model(mf: scf.RHF, model: torch.nn.Module, data: dict = None):
    """
    Calculates the energy for a given mean-field object and model.

    Args:
        mf (scf.RHF): The mean-field object.
        model (torch.nn.Module): The model used for prediction.
        data (dict, optional): Additional data for the model. Defaults to None.

    Returns:
        float: The energy.
    """
    h = get_hamiltonian_by_model(mf, model, data)
    return get_energy_from_h(mf, h)


def get_homo_lumo_from_h(mf: scf.RHF, h: np.ndarray, s1e: np.ndarray = None):
    if s1e is None:
        s1e = mf.get_ovlp()
    mo_energy, _ = mf.eig(h, s1e)
    e_idx = np.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nocc = mf.mol.nelectron // 2
    homo, lumo = e_sort[nocc - 1], e_sort[nocc]
    return homo, lumo


def get_homo_lumo_by_model(
    mf: scf.RHF, model: torch.nn.Module, data: dict = None, s1e: np.ndarray = None
):
    """
    Calculate the highest occupied molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO)
    energies using a given model and PySCF mean-field object.

    Args:
        mf (scf.RHF): PySCF mean-field object.
        model (torch.nn.Module): Neural network model.
        data (dict, optional): Additional data for the model. Defaults to None.
        s1e (np.ndarray, optional): Overlap matrix. Defaults to None.

    Returns:
        float: HOMO energy.
        float: LUMO energy.
    """
    h = get_hamiltonian_by_model(mf, model, data)
    homo, lumo = get_homo_lumo_from_h(mf, h, s1e)
    return homo, lumo


# TODO:
# 1. Init Model class from arguments and them load model from checkout point
# 2. Add energy check in the test step.

# Add test for energy and homo-lumo
