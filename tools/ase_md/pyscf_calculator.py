#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Garnet Chan <gkc1000@gmail.com>
# adapted by Stefan Heinen <heini.phys.chem@gmail.com>:
# - added forces
# - changed the ase_atoms_to_pyscf function
# - added an mp2 wrapper

'''
ASE package interface
'''

import numpy as np
from ase.calculators.calculator import Calculator
from ase.units import Hartree

from pyscf import gto, scf, grad
from pyscf.dft import KS

def ase_atoms_to_pyscf(ase_atoms):
    '''Convert ASE atoms to PySCF atom.

    Note: ASE atoms always use A.
    '''
    return [[ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]] for i in range(len(ase_atoms.get_positions()))]

atoms_from_ase = ase_atoms_to_pyscf


class PySCFCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='PySCF', atoms=None, scratch=None,
                 mf_class=KS, mf_dict={'xc': 'b3lyp', 'basis': 'def2-svp', 'grids_level': 3},
                 use_gpu=False,
                 **kwargs):
        """Construct PySCF-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'PySCF'.

        mfclass: PySCF mean-field class
        molcell: PySCF :Mole: or :Cell:
        """
        Calculator.__init__(self, restart=None, ignore_bad_restart_file=False,
                            label='PySCF', atoms=None, scratch=None, **kwargs)

        # TODO
        # This explicitly refers to "cell". How to refer
        # to both cell and mol together?
        self.mf_class = mf_class
        self.mf_dict = mf_dict
        self.use_gpu = use_gpu

    def __repr__(self):
        return 'PySCF'

    def __str__(self):
        return 'PySCF'

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms)
        calc_molcell = gto.Mole(basis=self.mf_dict['basis'],verbose=0)
        calc_molcell.atom = ase_atoms_to_pyscf(atoms)
        calc_molcell.build(None,None)
        mf = self.mf_class(calc_molcell)
        for key in self.mf_dict:
                mf.__dict__[key] = self.mf_dict[key]
        mf.grids.level = self.mf_dict['grids_level']
        if self.use_gpu:
            mf = mf.to_gpu()
        mf.scf(verbose=0)

        self.results['energy']= mf.e_tot/Hartree
        grad_instance = mf.nuc_grad_method()
        if self.use_gpu:
            grad_instance = grad_instance.to_gpu()
        grad = grad_instance.kernel()
        self.results['forces']=-1*grad/Hartree # convert forces to gradient (*-1) !!!!! for the NEB run
