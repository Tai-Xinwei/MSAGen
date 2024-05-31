# -*- coding: utf-8 -*-
import sys
import unittest
from io import StringIO

import crab
import numpy as np
import torch


class TestProtein(unittest.TestCase):
    single_chain_protein = "sfm/data/prot_data/data/4REL_A.cif"
    multi_chain_protein_cif = "sfm/data/prot_data/data/1WBK.cif"
    multi_chain_protein_pdb = "sfm/data/prot_data/data/1WBK.pdb"

    def test_from_file_single(self):
        obj = crab.Structure.from_file(self.single_chain_protein)
        self.assertIsNotNone(obj)

    def test_from_file_multimer_pdb(self):
        obj = crab.Structure.from_file(self.multi_chain_protein_pdb)
        self.assertIsNotNone(obj)

    def test_to_CRAB_single(self):
        obj = crab.Structure.from_file(self.single_chain_protein)
        C, R, A = obj.to_CRAB()
        self.assertIsNotNone(C)
        self.assertIsNotNone(R)
        self.assertIsNotNone(A)
        self.assertEqual(len(C), len(R), "First dim of C and R should be equal")
        self.assertEqual(len(C), len(A), "First dim of C and A should be equal")
        self.assertEqual(
            len(A.shape),
            3,
            f"A should have a shape of (N_res, N_atom, 3), but got shape {A.shape}",
        )
        self.assertEqual(
            A.shape[-1],
            3,
            f"A should have a shape of (N_res, N_atom, 3), but got shape {A.shape}",
        )
        self.assertEqual(
            len(np.unique(C)), 1, "Single chain protein should have 1 chain"
        )

    def test_to_CRAB_multimer_pdb(self):
        obj = crab.Structure.from_file(self.multi_chain_protein_pdb)
        C, R, A = obj.to_CRAB()
        self.assertIsNotNone(C)
        self.assertIsNotNone(R)
        self.assertIsNotNone(A)
        self.assertEqual(len(C), len(R), "First dim of C and R should be equal")
        self.assertEqual(len(C), len(A), "First dim of C and A should be equal")
        self.assertEqual(
            len(A.shape),
            3,
            f"A should have a shape of (N_res, N_atom, 3), but got shape {A.shape}",
        )
        self.assertEqual(
            A.shape[-1],
            3,
            f"A should have a shape of (N_res, N_atom, 3), but got shape {A.shape}",
        )
        self.assertEqual(len(np.unique(C)), 2, "Multimer protein should have 2 chains")

    def test_from_CRAB_single(self):
        obj = crab.Structure.from_file(self.single_chain_protein)
        C, R, A = obj.to_CRAB()
        obj2 = crab.Structure.from_CRAB(C, R, A)
        self.assertIsNotNone(obj2)

        self.assertEqual(len(obj), len(obj2), "Number of chains should be equal")
        for chain1, chain2 in zip(obj, obj2):
            self.assertEqual(
                len(chain1), len(chain2), "Number of residues should be equal"
            )
            for res1, res2 in zip(chain1, chain2):
                self.assertEqual(res1.resname, res2.resname)
                for atom1, atom2 in zip(res1, res2):
                    self.assertEqual(atom1.id, atom2.id, "Atom types should be equal")
                    self.assertTrue(
                        (atom1.coord - atom2.coord).sum() < 1e-4,
                        "Coordinates should be equal",
                    )

    def test_from_CRAB_multimer(self):
        obj = crab.Structure.from_file(self.multi_chain_protein_pdb)
        C, R, A = obj.to_CRAB()
        obj2 = crab.Structure.from_CRAB(C, R, A)
        self.assertIsNotNone(obj2)

        self.assertEqual(len(obj), len(obj2), "Number of chains should be equal")
        for chain1, chain2 in zip(obj, obj2):
            self.assertEqual(
                len(chain1), len(chain2), "Number of residues should be equal"
            )
            for res1, res2 in zip(chain1, chain2):
                self.assertEqual(res1.resname, res2.resname)
                for atom1, atom2 in zip(res1, res2):
                    self.assertEqual(atom1.id, atom2.id, "Atom types should be equal")
                    self.assertTrue(
                        (atom1.coord - atom2.coord).sum() < 1e-4,
                        "Coordinates should be equal",
                    )

    def test_to_file_single(self):
        obj = crab.Structure.from_file(self.single_chain_protein)
        C, R, A = obj.to_CRAB()
        obj2 = crab.Structure.from_CRAB(C, R, A)

        with StringIO() as f:
            obj.to_file(f, format="pdb")
            s1 = f.getvalue()
            self.assertTrue(len(s1) > 0, "Output string should not be empty")
            # s1 should be complete all-atom

        with StringIO() as f:
            obj2.to_file(f, format="pdb")
            s2 = f.getvalue()
            self.assertTrue(len(s2) > 0, "Output string should not be empty")
            # s2 contain only 4 backbone atoms

    def test_to_file_multimer(self):
        obj = crab.Structure.from_file(self.multi_chain_protein_pdb)
        C, R, A = obj.to_CRAB()
        obj2 = crab.Structure.from_CRAB(C, R, A)

        with StringIO() as f:
            obj.to_file(f, format="pdb")
            s1 = f.getvalue()
            with open("s1.pdb", "w") as f1:
                f1.write(s1)
            self.assertTrue(len(s1) > 0, "Output string should not be empty")
            # s1 should be complete all-atom

        with StringIO() as f:
            obj2.to_file(f, format="pdb")
            s2 = f.getvalue()
            with open("s2.pdb", "w") as f1:
                f1.write(s2)
            self.assertTrue(len(s2) > 0, "Output string should not be empty")
            # s2 contain only 4 backbone atoms

    def test_BondLengthCalculator(self):
        blc = crab.BondLengthCalculator()
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        self.assertEqual(blc(a, b), 1.0)

        a = torch.tensor([0.0, 3.0, 0.0])
        b = torch.tensor([0.0, 0.0, 4.0])
        self.assertEqual(blc(a, b), 5.0)

    def test_BondAngleCalculator(self):
        bac = crab.BondAngleCalculator()
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.tensor([1.0, 1.0, 0.0])
        self.assertAlmostEqual(bac(a, b, c), 1.5708, delta=1e-4)

        a = torch.tensor([0.0, 8.3, 5.0])
        b = torch.tensor([0.0, 1.0, 9.7])
        c = torch.tensor([1.0, 1.0, -10.3])
        self.assertAlmostEqual(bac(a, b, c), 0.9996, delta=1e-4)

    def test_DihedralAngleCalculator(self):
        dac = crab.DihedralAngleCalculator()
        a = torch.tensor([0.0, 0.0, -1.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.tensor([1.0, 1.0, 0.0])
        d = torch.tensor([1.0, 7.0, 8.0])
        self.assertAlmostEqual(dac(a, b, c, d), -2.9997, delta=1e-4)

        a = torch.tensor([0.0, 8.3, 5.0])
        b = torch.tensor([0.0, 1.0, 9.7])
        c = torch.tensor([1.0, 1.0, -10.3])
        d = torch.tensor([1.0, 0.0, 0.0])
        self.assertAlmostEqual(dac(a, b, c, d), 2.6987, delta=1e-4)

    def test_FourthAtomCalculator(self):
        fac = crab.FourthAtomCalculator()
        blc = crab.BondLengthCalculator()
        bac = crab.BondAngleCalculator()
        dac = crab.DihedralAngleCalculator()

        a = torch.tensor([29.1540, 40.4370, 5.1900])
        b = torch.tensor([30.1090, 39.4260, 4.6860])
        c = torch.tensor([29.4880, 38.5580, 3.5970])
        d = torch.tensor([30.3020, 37.6830, 3.0140])

        length = blc(c, d)
        angle = bac(b, c, d)
        dihedral = dac(a, b, c, d)

        d2 = fac(a, b, c, length, angle, dihedral)
        self.assertAlmostEqual(d[0], d2[0], delta=1e-4)
        self.assertAlmostEqual(d[1], d2[1], delta=1e-4)
        self.assertAlmostEqual(d[2], d2[2], delta=1e-4)


if __name__ == "__main__":
    unittest.main()
