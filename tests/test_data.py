# -*- coding: utf-8 -*-
import unittest

from sfm.data.molecule import Molecule


class TestCreateMolecule(unittest.TestCase):
    def test_create_molecule(self):
        Molecule()


if __name__ == "__main__":
    unittest.main()
