# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from openbabel import openbabel


def read_xyz_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


def xyz_to_smiles(xyz_string):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "can")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz_string)

    canonical_smiles = obConversion.WriteString(mol).strip().split()[0]

    return canonical_smiles


# def xyz_to_item(file_path):
#     smile_string = read_xyz_file(file_path)
#     smiles = xyz_to_smiles(smile_string)

if __name__ == "__main__":
    file_path = (
        "/mnt/pm6raw/Compound_000025001_000050000/000025976/000025976.PM6.S0.xyz"
    )
    smile_string = read_xyz_file(file_path)
    smiles = xyz_to_smiles(smile_string)
    print(smiles)
