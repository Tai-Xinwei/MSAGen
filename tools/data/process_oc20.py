# -*- coding: utf-8 -*-
import lmdb
from tqdm import tqdm
import pickle
import periodictable
from rdkit import Chem


prompt="The adsorption energy of <material> {material} </material> and <mol> {molecule} </mol> is {energy} eV."


def convert_atomic_number_to_symbol(atomic_number):
    return periodictable.elements[atomic_number].symbol


def get_formula(atomic_symbols):
    return " ".join(atomic_symbols)


def get_smiles(atomic_symbols):
    mol = Chem.MolFromSmiles("".join(atomic_symbols))
    # Obtain the SMILES string
    smiles = Chem.MolToSmiles(mol)
    return smiles


def dump_db(path="/hai1mfm/data/oc4gpt/oc20-2m", output_path="/hai1mfm/data/oc4gpt/oc20-2m.txt"):
    # Create the LMDB database
    env = lmdb.open(path, readonly=True)

    # Get the keys
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        with open(output_path, "w") as f:
            for key, val in tqdm(cursor):
                key = key.decode("utf-8")
                val = pickle.loads(val)
                energy = val.y
                tags = val.tags.numpy().tolist()
                atomic_numbers = val.atomic_numbers.numpy().tolist()
                material = []
                molecule = []
                for i in range(len(tags)):
                    if tags[i] == 2:
                        molecule.append(convert_atomic_number_to_symbol(atomic_numbers[i]))
                    else:
                        material.append(convert_atomic_number_to_symbol(atomic_numbers[i]))
                material = get_formula(material)
                #molecule = get_smiles(molecule)
                molecule = get_formula(molecule)
                sentence = prompt.format(material=material, molecule=molecule, energy=energy)
                f.write(sentence + "\n")


if __name__ == "__main__":
    dump_db()
