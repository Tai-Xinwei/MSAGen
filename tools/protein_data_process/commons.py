# -*- coding: utf-8 -*-
import pickle
from sfm.logging import logger
import zlib
from io import StringIO
from typing import List, Union

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Chain, PDBIO
from Bio.PDB import PDBParser

import tools.protein_data_process.residue_constants as residue_constants
from pathlib import Path
import gzip
from urllib.request import urlretrieve
from urllib.parse import urlparse
import hashlib
import zipfile
import tarfile


ANGLE_STRS: List[str]=["psi", "phi", "omg", "tau", "chi1", "chi2", "chi3", "chi4", "chi5"]

def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))


def validate_lmdb_keys():
    """
    Validate the dict keys in the lmdb file, providing a summary of the keys.
    """
    pass

def fix_structure(file_like, format: str) -> StringIO:
    """
    Fix the pdb/cif file using pdbfixer. It will add missing residues, atoms, and standardize the residues.
    For fixing, user should use it with files directly from PDB, because it needs RESSEQ lines from header.

    Args:
        file_like (file-like object): a file like object containing pdb/cif file.
        format (str): specify the format of the file, can be "pdb" or "cif".

    Returns:
        str: the relaxed and fixed pdb/cif file.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile, PDBxFile
    # t1 = time.time()
    output = StringIO()
    if format in ["pdb", "ent"]:
        fixer = PDBFixer(pdbfile=file_like)
    elif format == "cif":
        fixer = PDBFixer(pdbxfile=file_like)
    fixer.findMissingResidues()
    # logger.warning(f"Missing residues: {fixer.missingResidues}")
    fixer.findNonstandardResidues()
    # logger.warning(f"Nonstandard residues: {fixer.nonstandardResidues}")
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    # logger.warning(f"Missing atoms: {fixer.missingAtoms}")
    # logger.warning(f"Missing terminals: {fixer.missingTerminals}")
    fixer.addMissingAtoms()
    # fixer.addMissingHydrogens(7.0)
    # fixer.addSolvent(fixer.topology.getUnitCellDimensions())
    if format == "pdb":
        PDBFile.writeFile(fixer.topology, fixer.positions, output)
    elif format == "cif":
        PDBxFile.writeFile(fixer.topology, fixer.positions, output)
    # logger.info(f"Relaxing and fixing pdb takes {time.time() - t1:.2f} seconds.")
    return output


class Protein:
    def __init__(self, name: str, chain: Union[Chain.Chain, str]):
        """Protein manupulation class. Currently, it has the following functions:
        1. Read pdb/cif files and process them into a dict containing position, angle, amino acid features.

        TODO: add protein / residue level features; protein comparision; protein visualization; structure rebuilding.

        Args:
            name (str): global name of the protein, it is user's responsibility to make sure it is unique.
            chain (Chain): Bio.PDB.Chain object, it should be a single chain for calculating all features.
        """
        self.name = name
        self.chain = chain
        self.target = None
        self.size = len(chain)
        self.processed_dict = None

    def __len__(self):
        return self.size

    def _process_chain_pos(self):
        # natoms = len([a for a in chain.get_atoms() if a.name.strip() != ''])
        pos = np.zeros([len(self), 37, 3], dtype=np.float32)
        pos_mask = np.zeros([len(self), 37], dtype=np.int32)
        for ridx, res in enumerate(self.chain):
            if res.resname not in residue_constants.restype_3to1 and res.resname != "UNK":
                # not a standard amino acid, UNK instead
                logger.warning(f"Nonstandard residue {res.resname} in {self.name}, skip.")
                # res.resname = "UNK"
                continue
            for atm_name in residue_constants.residue_atoms[res.resname]:
                try:
                    atm = res[atm_name]
                except KeyError:
                    pos_mask[ridx, residue_constants.atom_order[atm_name]] = 2
                    # should have an atom here, but don't
                    continue
                pos[ridx, residue_constants.atom_order[atm_name], :] = atm.coord
                pos_mask[ridx, residue_constants.atom_order[atm_name]] = 1
        self.processed_dict["pos"] = pos
        self.processed_dict["pos_mask"] = pos_mask

    def _process_chain_ang(self):
        """
        Extract hedronal and dihedral angles from the chain.
        Note: the angles are in degrees and are redundant for the structure.
        """
        ang = np.ones([len(self), 9], dtype=np.float32) * 360.0
        ang_mask = np.zeros([len(self), 9], dtype=np.int32)
        for ridx, res in enumerate(self.chain):
            if res.resname not in residue_constants.restype_3to1 and res.resname != "UNK":
                # not a standard amino acid, skip
                continue
            for aidx, defi in enumerate(ANGLE_STRS):
                if res.internal_coord is None:
                    continue
                a = res.internal_coord.get_angle(defi)
                if a is not None:
                    ang_mask[ridx, aidx] = 1
                    ang[ridx, aidx] = a
        self.processed_dict["ang"] = ang
        self.processed_dict["ang_mask"] = ang_mask


    def _process_chain_aa(self):
        """
        Extract amino acid sequence from the chain.
        """
        aa = []
        for res in self.chain:
            if res.resname not in residue_constants.restype_3to1 and res.resname != "UNK":
                # not a standard amino acid, skip
                continue
            aa.append(residue_constants.restype_3to1[res.resname.upper()] if res.resname != "UNK" else "X")
        self.processed_dict["aa"] = np.array(aa)


    def process_chain(self):
        """
        Process the chain into a dict containing position, angle, amino acid features.
        Note: this function should only work on chain without flaws, e.g. missing residues / atoms.
        """
        if self.processed_dict is not None:
            logger.warning("This protein has already been processed. Skip.")
            return
        if isinstance(self.chain, str):
            self.processed_dict = dict()
            self.processed_dict['aa'] = list(self.chain.strip())
            self.processed_dict['pos'] = np.zeros([len(self), 37, 3], dtype=np.float32)
            self.processed_dict['pos_mask'] = np.zeros([len(self), 37], dtype=np.int32)
            self.processed_dict['ang'] = np.ones([len(self), 9], dtype=np.float32) * 360.0
            self.processed_dict['ang_mask'] = np.zeros([len(self), 9], dtype=np.int32)
            return
        elif isinstance(self.chain, Chain.Chain):
            self.chain.atom_to_internal_coordinates()
            self.processed_dict = dict()
            self._process_chain_pos()
            self._process_chain_ang()
            self._process_chain_aa()
            # check the shapes
            assert self.processed_dict["pos"].shape == (len(self), 37, 3)
            assert self.processed_dict["pos_mask"].shape == (len(self), 37)
            assert self.processed_dict["ang"].shape == (len(self), 9)
            assert self.processed_dict["ang_mask"].shape == (len(self), 9)
            assert self.processed_dict["aa"].shape == (len(self),)
        else:
            raise ValueError(f"Unknown chain type {type(self.chain)}, only support str and Bio.PDB.Chain.")


    @staticmethod
    def download(url: str, path: Union[Path, str], md5: str, filename: str=None) -> Path:
        """
        Download a file from the specified url.
        Skip the downloading step if there exists a file satisfying the given MD5.

        Parameters:
            url (str): URL to download
            path (Union[Path, str]): path to store the downloaded file
            md5 (str): MD5 of the file for validation
            save_file (str, optional): name of save file. If not specified, infer the file name from the URL.
        """
        filename = filename if filename else Path(urlparse(url).path).name
        filepath = Path(path) / filename
        if not filepath.exists() or Protein.compute_md5(filepath) != md5:
            logger.info(f"Downloading {url} to {filepath}")
            urlretrieve(url, filepath)
        return filepath

    @staticmethod
    def extract(zip_file: Path, extract_dir: Path=None):
        """Extract files from a zip file.

        Args:
            zip_file (Path): Path to the zip file
            extract_dir (Path, optional): Path to the directory to extract files. If not specified, use the same directory as the zip file.

        Returns:
            _type_: _description_
        """
        if zip_file.name.endswith(".zip"):
            filename = zip_file.name.replace(".zip", "")
            filedir = zip_file.parent
            logger.info(f"Extracting {zip_file} to {filedir / filename}")
            zipped = zipfile.ZipFile(zip_file)
            zipped.extractall(path=filedir)
        elif zip_file.name.endswith(".tar.gz"):
            filename = zip_file.name.replace(".tar.gz", "")
            filedir = zip_file.parent
            logger.info(f"Extracting {zip_file} to {filedir / filename}")
            zipped = tarfile.open(zip_file)
            zipped.extractall(path=filedir)
        else:
            raise ValueError(f"Unknown file type {zip_file}, only support .zip and .tar.gz")
        return extract_dir if extract_dir is not None else filedir / filename

    @staticmethod
    def compute_md5(filepath: Path, chunk_size: int=1024**2) -> str:
        """
        Compute MD5 of the file.

        Parameters:
            file_name (str): file name
            chunk_size (int, optional): chunk size for reading large files
        """
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()


    @staticmethod
    def get_pdb_rcsb(pdbid: str, path: Union[Path, str], format: str='pdb') -> Path:
        """
        Get the pdb file from rcsb.org

        Parameters:
            pdbid (str): pdb id
            path (Union[Path, str]): path to store the downloaded file
            format (str, optional): file format
        """
        url = f"https://files.rcsb.org/download/{pdbid}.{format}"
        filename = urlparse(url).path.split('/')[-1]
        filepath = Protein.download(url, path, None, filename)
        return filepath

    @classmethod
    def from_sequence(cls, seq: str, name: str, target=None, **kwargs):
        obj = cls(name, seq, **kwargs)
        obj.process_chain()
        obj.target = target
        return obj


    @classmethod
    def from_file(cls, file: Union[str, Path, StringIO], name: str, fix: bool=False, format: str=None, target=None, **kwargs):
        # pdb can be a string path, a path object, or a file-like object
        if isinstance(file, (str, Path)):
            if str(file).endswith('.pdb.gz') or str(file).endswith('.cif.gz'):
                format = str(file).split('.')[-2]
                file = gzip.open(file, "rt")
            elif str(file).endswith('.pdb') or str(file).endswith('.cif'):
                format = str(file).split('.')[-1]
                file = open(file, "r")
            else:
                raise ValueError(f"Unknown file type {file}, only support .pdb and .pdb.gz")
        if format is None:
            raise ValueError("Please specify the file format because your input is an file-like object.")
        # now file is a file-like object
        if fix:
            try:
                file = fix_structure(file, format)
            except Exception as e:
                print(f"Error in {name}: {e} {e.args}, stop here.")
                return None
        file.seek(0)
        if format == "pdb":
            parser = PDBParser()
        elif format == "cif":
            parser = MMCIFParser()
        else:
            raise ValueError(f"Unknown file format {format}, only support pdb and cif.")
            # should not reach here

        structure = parser.get_structure("", file)
        assert len(structure.child_list[0].child_dict) == 1, f"More than one chain in {name}, currently not allowed."
        chain = structure.child_list[0].child_list[0]
        obj = cls(name, chain, **kwargs)
        obj.process_chain()
        if target is not None:
            obj.target = target
        return obj

    def to_dict(self):
        if self.processed_dict is None:
            logger.error("This protein has not been processed yet. Please call process_chain() first.")
            return
        d = {**self.processed_dict, 'name': self.name, 'size': len(self)}
        if self.target is not None:
            d['target'] = self.target
        return d

    def encode(self) -> bytes:
        if self.processed_dict is None:
            logger.error("This protein has not been processed yet. Please call process_chain() first.")
            return
        return obj2bstr(self.to_dict())

    @staticmethod
    def chain_split(structure_file: Union[str, Path], dst_dir: Union[str, Path], cutoff: int, format="pdb"):
        """
        Split the structure into chains, and save them to different files.
        Parameters:
            structure_file (Union[str, Path]): path to the structure file,
            path (Union[Path, str]): path to store the downloaded file
            format (str, optional): file format
        """
        if format == "pdb":
            parser = PDBParser()
        elif format == "cif":
            parser = MMCIFParser()
        else:
            raise ValueError(f"Unknown file format {format}, only support pdb and cif.")
        if isinstance(structure_file, str):
            structure_file = Path(structure_file)

        if structure_file.suffix == ".gz":
            with gzip.open(structure_file, "rt") as f:
                structure = parser.get_structure("", f)
        else:
            structure = parser.get_structure("", structure_file)
        for chain in structure.get_chains():
            chainid, length = chain.id, len(chain)
            flag = True
            for res in chain:
                if res.resname not in residue_constants.restype_3to1:
                    logger.warning(f"Nonstandard residue {res.resname}-{res.index} in {chainid} of structure {structure_file}, skip this chain.")
                    # not a standard amino acid, skip
                    flag = False
                    break
            if flag and length >= cutoff:
                io = PDBIO()
                io.set_structure(chain)
                io.save(dst_dir / f"{structure_file.stem}-{chain.id}.pdb")
