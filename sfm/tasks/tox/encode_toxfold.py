# -*- coding: utf-8 -*-
import os
import sys

from sfm.data.dataset import Batch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import dataclasses
import io
import string
from functools import partial
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from finetune_toxfold import DownstreamConfig, StructureModel
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    IterableDataset,
    RandomSampler,
)
from tqdm import tqdm

from sfm.criterions.mae3d import ProteinPMLM
from sfm.criterions.mae3ddiff import ProteinMAE3dCriterions
from sfm.data.prot_data.dataset import BatchedDataDataset, ProteinLMDBDataset
from sfm.logging import logger
from sfm.models.pfm.openfold import residue_constants
from sfm.models.pfm.openfold.openfold_config import (
    HeadsConfig,
    StructureModuleConfig,
    loss_config,
)
from sfm.models.tox.modules.mae3ddiff import ProteinMAEDistCriterions
from sfm.models.tox.tox_config import TOXConfig
from sfm.models.tox.toxmodel import TOXModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert PDB_MAX_CHAINS == 62


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} "
                "chains because these cannot be written to PDB format"
            )


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents_chain_index is not None:
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]

    if parents is None or len(parents) == 0:
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]

    def res_1to3(r):
        return residue_constants.restype_1to3.get(restypes[r], "UNK")

    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index.astype(np.int32)

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = PDB_CHAIN_IDS[i]

    headers = get_pdb_headers(prot)
    if len(headers) > 0:
        pdb_lines.extend(headers)

    pdb_lines.append("MODEL     1")
    n = aatype.shape[0]
    atom_index = 1
    last_chain_index = chain_index[0]
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase

    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""

            chain_tag = "A"
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                # TODO: check this refactor, chose main branch version
                # f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = i == n - 1
        if chain_index is not None:
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_1to3(aatype[i]):>3} "
                f"{chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if i != n - 1:
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def from_prediction(
    features,
    result,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
    remark: Optional[str] = None,
    parents: Optional[Sequence[str]] = None,
    parents_chain_index: Optional[Sequence[int]] = None,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values
      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"]) - 1
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])

    return Protein(
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=result["final_atom_positions"],
        atom_mask=result["final_atom_mask"],
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        b_factors=b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=parents,
        parents_chain_index=parents_chain_index,
    )


def prep_output(
    out,
    batch,
    feature_dict,
    feature_processor,
    config_preset,
    multimer_ri_gap,
    subtract_plddt,
):
    plddt = out["plddt"]

    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if subtract_plddt:
        plddt_b_factors = 100 - plddt_b_factors

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    remark = ""

    # For multi-chain FASTAs
    ri = feature_dict["residue_index"]
    chain_index = (ri - np.arange(ri.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if c != cur_chain:
            cur_chain = c
            prev_chain_max = i + cur_chain * multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein


def tensorprint(d, name):
    print(f"=========={name}==========")
    for k, v in d.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            print(k, v.shape)
        elif isinstance(v, dict):
            print(k, ":")
            for k1, v1 in v.items():
                if isinstance(v1, (torch.Tensor, np.ndarray)):
                    print("    ", k1, v1.shape)
                else:
                    print("    ", k1, v1)
        else:
            print(k, v)


from sfm.data.prot_data.dataset import Alphabet, FoundationModelDataset


def embed_fn(rank, world_size, args, load_ckpt, loader):
    device = f"cuda:{rank % world_size}"

    basemodel = TOXModel(args, loss_fn=ProteinPMLM, load_ckpt=load_ckpt)
    model = StructureModel(args, basemodel)

    # load downstream ckpt
    checkpoints_state = torch.load(args.loadcheck_path, map_location="cpu")
    if "model" in checkpoints_state:
        checkpoints_state = checkpoints_state["model"]
    elif "module" in checkpoints_state:
        checkpoints_state = checkpoints_state["module"]

    IncompatibleKeys = model.load_state_dict(checkpoints_state, strict=False)
    IncompatibleKeys = IncompatibleKeys._asdict()
    print(f"checkpoint: {args.loadcheck_path} is loaded")
    print(f"Following keys are incompatible: {IncompatibleKeys.keys()}")
    # end

    model.to(device)
    model.eval()

    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, ncols=80, desc=f"Rank {rank}"):
            batch = move_to_device(batch, device)
            tensorprint(batch, "input")
            # B, L, D
            # embed = model.model.ft_forward(batch)
            outputs, angle_output, _ = model(batch)
            tensorprint(outputs, "output")

            for pidx in range(angle_output.shape[0]):
                naa = batch["naa"][pidx].item()
                out = {
                    k: v[pidx][:naa]
                    for k, v in outputs.items()
                    if isinstance(v, torch.Tensor)
                }
                out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
                tensorprint(out, "out")

                idx_to_tok = {v: k for k, v in Alphabet().tok_to_idx.items()}
                seq = "".join(
                    [
                        idx_to_tok[i.item()]
                        for idx, i in enumerate(batch["x"][pidx])
                        if idx < batch["naa"][pidx]
                    ]
                )
                myaatype = np.array(
                    [residue_constants.restype_order_with_x[i] for i in seq],
                    dtype=np.int64,
                )

                featuredict = {
                    "residue_index": np.arange(naa),
                    "aatype": myaatype,
                }  # out['aatype']}
                print("aatype:", featuredict["aatype"])
                print(
                    "aatype:",
                    "".join(
                        [residue_constants.restypes[i] for i in featuredict["aatype"]]
                    ),
                )

                tensorprint(featuredict, "featuredict")
                pred_protein = prep_output(
                    out, featuredict, featuredict, None, None, 200, False
                )
                # print(pred_protein)
                pdb_str = to_pdb(pred_protein)
                # with open(f'/home/yaosen/toxoutput/{batch["name"][pidx].split(".")[0]}.pdb', 'w') as f:
                #     f.write(pdb_str)
                with open(
                    f'/home/yaosen/toxoutput/{batch["name"][pidx].split(".")[0]}.pdb',
                    "w",
                ) as f:
                    f.write(pdb_str)

            idx += 1
            if idx >= 500:
                break


from Bio import SeqIO

from sfm.data.prot_data.collater import (
    pad_1d_unsqueeze,
    pad_2d_unsqueeze,
    pad_3d_unsqueeze,
)


class FastaDataset(FoundationModelDataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = self.set_default_args(args)
        self.vocab = Alphabet()
        self.seqs = []
        self.ids = []
        for record in SeqIO.parse(args.fasta_file, "fasta"):
            self.ids.append(record.id)
            self.seqs.append(str(record.seq))
        print(self.vocab.tok_to_idx)
        print(residue_constants.resname_to_idx)

    def set_default_args(self, args):
        if not hasattr(args, "max_length"):
            args.max_length = 1024
        if not hasattr(args, "fasta_file"):
            raise ValueError("Please specify fasta_file")

    def __getitem__(self, index: int) -> dict:
        item = {"id": index, "name": self.ids[index], "aa": self.seqs[index]}
        print(item)
        tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        # if self.vocab.prepend_bos:
        #     tokens.insert(0, self.vocab.cls_idx)
        # if self.vocab.append_eos:
        #     tokens.append(self.vocab.eos_idx)
        item["aa"] = np.array(tokens, dtype=np.int64)
        item["seq_length"] = len(tokens)
        item["pos"] = np.zeros((len(tokens), 37, 3), dtype=np.float32)
        item["pos_mask"] = np.ones((len(tokens), 37), dtype=np.float32)
        item["ang"] = np.zeros((len(tokens), 9), dtype=np.float32)
        item["ang_mask"] = np.ones((len(tokens), 9), dtype=np.float32)
        return item

    def __len__(self) -> int:
        return len(self.seqs)

    def size(self, index: int) -> int:
        return len(self.seqs[index])

    def num_tokens(self, index: int) -> int:
        return len(self.seqs[index]) + 2

    def num_tokens_vec(self, indices):
        raise NotImplementedError()

    def collate(self, samples: list) -> dict:
        max_tokens = max(len(s["aa"]) for s in samples)
        batch = dict()

        batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)
        batch["name"] = [s["name"] for s in samples]
        batch["naa"] = torch.tensor([len(s["aa"]) for s in samples], dtype=torch.long)

        batch["seq_length"] = torch.tensor(
            [s["seq_length"] for s in samples], dtype=torch.long
        )
        # (Nres+2,) -> (B, Nres+2)
        batch["x"] = torch.cat(
            [
                pad_1d_unsqueeze(
                    torch.from_numpy(s["aa"]), max_tokens, 0, self.vocab.padding_idx
                )
                for s in samples
            ]
        )

        batch["seq_length"] = torch.cat(
            [torch.tensor([s["seq_length"]]) for s in samples]
        )

        batch["pos"] = torch.cat(
            [
                pad_3d_unsqueeze(torch.from_numpy(s["pos"]), max_tokens, 0, torch.inf)
                for s in samples
            ]
        )
        batch["pos_mask"] = torch.cat(
            [
                pad_2d_unsqueeze(torch.from_numpy(s["pos_mask"]), max_tokens, 0, 0)
                for s in samples
            ]
        )
        batch["ang"] = torch.cat(
            [
                pad_2d_unsqueeze(torch.from_numpy(s["ang"]), max_tokens, 0, torch.inf)
                for s in samples
            ]
        )
        batch["ang_mask"] = torch.cat(
            [
                pad_2d_unsqueeze(torch.from_numpy(s["ang_mask"]), max_tokens, 0, 0)
                for s in samples
            ]
        )
        return batch


@cli(DistributedTrainConfig, TOXConfig, StructureModuleConfig, DownstreamConfig)
def main(args) -> None:
    assert (
        args.data_path is not None and len(args.data_path) > 0
    ), f"lmdb_path is {args.data_path} it should not be None or empty"

    # valset = ProteinLMDBDataset(args)
    dataset = ProteinLMDBDataset(args)
    trainset, valset = dataset.split_dataset(sort=False)

    # args.fasta_file = "/home/yaosen/2dri.fasta"
    # valset = FastaDataset(args)

    BatchedDataDataset(
        trainset,
        args=args,
        vocab=dataset.vocab,
    )
    val_data = BatchedDataDataset(
        valset,
        args=args,
        vocab=valset.vocab,
    )
    print("esm toks, ", valset.vocab.all_toks)
    sampler = RandomSampler(val_data)
    data_loader = DataLoader(
        val_data,
        sampler=sampler,
        batch_size=2,
        collate_fn=val_data.collate,
        drop_last=False,
    )
    print(next(iter(data_loader)))
    # exit(0)
    args.infer = True
    assert args.loadcheck_path, "checkpath is not provided"
    embed_fn(0, 1, args, False, data_loader)


if __name__ == "__main__":
    main()


# -WFIWDSQQQWWFMQQSIPISRDASQCSCHNVQLEYGPFTMQIDYGSPGSQNTPQTSVWNSWHQPCGWSPNKSPKSICIFLKSIPSDNTAIVVRAAIWGWAAQNNCNIYQFQTWWTWSSASTHQMSWCMIAMQTAAIQRIWHNVCMDMWTWSYKSQAITSQWETCMAWIFSSQIVSISSWWTWMSSSIQTFTVGRKWKEHMSKTHWNHHWDM
# -VQLVESGGGVVQPGGSLRLSCEASGFSFKDYGMHWIRQTPG--LEWISRISGDTRGTSYVDSVKGRFIVSRDNSRNSLFLQMNSLRSEDTALYYCAALVIVAAGDDFDLWGQGTVVTVSSASTKGPSVFPLAP-------GTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP
# EVQLVESGGGVVQPGGSLRLSCEASGFSFKDYGMHWIRQTPGKGLEWISRISGDTRGTSYVDSVKGRFIVSRDNSRNSLFLQMNSLRSEDTALYYCAALVIVAAGDDFDLWGQGTVVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP
# id torch.Size([4])
# name ['AF-B1XP37-F1-model_v4.cif', 'AF-Q6GHH9-F1-model_v4.cif', 'AF-A0A1D6LF90-F1-model_v4.cif', 'AF-Q5PKD5-F1-model_v4.cif']
# naa torch.Size([4])
# x torch.Size([4, 255])
# masked_aa torch.Size([4, 255, 1])
# mask_pos torch.Size([4, 255, 1])
# seq_length torch.Size([4])
# pos torch.Size([4, 255, 37, 3])
# pos_mask torch.Size([4, 255, 37])
# ang torch.Size([4, 255, 9])
# ang_mask torch.Size([4, 255, 9])
# ang_noise torch.Size([4, 255, 9])
# node_type_edge torch.Size([4, 255, 255, 2])


# ==========output==========
# single torch.Size([4, 255, 768])
# pair torch.Size([4, 255, 255, 192])
# aatype torch.Size([4, 255])
# sm :
#      frames torch.Size([4, 4, 255, 7])
#      sidechain_frames torch.Size([4, 4, 255, 8, 4, 4])
#      unnormalized_angles torch.Size([4, 4, 255, 7, 2])
#      angles torch.Size([4, 4, 255, 7, 2])
#      positions torch.Size([4, 4, 255, 14, 3])
#      states torch.Size([4, 4, 255, 768])
#      single torch.Size([4, 255, 768])
# final_atom_positions torch.Size([4, 255, 37, 3])
# final_atom_mask torch.Size([4, 255, 37])
# final_affine_tensor torch.Size([4, 255, 7])
# lddt_logits torch.Size([4, 255, 64])
# plddt torch.Size([4, 255])
# distogram_logits torch.Size([4, 255, 255, 64])
# experimentally_resolved_logits torch.Size([4, 255, 37])
