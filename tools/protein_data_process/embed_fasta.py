# -*- coding: utf-8 -*-
import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import asdict, dataclass
from pathlib import Path
from finetune_pfm_v2 import (
    DownstreamConfig,
    init_model,
    load_batched_dataset,
    multi_label_transform,
)

from sfm.data.prot_data.dataset import FoundationModelDataset, Alphabet
from sfm.data.prot_data.collater import pad_1d_unsqueeze
from sfm.logging import logger, metric_logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device
from Bio import SeqIO

@dataclass
class FastaConfig:
    fasta_file: str = ""


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

    def set_default_args(self, args):
        if not hasattr(args, "max_length"):
            args.max_length = 2048
        if not hasattr(args, "fasta_file"):
            raise ValueError("Please specify fasta_file")

    def __getitem__(self, index: int) -> dict:
        item = {"id": index, 'name': self.ids[index], "aa": self.seqs[index]}
        tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        if self.vocab.prepend_bos:
            tokens.insert(0, self.vocab.cls_idx)
        if self.vocab.append_eos:
            tokens.append(self.vocab.eos_idx)
        item["aa"] = np.array(tokens, dtype=np.int64)
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
        batch["naa"] = torch.tensor([len(s["aa"]) for s in samples], dtype=torch.long)

        # (Nres+2,) -> (B, Nres+2)
        batch["x"] = torch.cat(
            [
                pad_1d_unsqueeze(
                    torch.from_numpy(s["aa"]), max_tokens, 0, self.vocab.padding_idx
                )
                for s in samples
            ]
        )
        return batch


def embed(trainer):
    """
    Validate the model on the validation data loader.
    """
    if trainer.valid_data_loader is None:
        logger.warning("No validation data, skip validation")
        return

    logger.info(
        "Start validation for epoch: {}, global step: {}",
        trainer.state.epoch,
        trainer.state.global_step,
    )

    pred = []

    for idx, batch_data in enumerate(trainer.valid_data_loader):
        trainer.model.eval()
        trainer.model.to(trainer.accelerator.device)
        batch_data = move_to_device(batch_data, trainer.accelerator.device)
        with torch.no_grad():
            output = trainer.model(batch_data)
            pred.append(output.to(torch.float32).detach().cpu())

        pred = torch.cat(pred, axis=0)
    return pred.numpy()


@cli(DistributedTrainConfig, PFMConfig, DownstreamConfig, FastaConfig)
def test_checkpoint(args) -> None:
    assert Path(args.loadcheck_path).is_file()

    dataset = FastaDataset(args)
    model = init_model(args, load_ckpt=False)

    checkpoints_state = torch.load(args.loadcheck_path, map_location="cpu")
    if "model" in checkpoints_state:
        checkpoints_state = checkpoints_state["model"]
    elif "module" in checkpoints_state:
        checkpoints_state = checkpoints_state["module"]

    IncompatibleKeys = model.load_state_dict(checkpoints_state, strict=False)
    IncompatibleKeys = IncompatibleKeys._asdict()
    logger.info(f"checkpoint: {args.loadcheck_path} is loaded")
    logger.warning(f"Following keys are incompatible: {IncompatibleKeys.keys()}")

    trainer = Trainer(args, model, train_data=dataset,)
    embeddings = embed(trainer)
    np.save(f"embeddings_{args.fasta_file}.npy", embeddings)


if __name__ == "__main__":
    test_checkpoint()
