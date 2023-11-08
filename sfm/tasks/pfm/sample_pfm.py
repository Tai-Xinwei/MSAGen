# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import torch
from spyrmsd import molecule, rmsd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from sfm.criterions.mae3d import ProteinMAE3dCriterions
from sfm.data.prot_data.dataset import BatchedDataDataset, ProteinLMDBDataset
from sfm.logging import logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfmmodel import PFMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def linear_molecule_adjacency(n_atoms):
    adjacency_matrix = np.zeros((n_atoms, n_atoms))
    np.fill_diagonal(adjacency_matrix[1:], 1)
    np.fill_diagonal(adjacency_matrix[:, 1:], 1)
    return adjacency_matrix


# def calculate_rmsd(actual_values, predicted_values):
#     # Make sure the arrays have the same size
#     assert len(actual_values) == len(predicted_values)

#     # Calculate the square differences
#     square_diffs = (actual_values - predicted_values) ** 2

#     # Sum the square differences for each point
#     sum_square_diffs = np.sum(square_diffs, axis=1)

#     # Calculate the mean of square differences
#     mean_square_diff = np.mean(sum_square_diffs)

#     # Take the square root of the mean
#     rmsd = np.sqrt(mean_square_diff)

#     return rmsd


@cli(DistributedTrainConfig, PFMConfig)
def main(args) -> None:
    torch.distributed.init_process_group()

    assert (
        args.data_path is not None and len(args.data_path) > 0
    ), f"lmdb_path is {args.data_path} it should not be None or empty"

    dataset = ProteinLMDBDataset(args)

    trainset, valset = dataset.split_dataset(sort=False)

    BatchedDataDataset(
        trainset,
        args=args,
        vocab=dataset.vocab,
    )
    val_data = BatchedDataDataset(
        valset,
        args=args,
        vocab=dataset.vocab,
    )

    model = PFMModel(args, loss_fn=ProteinMAE3dCriterions).cuda()
    model.load_state_dict(
        torch.load(args.loadcheck_path + "/mp_rank_00_model_states.pt")["module"]
    )
    logger.info(f"load model from {args.loadcheck_path}")

    model.eval()

    logger.info(f"world size is {args.world_size}, local rank: {args.local_rank}")
    sampler = torch.utils.data.distributed.DistributedSampler(
        val_data,
        num_replicas=args.world_size,
        rank=args.local_rank,
    )

    dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        collate_fn=val_data.collate,
        sampler=sampler,
    )

    loss_pos = torch.nn.L1Loss(reduction="mean")
    RMSD = 0.0
    sample_num = 0
    batch_num = 0
    pos_loss = 0

    beta_list = model.net.diffnoise.beta_list
    alphas_cumprod_list = model.net.diffnoise.alphas_cumprod

    pbar = tqdm(dataloader)
    for batch in tqdm(pbar):
        batch = move_to_device(batch, args.local_rank)
        with torch.no_grad():
            ori_pos = batch["pos"].clone()
            residue_seq = batch["x"]
            mask_aa = batch["masked_aa"]
            mask_pos = batch["mask_pos"]

            mask_aa, mask_pos, padding_mask, _, mode_mask = model.net._set_mask(
                mask_aa, mask_pos, residue_seq
            )
            pos, time_pos, time_aa = model.net._set_noise(
                ori_pos, mask_pos, mode_mask, time_step=999
            )
            batch["pos"] = pos

            for t in range(0, 1000)[::-1]:
                last_step_coords = batch["pos"]
                logits, node_output, mask_pos, mask_aa = model(
                    batch,
                    time_step=t,
                    mask_aa=mask_aa,
                    mask_pos=mask_pos,
                    padding_mask=padding_mask,
                    mode_mask=mode_mask,
                    time_pos=time_pos,
                    time_aa=time_aa,
                )
                visible_pos = ori_pos.masked_fill_(mask_pos, 0.0)

                if t != 0:
                    noisy_pos = (
                        torch.sqrt(1 - beta_list[t])
                        * (1 - alphas_cumprod_list[t - 1])
                        * last_step_coords
                        + beta_list[t]
                        * torch.sqrt(alphas_cumprod_list[t - 1])
                        * node_output
                    ) / (1 - alphas_cumprod_list[t])
                else:
                    noisy_pos = (beta_list[t] * node_output) / (
                        1 - alphas_cumprod_list[t]
                    )

                noisy_pos = noisy_pos.masked_fill(~mask_pos.bool(), 0.0)
                batch["pos"] = noisy_pos + visible_pos

            node_output = batch["pos"]

            # logits, node_output, mask_pos, mask_aa = model(batch, time_step=999)

            label_pos = ori_pos[mask_pos.squeeze(-1)]
            pos_loss += (
                loss_pos(
                    node_output[mask_pos.squeeze(-1)].to(torch.float32),
                    label_pos.to(torch.float32),
                )
                .sum(dim=-1)
                .item()
            )

            for i in range(batch["pos"].shape[0]):
                coords1 = ori_pos[i][mask_pos[i].squeeze(-1)].view(-1, 3)
                coords2 = node_output[i][mask_pos[i].squeeze(-1)].view(-1, 3)
                natom = coords1.shape[0]
                adj_matrx = linear_molecule_adjacency(natom)

                coords1 = coords1.cpu().numpy()
                coords2 = coords2.cpu().numpy()

                RMSD += rmsd.symmrmsd(
                    coords1,
                    coords2,
                    batch["x"][i, ...].cpu().numpy(),
                    batch["x"][i, ...].cpu().numpy(),
                    adj_matrx,
                    adj_matrx,
                    center=True,
                    minimize=True,
                )

            sample_num += batch["pos"].shape[0]
            batch_num += 1
            running_rmsd = RMSD / sample_num
            running_loss = pos_loss / batch_num

            pbar.set_postfix(Running_rmsd=running_rmsd, running_loss=running_loss)

        break
        # with torch.no_grad():
        #     aa_seq = batch["x"][mask_aa.squeeze(-1).bool()]

        # logits = logits[:, :, :][mask_aa.squeeze(-1).bool()]

        # right_num += (
        #     logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq
        # ).sum().to(torch.float32)
        # total_res += aa_seq.view(-1).size(0)
        # pbar.set_postfix(type_acc=right_num/total_res)

    # print(f"type acc: {right_num / total_res}, total residue num: {total_res}")

    # all reduce rmsd and sample num
    RMSD = torch.tensor(RMSD).cuda()
    sample_num = torch.tensor(sample_num).cuda()
    torch.distributed.all_reduce(RMSD)
    torch.distributed.all_reduce(sample_num)
    RMSD = RMSD.item()
    sample_num = sample_num.item()
    logger.info(f"RMSD: {RMSD / sample_num}, sample num: {sample_num}")


if __name__ == "__main__":
    main()
