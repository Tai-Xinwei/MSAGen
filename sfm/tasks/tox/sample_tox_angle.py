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
from sfm.models.tox.tox_config import TOXConfig
from sfm.models.tox.toxmodel import TOXModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def linear_molecule_adjacency(n_atoms):
    adjacency_matrix = np.zeros((n_atoms, n_atoms))
    np.fill_diagonal(adjacency_matrix[1:], 1)
    np.fill_diagonal(adjacency_matrix[:, 1:], 1)
    return adjacency_matrix


@cli(DistributedTrainConfig, TOXConfig)
def main(args) -> None:
    # init distributed env, data and model
    torch.distributed.init_process_group()

    assert (
        args.data_path is not None and len(args.data_path) > 0
    ), f"lmdb_path is {args.data_path} it should not be None or empty"

    dataset = ProteinLMDBDataset(args)

    logger.info(f"load data from {args.data_path}")

    trainset, valset = dataset.split_dataset(sort=False)

    val_data = BatchedDataDataset(
        valset,
        args=args,
        vocab=dataset.vocab,
    )

    model = TOXModel(args, loss_fn=ProteinMAE3dCriterions, load_ckpt=True).cuda()
    # model.load_state_dict(
    #     torch.load(args.loadcheck_path + "/mp_rank_00_model_states.pt")["module"]
    # )
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

    torch.nn.MSELoss(reduction="mean")
    loss_angle = torch.nn.MSELoss(reduction="mean")
    batch_ang_RMSD = 0.0
    good_ang_RMSD = 0.0
    sample_num = 0
    batch_num = 0
    ang_loss = 0
    total_res = 0
    right_num = 0

    # params for computing the noise of pos and angle
    ## pos VP
    beta_list = model.net.diffnoise.beta_list
    alpha_cumprod = model.net.diffnoise.alphas_cumprod

    ## angle VE
    sigma_min = 0.01 * torch.pi
    sigma_max = 1.0 * torch.pi
    sigma_square_list = []

    discret_t_total = beta_list.shape[0]  # total time steps discretized MAX
    # discret_t_total = 10

    for i in range(0, discret_t_total):
        t = i / (discret_t_total - 1)
        sigma_square_list.append(
            (sigma_min ** (1 - t) * sigma_max**t) ** 2
        )  # SMLD (31), increase
    sigma_square_list = torch.tensor(
        sigma_square_list, dtype=beta_list.dtype, device=beta_list.device
    )

    # prepross the data
    pbar = tqdm(dataloader)
    for batch in pbar:
        batch = move_to_device(batch, args.local_rank)
        with torch.no_grad():
            """--------------------------------inference--------------------------------"""
            # mask the pos and angle
            ori_pos = batch["pos"].clone()
            ori_angle = batch["ang"].clone()

            pos_mask = (batch["pos_mask"] == 1).unsqueeze(
                -1
            )  # pos_mask has the same shape to ori_pos
            ori_pos = ori_pos.masked_fill(~pos_mask, 0.0)

            angle_mask = batch["ang_mask"] == 1
            ori_angle = ori_angle.masked_fill(~angle_mask, 100.0).to(ori_pos.dtype)

            residue_seq = batch["x"]
            mask_aa = batch["masked_aa"]
            mask_pos = batch["mask_pos"]
            (
                mask_aa,
                mask_pos,
                padding_mask,
                _,
                _,
                mode_mask,
                mask_angle,
            ) = model.net._set_mask(mask_aa, mask_pos, residue_seq)

            # Final noise is GS
            time_pos = torch.tensor([1.0], device=ori_pos.device)
            time_aa = torch.tensor([1.0], device=ori_pos.device)

            batch["pos"] = torch.normal(
                mean=0.0,
                std=sigma_max,
                size=batch["pos"].shape,
                device=batch["pos"].device,
                dtype=batch["pos"].dtype,
            )
            batch["ang"] = torch.normal(
                mean=0.0,
                std=np.sqrt(sigma_max),
                size=batch["ang"].shape,
                device=batch["ang"].device,
                dtype=batch["ang"].dtype,
            )

            zeros_filled = torch.zeros(
                (batch["ang"].shape[0], batch["ang"].shape[1], 6),
                dtype=batch["ang"].dtype,
                device=batch["ang"].device,
            )

            # iterate over time steps to compute the x0 with weighted sum xt
            for i in range(1, discret_t_total)[
                ::-1
            ]:  # iterate times is discret_t_total-1
                ## x_{i}
                last_step_coords = batch["pos"]
                last_step_ang = batch["ang"]

                t = i / (discret_t_total - 1)

                ## x_{\theta}(x_{i}, i)
                logits, node_output, angle_output, _, _, _, _ = model.net.sample(
                    batch,
                    time_step=t,
                    mask_aa=mask_aa,
                    mask_pos=mask_pos,
                    mask_angle=mask_angle,
                    padding_mask=padding_mask,
                    mode_mask=mode_mask,
                    time_pos=time_pos,
                    time_aa=time_aa,
                )

                ## compute x_{i-1}
                if args.ode_mode:
                    noisy_pos = (
                        2 - torch.sqrt(1 - beta_list[i])
                    ) * last_step_coords + 1 / 2 * beta_list[i] * (
                        (torch.sqrt(alpha_cumprod[i]) * node_output - last_step_coords)
                        / (1 - alpha_cumprod[i])
                    )

                    noisy_ang = last_step_ang[:, :, :3] + 1 / 2 * (
                        sigma_square_list[i] - sigma_square_list[i - 1]
                    ) * (
                        (angle_output[:, :, :3] - last_step_ang[:, :, :3])
                        / (sigma_square_list[i] - sigma_square_list[0])
                    )
                else:
                    noisy_pos = (
                        (2 - torch.sqrt(1 - beta_list[i])) * last_step_coords
                        + beta_list[i]
                        * (
                            (
                                torch.sqrt(alpha_cumprod[i]) * node_output
                                - last_step_coords
                            )
                            / (1 - alpha_cumprod[i])
                        )
                        + torch.sqrt(beta_list[i])
                        * torch.normal(
                            mean=0.0,
                            std=1.0,
                            size=last_step_coords.shape,
                            device=last_step_coords.device,
                            dtype=last_step_coords.dtype,
                        )
                    )

                    noisy_ang = (
                        last_step_ang[:, :, :3]
                        + (sigma_square_list[i] - sigma_square_list[i - 1])
                        * (
                            (angle_output[:, :, :3] - last_step_ang[:, :, :3])
                            / (sigma_square_list[i] - sigma_square_list[0])
                        )
                        + torch.sqrt(sigma_square_list[i] - sigma_square_list[i - 1])
                        * torch.normal(
                            mean=0.0,
                            std=1.0,
                            size=angle_output[:, :, :3].shape,
                            device=angle_output.device,
                            dtype=angle_output.dtype,
                        )
                    )

                noisy_pos = noisy_pos.masked_fill(~mask_pos.bool(), 0.0)
                noisy_ang = noisy_ang.masked_fill(~mask_angle.bool(), 0.0)
                noisy_ang = torch.cat(
                    [noisy_ang, zeros_filled], dim=-1
                )  # Fill dimensions to satisfy the input of the encoder

                batch["pos"] = noisy_pos
                batch["ang"] = noisy_ang
                # logger.info(f"i={i}/{discret_t_total}, ori_angle={ori_angle[0][100]}, angle_output={noisy_ang[0][100]}")

            """--------------------------------compute loss--------------------------------"""
            """Note: Only compute pos_loss, ang_loss, angle's RMSD in the following code"""
            node_output = batch["pos"]
            angle_output = batch["ang"]

            # # compute pos data loss
            # mask_pos_full = mask_pos & pos_mask.expand(-1, -1, -1, 3) # mask_pos.shape = (B, L, 37, 3)
            # ori_pos = ori_pos.masked_fill(~mask_pos_full, 0.0)
            # node_output = node_output.masked_fill(~mask_pos_full, 0.0)
            # pos_loss += (
            #     loss_pos(
            #         node_output.to(torch.float32),
            #         ori_pos.to(torch.float32), # label_pos will loss the dim of batch
            #     )
            #     .sum(dim=-1)
            #     .item()
            # )

            # compute angle data loss only use the first 3 dimensions
            mask_angle_full = (
                mask_angle & angle_mask[:, :, :3]
            )  # mask_angle.shape = (B, L, 3)
            ori_angle = ori_angle[:, :, :3].masked_fill(~mask_angle_full, 100.0)
            angle_output = angle_output[:, :, :3].masked_fill(~mask_angle_full, 100.0)
            ang_loss += (
                loss_angle(
                    angle_output.to(torch.float32),
                    ori_angle.to(torch.float32),
                )
                .sum(dim=-1)
                .item()
            )

            # # compute rmsd for pos
            # for i in range(batch["pos"].shape[0]):
            #     coords1 = ori_pos[i][mask_pos_full[i].squeeze(-1)].view(-1, 3)
            #     logger.info(f"coords1.shape={coords1.shape}")
            #     coords2 = node_output[i][mask_pos_full[i].squeeze(-1)].view(-1, 3)
            #     natom = coords1.shape[0]
            #     assert coords1.shape == coords2.shape and natom > 0
            #     adj_matrx = linear_molecule_adjacency(natom)

            #     coords1 = coords1.cpu().numpy()
            #     coords2 = coords2.cpu().numpy()

            #     RMSD_POS = rmsd.symmrmsd(
            #         coords1,
            #         coords2,
            #         batch["x"][i, ...].cpu().numpy(),
            #         batch["x"][i, ...].cpu().numpy(),
            #         adj_matrx,
            #         adj_matrx,
            #         center=True,
            #         minimize=True,
            #     )
            #     if RMSD_POS <= 2.0:
            #         good_pos_RMSD += 1
            #     batch_pos_RMSD += RMSD_POS

            # compute rmsd for angle
            for i in range(batch["ang"].shape[0]):
                coords1 = ori_angle[i].view(-1, 3).cpu().numpy()
                coords2 = angle_output[i].view(-1, 3).cpu().numpy()
                assert coords1.shape == coords2.shape
                RMSD_ANG = np.sqrt(
                    np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=-1)))
                )
                if RMSD_ANG <= 2.0:
                    good_ang_RMSD += 1
                batch_ang_RMSD += RMSD_ANG

            # ## compute right_num
            # aa_seq = batch["x"][mask_aa.squeeze(-1).bool()]
            # print("aa_seq.shape", aa_seq.shape)
            # print("logits.shape", logits.shape)
            # right_num += (
            #     (logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq)
            #     .sum()
            #     .to(torch.float32)
            # )
            # total_res += aa_seq.view(-1).size(0)

            ## sum up
            sample_num += batch["ang"].shape[0]
            batch_num += 1
            # running_pos_loss = pos_loss / batch_num
            # running_pos_rmsd = batch_pos_RMSD / sample_num
            running_ang_loss = ang_loss / batch_num
            running_ang_rmsd = batch_ang_RMSD / sample_num

            pbar.set_postfix(
                # running_pos_rmsd=running_pos_rmsd,
                running_angle_rmsd=running_ang_rmsd,
                # pos_rmsd_good=good_pos_RMSD / sample_num,
                ang_rmsd_good=good_ang_RMSD / sample_num,
                # running_pos_loss=running_pos_loss,
                running_ang_loss=running_ang_loss,
                # type_acc=right_num / total_res,
            )

    # all reduce rmsd, sample num, and type_acc
    # RMSD_POS = torch.tensor(RMSD_POS).cuda()
    RMSD_ANG = torch.tensor(RMSD_ANG).cuda()
    sample_num = torch.tensor(sample_num).cuda()
    # total_res = torch.tensor(total_res).cuda()
    # right_num = torch.tensor(right_num).cuda()

    # torch.distributed.all_reduce(RMSD_POS)
    torch.distributed.all_reduce(RMSD_ANG)
    torch.distributed.all_reduce(sample_num)
    # torch.distributed.all_reduce(total_res)
    # torch.distributed.all_reduce(right_num)

    # RMSD_POS = RMSD_POS.item()
    RMSD_ANG = RMSD_ANG.item()
    sample_num = sample_num.item()
    # total_res = total_res.item()
    # right_num = right_num.item()

    logger.info(
        f"RMSD_ANG: {RMSD_ANG / sample_num}, total sample num: {sample_num}, residue acc: {right_num/total_res}"
    )


if __name__ == "__main__":
    main()
