# -*- coding: utf-8 -*-
import json
import math
import os
import sys
from typing import Callable

import numpy as np
import torch
from torch import distributed as dist
from torch.autograd import grad

# from spyrmsd import molecule, rmsd
from tqdm import tqdm

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import warnings

from sfm.criterions.mae3d import ProteinMAE3dCriterions
from sfm.data.prot_data.dataset import BatchedDataDataset, ProteinLMDBDataset
from sfm.logging import logger
from sfm.models.tox.modules.physics import VESDE
from sfm.models.tox.tox_config import TOXConfig
from sfm.models.tox.toxmodel import TOXModel, TOXPDEModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device

warnings.filterwarnings("ignore")


def linear_molecule_adjacency(n_atoms):
    """
    Args:
        - n_atoms: number of atoms
    Returns:
        - neighbourhood matrix of a linear molecule
    """
    adjacency_matrix = np.zeros((n_atoms, n_atoms))
    np.fill_diagonal(adjacency_matrix[1:], 1)
    np.fill_diagonal(adjacency_matrix[:, 1:], 1)
    return adjacency_matrix


def load_data(args):
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

    return dataloader


def load_model(args):
    model = TOXModel(args, loss_fn=ProteinMAE3dCriterions, load_ckpt=True).cuda()
    logger.info(f"load model from {args.loadcheck_path}")

    model.eval()

    return model


def get_VP(model: TOXPDEModel, t_total_steps=1000):
    """
    return the coefficient of mean and sigma^2 in VP
    """
    beta_list = model.net.diffnoise.beta_list.cuda()
    alpha_cumprod = model.net.diffnoise.alphas_cumprod.cuda()

    return alpha_cumprod, beta_list


def get_VE(t_total_steps=1000):
    """
    return the coefficient of mean and sigma^2 in VE
    """
    ve_sde = VESDE()
    # t : 0/(N-1), 1/(N-1), 2/(N-1) ... (N-1)/(N-1)
    t = torch.linspace(0, 1, t_total_steps)

    beta = ve_sde.sigma_term(t) ** 2
    alpha = torch.ones_like(beta)

    beta = beta.cuda()
    alpha = alpha.cuda()

    return alpha, beta


def compute_angle_loss(loss, label, pred, mask):
    # compute angle data loss only use the first 3 dimensions
    ori_angle = label[mask]
    angle_pred = pred[mask]
    angle_loss = loss(
        angle_pred.to(torch.float32),
        ori_angle.to(torch.float32),
    )

    return angle_loss


def compute_mae(label, pred, mask):
    label_masked = label.masked_fill(~mask, 0.0)
    pred_masked = pred.masked_fill(~mask, 0.0)
    mae = torch.sum(torch.abs(label_masked - pred_masked), dim=(-1, -2))
    mae = mae / torch.sum(mask)
    mae_mean = torch.mean(mae)

    return mae, mae_mean


def compute_angle_rmsd(label, pred, mask, good_thresh=0.5):
    good_angle_RMSD = 0
    batch_angle_RMSD = 0.0
    ori_angle = label.masked_fill(~mask, 0.0).cpu().numpy()
    angle_pred = pred.masked_fill(~mask, 0.0).cpu().numpy()
    for i in range(label.shape[0]):
        ori_angle_i = ori_angle[i].reshape(-1, 3)
        angle_pred_i = angle_pred[i].reshape(-1, 3)
        RMSD_ANG = np.sqrt(np.mean(np.sum((ori_angle_i - angle_pred_i) ** 2, axis=-1)))
        if RMSD_ANG <= good_thresh:
            good_angle_RMSD += 1
        batch_angle_RMSD += RMSD_ANG

    return batch_angle_RMSD, good_angle_RMSD


def derivative(t: torch.Tensor, func: Callable):
    with torch.enable_grad():
        t.requires_grad_(True)
        derivative = grad(func(t).sum(), t, create_graph=False)[0].detach()
        t.requires_grad_(False)
    return derivative


def export_json(file, tensor_dict):
    with open(file, "w") as f:
        for key, value in tensor_dict.items():
            tensor_dict[key] = value.cpu().numpy().tolist()
        json.dump(tensor_dict, f)


@cli(DistributedTrainConfig, TOXConfig)
def main(args) -> None:
    # init distributed
    torch.distributed.init_process_group()
    torch.cuda.set_device(args.local_rank)
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(profile="full")

    # load_data
    dataloader = load_data(args)

    # load model
    model = load_model(args)

    # X~N[alpha * X_0, beta * I]
    t_total_steps = args.num_timesteps

    if args.diffmode == "VE":
        alpha, beta = get_VE(t_total_steps)
    else:
        alpha, beta = get_VP(model, t_total_steps)

    loss_angle = torch.nn.MSELoss(reduction="mean")
    batch_angle_RMSD = 0.0
    good_angle_RMSD = 0.0
    sample_num = 0
    batch_num = 0
    angle_loss = 0

    # sample
    pbar = tqdm(dataloader)
    for batch in pbar:
        batch = move_to_device(batch, args.local_rank)
        with torch.no_grad():
            """--------------------------------Mask--------------------------------"""
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

            angle_mask = batch["ang_mask"].bool()
            mask_angle = mask_pos.squeeze(-1)

            padding_mask = residue_seq.eq(1)

            unified_angle_mask = (
                angle_mask[:,] & mask_angle & (~padding_mask.unsqueeze(-1))
            )
            unified_angle_mask = unified_angle_mask[:, :, :3]

            """--------------------------------Save ground truth x_0--------------------------------"""
            ori_pos = batch["pos"].clone()
            ori_angle = batch["ang"].clone()

            """--------------------------------Sample X_T from Gaussion--------------------------------"""
            VESDE()

            batch["ang"] = torch.normal(
                mean=0.0,
                std=1.0,
                size=batch["ang"].shape,
                device=batch["ang"].device,
                dtype=batch["ang"].dtype,
            )

            const_filled = ori_angle[:, :, 3:]

            time_pos = torch.tensor([1.0], device=ori_pos.device)
            time_aa = torch.tensor([0.0], device=ori_pos.device)

            """--------------------------------Solve Reverse SDE--------------------------------"""

            # iterate over time steps to compute the x0 with weighted sum xt
            for i in range(0, t_total_steps)[::-1]:
                # We will compute X_i this time:
                # i / (t_total_steps - 1)
                t1 = (i + 1) / t_total_steps
                last_angle = batch["ang"]

                # Now the output is epsilon
                logits, node_output, epsilon_output, _, _, _, _ = model.net.sample(
                    batch,
                    time_step=t1,
                    mask_aa=mask_aa,
                    mask_pos=mask_pos,
                    mask_angle=mask_angle,
                    padding_mask=padding_mask,
                    mode_mask=mode_mask,
                    time_pos=time_pos,
                    time_aa=time_aa,
                )

                if args.diffmode == "epsilon":
                    hat_alpha_t = alpha[i]
                    hat_alpha_t_1 = 1.0 if i == 0 else alpha[i - 1]
                    alpha_t = hat_alpha_t / hat_alpha_t_1
                    beta_t = 1 - alpha_t
                    beta_tilde_t = (
                        0.0
                        if i == 0
                        else (
                            (1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t
                        ).sqrt()
                    )

                    epsilon = torch.randn_like(epsilon_output)

                    noisy_angle = (
                        last_angle[:, :, :3]
                        - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * epsilon_output
                    ) / alpha_t.sqrt() + beta_tilde_t * epsilon

                else:
                    # TODO:whether there is -1 or 1
                    last_score = -epsilon_output / torch.sqrt(beta[i + 1])

                    if args.ode_mode:
                        # noisy_angle = (
                        #     last_angle[:, :, :3]
                        #     + 1 / 2 * (beta[i + 1] - beta[i]) * last_score
                        # )
                        noisy_angle = (
                            last_angle[:, :, :3] + (beta[i + 1] - beta[i]) * last_score
                        )
                    else:
                        z = torch.normal(
                            mean=0.0,
                            std=1.0,
                            size=last_angle[:, :, :3].shape,
                            device=last_angle.device,
                            dtype=last_angle.dtype,
                        )

                        # # predictor without corrector
                        # noisy_angle = (
                        #     last_angle[:, :, :3]
                        #     + 25 * (beta[i + 1] - beta[i]) * last_score
                        #     + torch.sqrt(beta[i + 1] - beta[i]) * standard_gs
                        # )

                        # # auto-diff
                        # dt = t1 - t0
                        # noisy_angle = (
                        #     last_angle[:, :, :3]
                        #     + 3 / 2 * (beta[i + 1] - beta[i]) * last_score
                        #     + derivative(
                        #         torch.tensor(t1).cuda(), lambda x: ve_sde.sigma_term(x) ** 2
                        #     )
                        #     * dt
                        #     * z
                        # )

                        # ancestral sampling
                        noisy_angle = (
                            last_angle[:, :, :3]
                            + 25 * (beta[i + 1] - beta[i]) * last_score
                            + torch.sqrt(
                                (beta[i] / beta[i + 1]) * (beta[i + 1] - beta[i])
                            )
                            * z
                        )

                noisy_angle = noisy_angle.masked_fill(~mask_angle.bool(), 0.0)
                noisy_angle = torch.cat([noisy_angle, const_filled], dim=-1)

                batch["ang"] = noisy_angle

                if i % 50 == 0:
                    mae, mae_mean = compute_mae(
                        label=ori_angle[:, :, :3],
                        pred=batch["ang"][:, :, :3],
                        mask=unified_angle_mask,
                    )

                    angle_loss_curr = compute_angle_loss(
                        loss=loss_angle,
                        label=ori_angle[:, :, :3],
                        pred=batch["ang"][:, :, :3],
                        mask=unified_angle_mask,
                    )

                    if args.local_rank == 0:
                        logger.info(
                            f"t_step: {i} rank: {args.local_rank} mae: {mae} mae_mean: {mae_mean} mse_mean: {angle_loss_curr}"
                        )

            """--------------------------------exsport json--------------------------------"""
            tensor_dict = {
                "ori_aa": batch["x"],
                "ori_angle": ori_angle,
                "pred_angle": batch["ang"],
                "pos": ori_pos,
                "ori_psi": ori_angle[:, :-1, 0],
                "ori_phi": ori_angle[:, 1:, 1],
                "ori_omg": ori_angle[:, 1:, 2],
                "mae": mae,
                "mae_mean": mae_mean,
                "mse_mean": angle_loss_curr,
            }
            export_json(
                f"output_protein_{args.local_rank}_batch_{batch_num}.json", tensor_dict
            )
            """--------------------------------compute loss--------------------------------"""
            angle_loss_curr = compute_angle_loss(
                loss=loss_angle,
                label=ori_angle[:, :, :3],
                pred=batch["ang"][:, :, :3],
                mask=unified_angle_mask,
            ).item()
            angle_loss += angle_loss_curr

            """--------------------------------compute rmsd--------------------------------"""
            batch_angle_RMSD_curr, good_angle_RMSD_curr = compute_angle_rmsd(
                label=ori_angle[:, :, :3],
                pred=batch["ang"][:, :, :3],
                mask=unified_angle_mask,
                good_thresh=1.0,
            )

            batch_angle_RMSD += batch_angle_RMSD_curr
            good_angle_RMSD += good_angle_RMSD_curr

            ## sum up
            sample_num += batch["ang"].shape[0]
            batch_num += 1

            pbar.set_postfix(
                running_angle_loss=angle_loss / batch_num,
                running_angle_rmsd=good_angle_RMSD / sample_num,
                good_angle_RMSD=good_angle_RMSD / sample_num,
            )

            # if args.local_rank == 0:
            #     wandb.log(
            #         {
            #             "angle_loss": angle_loss / batch_num,
            #             "angle_rmsd": batch_angle_RMSD / sample_num,
            #             "good_angle_rmsd": good_angle_RMSD / sample_num,
            #             "total_sample_num": sample_num,
            #             "batch_num": batch_num,
            #         }
            #     )

    # all reduce the result
    angle_loss = torch.tensor(angle_loss).cuda()
    batch_angle_RMSD = torch.tensor(batch_angle_RMSD).cuda()
    good_angle_RMSD = torch.tensor(good_angle_RMSD).cuda()
    sample_num = torch.tensor(sample_num).cuda()
    batch_num = torch.tensor(batch_num).cuda()

    dist.all_reduce(angle_loss)
    dist.all_reduce(batch_angle_RMSD)
    dist.all_reduce(good_angle_RMSD)
    dist.all_reduce(sample_num)
    dist.all_reduce(batch_num)

    angle_loss = angle_loss.item()
    batch_angle_RMSD = batch_angle_RMSD.item()
    good_angle_RMSD = good_angle_RMSD.item()
    sample_num = sample_num.item()
    batch_num = batch_num.item()

    logger.info(
        f"angle_loss: {angle_loss / batch_num}, total sample num: {sample_num}, angle_rmsd: {batch_angle_RMSD / sample_num}, good_rmsd: {good_angle_RMSD / sample_num}"
    )


if __name__ == "__main__":
    try:
        os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
