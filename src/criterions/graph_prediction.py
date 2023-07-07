# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.atom_loss_coeff = task.cfg.atom_loss_coeff
        self.pos_loss_coeff = task.cfg.pos_loss_coeff

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        # add gaussian noise
        # ori_pos = sample['net_input']['batched_data']['pos']
        # noise = torch.randn(ori_pos.shape).to(ori_pos) * 0.01
        # noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
        # noise = noise.masked_fill_(noise_mask, 0.0)
        # sample['net_input']['batched_data']['pos'] = ori_pos + noise

        model_output = model(**sample["net_input"])
        logits, node_output = model_output[0], model_output[1]
        # logits = logits[:,0,:]
        # targets = model.get_targets(sample, [logits])

        # loss = nn.L1Loss(reduction='sum')(logits, targets)

        node_mask = sample["net_input"]["batched_data"]["node_mask"].squeeze(-1).bool()
        logits = logits[:, 1:, :][node_mask]
        targets = sample["net_input"]["batched_data"]["x"][:, :, 0][node_mask]
        loss = (
            modules.cross_entropy(
                logits.view(-1, logits.size(-1)).to(torch.float32),
                targets.view(-1),
                reduction="sum",
                ignore_index=0,
            )
            * self.atom_loss_coeff
        )

        node_output = node_output[node_mask]

        # try:
        #     node_output = node_output[node_mask]
        # except:
        #     logging_output = {
        #         "loss": loss.data,
        #         "node_output_loss": 0,
        #         "total_loss": loss.data,
        #         "sample_size": node_mask.sum(),
        #         "nsentences": sample_size,
        #         "ntokens": natoms,
        #     }

        #     return loss, node_mask.sum(), logging_output

        ori_pos = sample["net_input"]["batched_data"]["pos"][node_mask]

        node_output_loss = (
            nn.L1Loss(reduction="sum")(
                node_output.to(torch.float32), ori_pos.to(torch.float32)
            ).sum(dim=-1)
            * self.pos_loss_coeff
        )
        # node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), ori_pos.to(torch.float32))).sum(dim=-1) * self.pos_loss_coeff
        logging_output = {
            "loss": loss.data,
            "node_output_loss": node_output_loss.data,
            "total_loss": loss.data + node_output_loss.data,
            "sample_size": node_mask.sum(),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss + node_output_loss, node_mask.sum(), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        node_output_loss_sum = sum(
            log.get("node_output_loss", 0) for log in logging_outputs
        )
        total_loss_sum = sum(log.get("total_loss", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        metrics.log_scalar(
            "node_output_loss", node_output_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "total_loss", total_loss_sum / sample_size, sample_size, round=6
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion(
    "shortest_path_distance_prediction", dataclass=GraphPredictionConfig
)
class SPDPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])[1]["spatial_out"]
        logits = torch.abs(logits)
        logits += 1.0
        targets = sample["net_input"]["batched_data"]["spatial_pos"]
        logits[targets == 0] = 0
        # targets[targets != 0] = targets[targets != 0] - 1.0
        # targets = model.get_targets(sample, [logits])

        loss = nn.L1Loss(reduction="sum")(logits.squeeze(-1), targets)

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_prediction_multi_task", dataclass=GraphPredictionConfig)
class GraphPredictionMultiTaskLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.

    task_loss_mapping:
    data.y1 = torch.Tensor([homolumogap])
    data.y2 = torch.Tensor([homo])
    data.y3 = torch.Tensor([lumo])
    data.y4 = torch.Tensor([total_energy])
    data.y5 = torch.Tensor([smallest_energy])
    data.y6 = torch.Tensor([largest_energy])
    data.y7 = torch.Tensor([enthalpy])
    data.y8 = torch.Tensor([total_dipole_moment])
    data.y9 = torch.Tensor([mo_number])
    data.y10 = torch.Tensor([hm_number])

    """

    task_loss_mapping = {
        0: "homo_lumo_gap",
        1: "homo",
        2: "lumo",
        3: "total_energy",
        4: "smallest_energy",
        5: "largest_energy",
        6: "enthalpy",
        7: "total_dipole_moment",
        8: "mo_number",
        9: "hm_number",
    }

    loss_task_mapping = {v: k for k, v in task_loss_mapping.items()}

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.regression_num_tasks = getattr(task.cfg, "regression_num_tasks", 8)
        self.classification_num_tasks = getattr(task.cfg, "classification_num_tasks", 2)
        self.regression_tasks_list = getattr(task.cfg, "regression_tasks_list", None)
        self.classification_tasks_list = getattr(
            task.cfg, "classification_tasks_list", None
        )
        assert self.regression_tasks_list is not None, "Must set regression tasks list"
        assert (
            self.classification_tasks_list is not None
        ), "Must set classification tasks list"
        if (
            self.regression_tasks_list == "None"
            and self.classification_tasks_list == "None"
        ):
            raise RuntimeError("regression or classification must be set")
        self.regression_tasks_list = self.regression_tasks_list.split(",")
        self.classification_tasks_list = self.classification_tasks_list.split(",")
        self.regression_tasks_list = [
            item for item in self.regression_tasks_list if item != "None"
        ]
        self.classification_tasks_list = [
            item for item in self.classification_tasks_list if item != "None"
        ]
        for item in self.regression_tasks_list + self.classification_tasks_list:
            assert (
                item in self.loss_task_mapping.keys()
            ), f"regression/classification task names must be in the supported list {self.loss_task_mapping.keys()}"

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])[0]
        targets = model.get_targets(sample, None)
        loss = 0

        loss_logging_output = {}
        for i in range(self.regression_num_tasks):
            cur_regression_task_idx = self.loss_task_mapping[
                self.regression_tasks_list[i]
            ]
            cur_loss = nn.L1Loss(reduction="sum")(
                logits[i].squeeze(), targets[:, cur_regression_task_idx]
            )
            loss_logging_output[self.regression_tasks_list[i]] = cur_loss.data
            loss = loss + cur_loss / (
                self.regression_num_tasks + self.classification_num_tasks
            )
        for i in range(self.classification_num_tasks):
            cur_classification_task_idx = self.loss_task_mapping[
                self.classification_tasks_list[i]
            ]
            cur_loss = nn.CrossEntropyLoss(reduction="sum")(
                logits[i + self.regression_num_tasks],
                targets[:, cur_classification_task_idx].long(),
            )
            loss_logging_output[self.classification_tasks_list[i]] = cur_loss.data
            loss = loss + cur_loss / (
                self.regression_num_tasks + self.classification_num_tasks
            )

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        logging_output.update(loss_logging_output)
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)

        for k, i in GraphPredictionMultiTaskLoss.task_loss_mapping.items():
            cur_sum = sum(log.get(i, 0) for log in logging_outputs)
            metrics.log_scalar(i, cur_sum / sample_size, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
