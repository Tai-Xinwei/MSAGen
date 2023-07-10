# -*- coding: utf-8 -*-
import math
import os
from logging.loggers import sfm_logger

import deepspeed
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.runtime.utils import see_memory_usage
from models.generalist.graphormer_llama import GraphormerLlamaModel
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils.get_paranum import count_paranum
from utils.move_to_device import move_to_device
from utils.set_lr import groupWarmupDecayLR, myGroupAdam

from sfm.criterions.copilotloss import CopilotCriterions


class Trainer:
    def __init__(self, args, train_data, vocab_size):
        super().__init__()

        net = GraphormerLlamaModel(args)
        count_paranum(net)
        self.args = args
        if args.rank == 0:
            self.writer = SummaryWriter("../output")

        see_memory_usage("Model built", force=True)

        parameters = filter(lambda p: p.requires_grad, net.parameters())

        optimizer = myGroupAdam(
            net,
            mode="adaptoronly",
            lr=args.max_lr,
            betas=[0.9, 0.999],
            weight_decay=0.0,
            eps=1e-8,
        )
        scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=args.total_num_steps,
            warmup_max_lr=args.max_lr,
            warmup_num_steps=args.warmup_num_steps,
        )

        self.LlmLoss = CopilotCriterions(args, vocab_size)

        self.model_engine, _, self.train_loader, _ = deepspeed.initialize(
            args=args,
            model=net,
            model_parameters=parameters,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            training_data=train_data,
            collate_fn=train_data.collater,
        )

    def __call__(self):
        sfm_logger.info("start training")
        self.model_engine.module.train()
        global_step = 0
        for epoch in range(self.args.epochs):
            for i, batch_data in enumerate(self.train_loader):
                batch_data = move_to_device(
                    batch_data, device=self.args.local_rank, non_blocking=True
                )

                model_output = self.model_engine(batch_data)
                logits, node_output = model_output[0], model_output[1]

                if self.args.add_3d:
                    loss = self.criterion_3d(batch_data, logits, node_output)
                else:
                    logits = logits[:, 0, :]
                    targets = batch_data["y"]
                    loss = self.criterion_2d(
                        logits.squeeze(-1), targets[: logits.size(0)]
                    )

                self.model_engine.backward(loss)
                self.model_engine.step()

                if (global_step + 1) % 200 == 0 and self.args.rank == 0:
                    self.writer.add_scalar(
                        "Loss/train", loss.detach().item(), global_step=global_step
                    )

                if global_step % 10000 == 0:
                    self.model_engine.save_checkpoint(
                        save_dir=self.args.output_path,
                        client_state={"checkpoint_step": global_step},
                    )

                del loss
                torch.cuda.empty_cache()

                global_step += 1

            # path = os.path.join(self.args.output_path, "modelstate_epoch_{}_step_{}.pth".format(epoch, i))
            # torch.save(self.model_engine.module.state_dict(), path)
        self.writer.flush()
        self.writer.close()
