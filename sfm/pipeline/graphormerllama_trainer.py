# -*- coding: utf-8 -*-
import math
import os

import deepspeed
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import utils.mypp_engine as myPipeEngine
from criterions.copilotloss import CopilotCriterions, CopilotCriterionsPP
from deepspeed.runtime.utils import see_memory_usage
from models.generalist.graphormer_llama import GraphormerLlamaModel
from sfmlogging.loggers import sfm_logger
from utils.get_paranum import count_paranum
from utils.move_to_device import move_to_device
from utils.mypp_module import PipelineModule
from utils.set_lr import groupWarmupDecayLR, myAdam


class Trainer:
    def __init__(self, args, train_data, vocab_size):
        super().__init__()
        self.args = args

        if args.pipeline_parallelism == 0:
            net = GraphormerLlamaModel(args, vocab_size)
            count_paranum(net)
            optimizer = myAdam(
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
            see_memory_usage("Model built", force=True)

            parameters = filter(lambda p: p.requires_grad, net.parameters())

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
        else:
            net = GraphormerLlamaModel(args, vocab_size)
            net = PipelineModule(
                layers=net.to_layers(),
                num_stages=args.pipeline_parallelism,
                loss_fn=CopilotCriterionsPP(args, vocab_size),
                partition_method="manual",
                part_list=[0, 3, 8, 13, 18, 22, 26, 31, 36],
            )
            optimizer = myAdam(
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
            see_memory_usage("Model built", force=True)

            self.model_engine, _, _, _ = myPipeEngine.initialize(
                args=args,
                model=net,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                model_parameters=[p for p in net.parameters() if p.requires_grad],
                training_data=train_data,
                collate_fn=train_data.collater,
            )

            if args.infer:
                self.resume(args.resume_path)

    def resume(self, resume_path, ckpt_id=None):
        sfm_logger.info("resume from %s" % resume_path)
        if ckpt_id is None:
            self.model_engine.load_checkpoint(resume_path)
        else:
            self.model_engine.load_checkpoint(resume_path, tag=ckpt_id)

    def train(self):
        sfm_logger.info("start training")
        global_step = 1
        for epoch in range(self.args.epochs):
            for i, batch_data in enumerate(self.train_loader):
                batch_data = move_to_device(
                    batch_data, device=self.args.local_rank, non_blocking=True
                )

                logits = self.model_engine(batch_data)

                loss = self.LlmLoss(logits, batch_data["labels"])

                self.model_engine.backward(loss)
                self.model_engine.step()

                if global_step % 10000 == 0:
                    self.model_engine.save_checkpoint(
                        save_dir=self.args.output_path,
                        client_state={"checkpoint_step": global_step},
                    )

                del loss
                torch.cuda.empty_cache()

                global_step += 1

    def train_pipeline(self):
        sfm_logger.info("start pipeline training")

        for global_step in range(1, self.args.total_num_steps + 1):
            # self.save_ckp(global_step); exit(0)

            self.model_engine.train_batch()

            if global_step % 10000 == 0:
                self.save_ckp(global_step)

    def save_ckp(self, global_step):
        self.model_engine.save_checkpoint(
            save_dir=self.args.output_path,
            client_state={"checkpoint_step": global_step},
        )
