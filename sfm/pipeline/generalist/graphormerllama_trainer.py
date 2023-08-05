# -*- coding: utf-8 -*-
import math
import os

import deepspeed
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.runtime.utils import see_memory_usage

import sfm.utils.mypp_engine as myPipeEngine
from sfm.criterions.copilotloss import CopilotCriterions, CopilotCriterionsPP
from sfm.logging import logger
from sfm.models.generalist.graphormer_llama import GraphormerLlamaModel
from sfm.utils.get_paranum import count_paranum
from sfm.utils.move_to_device import move_to_device
from sfm.utils.mypp_module import PipelineModule
from sfm.utils.optimizer import myAdam
from sfm.utils.set_lr import groupWarmupDecayLR


class Trainer:
    def __init__(self, args, train_data, vocab_size, freeze_list=[], unfreeze_list=[]):
        super().__init__()
        self.args = args

        if args.pipeline_model_parallel_size == 0:
            net = GraphormerLlamaModel(args, vocab_size)
            count_paranum(net)
            optimizer = myAdam(
                net,
                freeze_list=freeze_list,
                unfreeze_list=unfreeze_list,
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
        elif args.tensor_model_parallel_size == 1:
            net = GraphormerLlamaModel(args, vocab_size)
            net = PipelineModule(
                layers=net.to_layers(),
                num_stages=args.pipeline_model_parallel_size,
                loss_fn=CopilotCriterionsPP(args, vocab_size),
                partition_method="manual",
                # part_list=[0, 4, 9, 14, 19, 23, 27, 32, 37],
                # part_list=[0, 5, 7, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30, 32, 34, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 70, 73, 76, 79, 81, 85],
                part_list=[0, 9, 19, 27, 37],
                device=args.local_rank,
            )

            optimizer, param_groups = myAdam(
                net,
                freeze_list=freeze_list,
                unfreeze_list=unfreeze_list,
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
                model_parameters=param_groups,  # [p for p in net.parameters() if p.requires_grad],
                training_data=train_data,
                collate_fn=train_data.collater,
            )

            if args.infer:
                self.resume(args.resume_path)
        else:
            raise NotImplementedError

    def resume(self, resume_path, ckpt_id=None):
        if os.path.isdir(resume_path) and os.paht.exists(
            os.path.join(resume_path, "latest")
        ):
            logger.info("resume from %s" % resume_path)
            if ckpt_id is None:
                self.model_engine.load_checkpoint(resume_path)
            else:
                self.model_engine.load_checkpoint(resume_path, tag=ckpt_id)

    def train(self):
        logger.info("start training")
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
                    self.save_ckp(global_step)

                del loss
                torch.cuda.empty_cache()

                global_step += 1

    def train_pipeline(self):
        logger.info("start pipeline training")

        for global_step in range(1, self.args.total_num_steps + 1):
            self.model_engine.train_batch()

            if global_step % 1000 == 0:
                self.save_ckp(global_step)

            torch.cuda.empty_cache()

    def save_ckp(self, global_step):
        self.model_engine.save_checkpoint(
            save_dir=self.args.save_dir,
            client_state={"checkpoint_step": global_step},
        )
