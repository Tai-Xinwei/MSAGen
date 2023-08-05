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
from sfm.logging import logger as sfm_logger
from sfm.models.generalist.graphormer_llama import GraphormerLlamaModel
from sfm.utils.get_paranum import count_paranum
from sfm.utils.move_to_device import move_to_device
from sfm.utils.mypp_module import PipelineModule
from sfm.utils.optimizer import myAdam
from sfm.utils.set_lr import groupWarmupDecayLR


class Trainer3D:
    def __init__(self, args, train_data, vocab_size, freeze_list=[], unfreeze_list=[]):
        super().__init__()
        self.args = args
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
        groupWarmupDecayLR(
            optimizer,
            total_num_steps=args.total_num_steps,
            warmup_max_lr=args.max_lr,
            warmup_num_steps=args.warmup_num_steps,
        )
        see_memory_usage("Model built", force=True)

    def resume(self, resume_path, ckpt_id=None):
        if os.path.isdir(resume_path) and os.paht.exists(
            os.path.join(resume_path, "latest")
        ):
            sfm_logger.info("resume from %s" % resume_path)
            if ckpt_id is None:
                self.model_engine.load_checkpoint(resume_path)
            else:
                self.model_engine.load_checkpoint(resume_path, tag=ckpt_id)

    def train_tensor_pipeline(self):
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
                    self.save_ckp(global_step)

                del loss
                torch.cuda.empty_cache()

                global_step += 1

    def save_ckp(self, global_step):
        self.model_engine.save_checkpoint(
            save_dir=self.args.save_dir,
            client_state={"checkpoint_step": global_step},
        )
