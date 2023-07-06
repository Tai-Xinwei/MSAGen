import torch
import torch.nn as nn
import deepspeed
from torch.utils.tensorboard import SummaryWriter

import os
from graphormer.models.graphormer import GraphormerModel
# from graphormer.data.dataset import PCQPreprocessedData, BatchedDataDataset
# from graphormer.data.wrapper import MyPygPCQM4MDataset
# from torch.nn.parallel import DistributedDataParallel as DDP
from graphormer.utils.move_to_device import move_to_device
from graphormer.data.dataset import DSDataLoader
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import math

# from tqdm import tqdm
# import logging
import psutil
from graphormer.utils.get_paranum import count_paranum
from graphormer.criterions.mae3d import MAE3d_criterions


class Trainer():
    def __init__(self, args, train_data, train_loader=None):
        super().__init__()
        net = GraphormerModel(args)
        count_paranum(net)
        self.args = args
        if args.rank == 0:
            self.writer = SummaryWriter("output")

        parameters = filter(lambda p: p.requires_grad, net.parameters())

        self.criterion_3d = MAE3d_criterions(args)
        self.criterion_2d = nn.L1Loss(reduction="sum")
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 50, 
        #                                     num_training_steps = total_steps)

        self.model_engine, optimizer, self.train_loader, _ = deepspeed.initialize(args=args, 
                                                                       model=net, 
                                                                       model_parameters=parameters,
                                                                    #    dist_init_required=False,
                                                                       training_data=train_data, 
                                                                       collate_fn=train_data.collater2,
                                                                      )
        
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00009, total_steps=200000, verbose=True, anneal_strategy='linear')
        # self.train_loader = DSDataLoader(train_data,
        #                                  batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
        #                                  local_rank=self.args.local_rank,
        #                                  collate_fn=train_data.collater,
        #                                  prefetch_factor=4,
        #                                  num_workers=4,
        #                                 )
        # data_sampler = DistributedSampler(train_data, shuffle=True)
        # # train_batch_sampler=torch.utils.data.BatchSampler(data_sampler, self.model_engine.train_micro_batch_size_per_gpu(), drop_last=True)
        # self.train_loader = DataLoader(train_data,
        #                                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
        #                                pin_memory=True,
        #                                sampler=data_sampler,
        #                             #    batch_sampler=train_batch_sampler,
        #                                collate_fn=train_data.collater2,
        #                                prefetch_factor=2,
        #                                num_workers=4,
        #                               )
        
    def __call__(self):
        print("start training")
        self.model_engine.module.train()
        running_loss = 0.0
        global_step = 0
        for epoch in range(self.args.epochs):
            # iterator = iter(self.train_loader)
            # for i in range(self.args.steps):
            #     try:
            #         batch_data = next(iterator)
            #     except StopIteration:
            #         # self.model_engine.save_checkpoint(save_dir=self.args.output_path, client_state={'checkpoint_step': epoch})
            #         break
            for i, batch_data in enumerate(self.train_loader):
                batch_data = move_to_device(batch_data, device=self.args.local_rank, non_blocking=True)

                model_output = self.model_engine(batch_data)
                logits, node_output = model_output[0], model_output[1]

                if self.args.add_3d:
                    loss = self.criterion_3d(batch_data, logits, node_output)
                else:
                    logits = logits[:, 0, :]
                    targets = batch_data['y']
                    loss = self.criterion_2d(logits.squeeze(-1), targets[:logits.size(0)])

                self.model_engine.backward(loss)
                self.model_engine.step()
                # self.scheduler.step()

                if (i + 1) % 1000 == 0:
                    if self.args.local_rank == 0:
                        virt_mem = psutil.virtual_memory()
                        # swap = psutil.swap_memory()
                        print("epoch={}, micro_step={}, vm %: {}, global_rank: {}".format(epoch, i, virt_mem.percent, self.args.rank))

                # if (i + 1) % 40000 == 0:
                # running_loss += loss.detach().item()

                global_step += 1

                if (global_step + 1) % 200 == 0 and self.args.rank==0:
                    self.writer.add_scalar("Loss/train", loss.detach().item(), global_step=global_step)
                    running_loss = 0

                if global_step % 10000 == 0:
                    self.model_engine.save_checkpoint(save_dir=self.args.output_path, client_state={'checkpoint_step': global_step})

                del loss
                torch.cuda.empty_cache()

            # path = os.path.join(self.args.output_path, "modelstate_epoch_{}_step_{}.pth".format(epoch, i))  
            # torch.save(self.model_engine.module.state_dict(), path)
        self.writer.flush()
        self.writer.close()

        