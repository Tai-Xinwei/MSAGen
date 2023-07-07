import copy
import itertools
from functools import lru_cache
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import ogb
# import pytorch_forecasting
import torch
from ogb.graphproppred import Evaluator
from ogb.lsc import PCQM4Mv2Evaluator
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

# from .ogb_datasets import OGBDatasetLookupTable
# from .pyg_datasets import PYGDatasetLookupTable, GraphormerPYGDataset
# from ..utils.FairseqDataset import FairseqDataset
from utils.move_to_device import move_to_device

from .collator import collator, collator_3d, collator_3d_pp, collator_ft
from .wrapper import (MyPygPCQM4MDataset, MyPygPCQM4MPosDataset,
                      PM6FullLMDBDataset, preprocess_item, smiles2graph)

# from nvidia.dali.pipeline import Pipeline
# import nvidia.dali.ops as ops
# from nvidia.dali.plugin.pytorch import DALIClassificationIterator
# from nvidia.dali.plugin.base_iterator import LastBatchPolicy


class PCQPreprocessedData:
    def __init__(self, args, dataset_name, dataset_path="../FMproj/pm6-86m-3d"):
        super().__init__()

        assert dataset_name in [
            "PCQM4M-LSC-V2",
            "PCQM4M-LSC-V2-TOY",
            "PCQM4M-LSC-V2-3D",
            "PM6-Full-3D",
        ], (
            "Only support PCQM4M-LSC-V2 or PCQM4M-LSC-V2-POS" or "PM6-Full-3D"
        )
        self.dataset_name = dataset_name
        if dataset_name == "PCQM4M-LSC-V2-3D":
            self.dataset = MyPygPCQM4MPosDataset(root=dataset_path)
            # self.dataset_version = '3D'
        elif dataset_name == "PM6-Full-3D":
            self.dataset = PM6FullLMDBDataset(
                root=dataset_path, mask_ratio=args.mask_ratio
            )
        else:
            self.dataset = MyPygPCQM4MDataset(root=dataset_path)
            # self.dataset_version = '2D'
        self.setup()

    def setup(self, stage: str = None):
        split_idx = self.dataset.get_idx_split()
        if self.dataset_name == "PCQM4M-LSC-V2":
            self.train_idx = split_idx["train"]
            # self.valid_idx = split_idx["train"]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["test-dev"]
        elif self.dataset_name == "PCQM4M-LSC-V2-3D":
            self.train_idx = split_idx["train"][0:3000000]
            self.valid_idx = split_idx["train"][3000000:]
            self.test_idx = split_idx["test-dev"]
        elif self.dataset_name == "PCQM4M-LSC-V2-TOY":
            self.train_idx = split_idx["train"][:5000]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["test-dev"]
        elif self.dataset_name == "PM6-Full-3D":
            self.train_idx = split_idx["train"]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["valid"]

        self.dataset_train = self.dataset.index_select(self.train_idx)
        self.dataset_val = self.dataset.index_select(self.valid_idx)
        self.dataset_test = self.dataset.index_select(self.test_idx)

        self.max_node = 32
        self.max_node2 = 256
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = "mae"
        self.metric_mode = "min"
        self.evaluator = (PCQM4Mv2Evaluator(),)


class PreprocessSmile(InMemoryDataset):
    def __init__(self, args, smiles):
        # super().__init__()

        self.smiles = smiles
        self.smiles2graph = smiles2graph
        self.smile_dict = {}
        self.max_node = 32
        self.max_node2 = 256
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024

        super(PreprocessSmile, self).__init__(".", transform=None, pre_transform=None)
        # self.dataset, self.smile_dict = self.process(smiles)
        # torch.save(self.dataset, '../data_processed.pt')

        #
        # for i, smile in enumerate(smiles):
        # self.smile_dict[i] = smile

        self.dataset = torch.load(self.processed_paths[0])

        # print(len(self.dataset));exit()

    # @property
    # def raw_file_names(self):
    #     return '../smile_list.pt'

    @property
    def processed_file_names(self):
        return "data_processed.pt"

    def process(self):
        data_list = []
        # smile_dict = {}
        length = len(self.smiles)
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, self.smiles)
            for idx, graph in tqdm(enumerate(iter), total=length):
                # for idx, smile in enumerate(smiles):
                data = Data()
                # graph = self.smiles2graph(smile)

                data.__num_nodes__ = int(graph["num_nodes"])
                if data.__num_nodes__ > self.max_node2:
                    continue

                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.pos = None
                data.y = None
                data.idx = idx
                data.smile = graph["smile"]

                # data_list.append(preprocess_item(data, mask_ratio=0.0))
                data_list.append(data)

                self.smile_dict[idx] = graph["smile"]

        torch.save(data_list, self.processed_paths[0])
        return data_list


def smile2data(smile, index):
    data = Data()
    graph = smiles2graph(smile)

    data.__num_nodes__ = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
    data.pos = None
    data.y = None
    data.idx = index
    data.smile = graph["smile"]

    data = preprocess_item(data, mask_ratio=0.0)

    return data


class BatchedDataDataset(torch.utils.data.Dataset):
    # class BatchedDataDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        dataset_version="2D",
        max_node=32,
        max_node2=128,
        multi_hop_max_dist=5,
        spatial_pos_max=1024,
        args=None,
        ft=False,
        infer=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.max_node2 = max_node2
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

        self.dataset_version = dataset_version
        assert self.dataset_version in ["2D", "3D"]
        self.args = args
        self.ft = ft
        self.infer = infer

    def __getitem__(self, index):
        # item = preprocess_item(self.dataset[int(index)], mask_ratio=self.args.mask_ratio)
        item = self.dataset[int(index)]
        return item

        # return next(self.generator())

    # def __iter__(self):
    #     # return self.generator()
    #     batch = []
    #     for data in self.dataset:
    #         batch.append(data)
    #         if len(batch) == 128:
    #             yield self.collater(batch)
    #             batch = []

    # def _infinite(self):
    #     #train_list_path = os.path.join(cfg.dataset_dir, "list", "pdb40_train.txt")
    #     # train_list_path = '/home/jiaxi/EDP/DF/v1.d256x8/protein_correct.txt'
    #     # train_list = open(train_list_path).readlines()
    #     train_list = [_.strip() for _ in self.dataset]
    #     ids = np.arange(len(train_list))
    #     while True:
    #         c = np.random.choice(ids) #generate a random integer from 0~ids-1
    #         #c = train_list.index('1h0a_A')
    #         yield train_list[c]

    # def generator(self):
    #     batch = []
    #     for data in self.dataset:
    #         batch.append(data)
    #         if len(batch) == 128:
    #             yield batch
    #             batch = []

    # def generator(self):
    #     for data in self.dataset:
    #         batch_samples = []
    #         while len(batch_samples) > 0:
    #             yield batch_samples.pop()

    # def generator(self):
    # while True:
    # idx = np.random.choice(len(self.dataset), 1)
    # yield self.dataset[int(idx)]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.args is not None and self.args.pipeline_parallelism > 0:
            collator_fn = collator_3d_pp
        else:
            collator_fn = collator if (self.dataset_version == "2D") else collator_3d

        return collator_fn(
            samples,
            min_node=-1,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
            infer=self.args.infer,
        )

    def collater2(self, samples):
        if self.args is not None and self.args.pipeline_parallelism > 0:
            collator_fn = collator_3d_pp
        else:
            collator_fn = collator if (self.dataset_version == "2D") else collator_3d

        return collator_fn(
            samples,
            min_node=-1,
            max_node=self.max_node2,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )

    def collaterft(self, samples):
        return collator_ft(
            samples,
            max_node=1024,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


class DSDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        local_rank,
        pin_memory=True,
        collate_fn=None,
        num_workers=None,
        data_sampler=None,
        prefetch_factor=2,
        data_parallel_world_size=None,
        data_parallel_rank=None,
        dataloader_drop_last=True,
    ):
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor

        if local_rank >= 0:
            if data_sampler is None:
                data_sampler = DistributedSampler(
                    dataset=dataset,
                    num_replicas=data_parallel_world_size,
                    rank=data_parallel_rank,
                )
            device_count = 1
        else:
            if data_sampler is None:
                data_sampler = RandomSampler(dataset)
            device_count = torch.cuda.device_count()
            batch_size *= device_count

        if num_workers is None:
            num_workers = 2 * device_count

        self.num_workers = num_workers
        self.data_sampler = data_sampler
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.device_count = device_count
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.data = None
        self.dataloader_drop_last = dataloader_drop_last
        self.post_process_func = None

        if self.dataloader_drop_last:
            self.len = len(self.data_sampler) // self.batch_size
        else:
            from math import ceil

            self.len = ceil(len(self.data_sampler) / self.batch_size)

    def __iter__(self):
        self._create_dataloader()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        return next(self.data)

    def _create_dataloader(self):
        if self.collate_fn is None:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                sampler=self.data_sampler,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=self.dataloader_drop_last,
            )
        else:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                sampler=self.data_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=self.dataloader_drop_last,
            )
        self.data = (x for x in self.dataloader)

        return self.dataloader


# class CacheAllDataset(BaseWrapperDataset):
#     def __init__(self, dataset):
#         super().__init__(dataset)

#     @lru_cache(maxsize=None)
#     def __getitem__(self, index):
#         return self.dataset[index]

#     def collater(self, samples):
#         return self.dataset.collater(samples)

# class EpochShuffleDataset(BaseWrapperDataset):
#     def __init__(self, dataset, size, seed):
#         super().__init__(dataset)
#         self.size = size
#         self.seed = seed
#         self.set_epoch(1)

#     def set_epoch(self, epoch):
#         with data_utils.numpy_seed(self.seed + epoch - 1):
#             self.sort_order = np.random.permutation(self.size)

#     def ordered_indices(self):
#         return self.sort_order

#     @property
#     def can_reuse_epoch_itr_across_epochs(self):
#         return False


class TargetDataset:
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)


# class TargetPM6Dataset(FairseqDataset):
#     def __init__(self, dataset):
#         super().__init__()
#         self.dataset = dataset

#     @lru_cache(maxsize=16)
#     def __getitem__(self, index):
#         return torch.Tensor((
#             self.dataset[index].y1, self.dataset[index].y2,
#             self.dataset[index].y3, self.dataset[index].y4,
#             self.dataset[index].y5, self.dataset[index].y6,
#             self.dataset[index].y7, self.dataset[index].y8,
#             self.dataset[index].y9, self.dataset[index].y10
#         ))

#     def __len__(self):
#         return len(self.dataset)

#     def collater(self, samples):
#         return torch.stack(samples, dim=0)


class data_prefetcher:
    def __init__(self, loader, args):
        self.args = args
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = move_to_device(
                self.batch, self.args.local_rank, non_blocking=True
            )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


# class DataLoaderX(torch.utils.data.dataloader.DataLoader):
#         def __iter__(self):
#             return BackgroundGenerator(super().__iter__())


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class myDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        self.batch_sampler = _RepeatSampler(self.batch_sampler)

        super().__init__(*args, **kwargs, batch_sampler=self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


# class TrainPipeline(Pipeline):
#     def __init__(self, dataset, batch_size, num_threads, device_id, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
#         super(TrainPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=4)
#         mode = 'gpu'
#         # self.decode = ops.decoders.Image(device='mixed')
#         self.max_node = max_node
#         self.multi_hop_max_dist = multi_hop_max_dist
#         self.spatial_pos_max = spatial_pos_max


#     def define_graph(self, items):
#         items = [item for item in items if item is not None and item.x.size(0) <= self.max_node]
#         items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
#                 item.out_degree, item.x, item.edge_input[:, :, :self.multi_hop_max_dist, :], item.y, item.pos, item.node_mask
#                 ) for item in items]

#         idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, node_masks = zip(*items)


#         for idx, _ in enumerate(attn_biases):
#             attn_biases[idx][1:, 1:][spatial_poses[idx] >= self.spatial_pos_max] = float('-inf')

#         max_node_num = max(i.size(0) for i in xs)
#         max_dist = max(i.size(-2) for i in edge_inputs)
#         y = torch.cat(ys)
#         x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
#         edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
#         attn_bias = torch.cat([pad_attn_bias_unsqueeze(
#             i, max_node_num + 1) for i in attn_biases])
#         attn_edge_type = torch.cat(
#             [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
#         spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
#                             for i in spatial_poses])
#         in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
#                             for i in in_degrees])

#         pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
#         node_mask = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in node_masks])

#         # @ Roger added
#         node_type_edges = []
#         for idx in range(len(items)):
#             node_atom_type = items[idx][6][:, 0]
#             n_nodes = items[idx][6].shape[0]
#             node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
#             node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
#             node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
#             node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
#             node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
#             node_atom_edge = convert_to_single_emb(node_atom_edge)
#             node_type_edges.append(node_atom_edge.long())

#         node_type_edge = torch.cat(node_type_edges)

#         return dict(
#             idx=torch.LongTensor(idxs),
#             # idx=idx_n,
#             attn_bias=attn_bias,
#             attn_edge_type=attn_edge_type,
#             spatial_pos=spatial_pos,
#             in_degree=in_degree,
#             out_degree=in_degree, # for undirected graph
#             x=x,
#             edge_input=edge_input,
#             y=y,
#             pos=pos,
#             node_type_edge=node_type_edge,
#             node_mask=node_mask,
#         )

# def get_dali_iter(dataset, mode, batch_size, num_threads, device_id):
#     pipe_train = TrainPipeline(dataset, batch_size, num_threads, device_id, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20)
#     pipe_train.build()
#     # DALIClassificationIterator: Returns 2 outputs (data and label) in the form of PyTorch’s Tensor, 即DataLoader
#     train_loader = DALIClassificationIterator(pipe_train, size=pipe_train.epoch_size('Reader'),
#                                                 last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
#     return train_loader

if __name__ == "__main__":
    dataset = PCQPreprocessedData("PCQM4M-LSC-V2")
    print(len(dataset))
