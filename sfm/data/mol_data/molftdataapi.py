# -*- coding: utf-8 -*-
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch_geometric.data import Data

from sfm.data.dataset import InMemoryFoundationModelDataset

from .collator import collator_ft
from .wrapper import preprocess_item, smiles2graph


class MolFTDataAPI(InMemoryFoundationModelDataset):
    def __init__(self, smiles: List[str], lable: List[float]):
        self.lable = lable
        self.smile_data_list = self.smile2data(smiles)
        super().__init__(self.smile_data_list)

    def smile2data(self, smiles):
        data_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smiles)
            for index, graph in enumerate(iter):
                data = Data()

                data.__num_nodes__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.pos = None
                data.y = self.lable[index]
                data.idx = index
                data.smile = graph["smile"]

                data_list.append(data)

        return data_list

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return preprocess_item(self.smile_data_list[idx])

    def collator(self, samples):
        return collator_ft(
            samples,
            max_node=1024,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


if __name__ == "__main__":
    pass
