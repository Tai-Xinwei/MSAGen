from sfm.pipeline.trainer import Trainer, TrainerConfig, Model, ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data.dataset import Dataset

import tempfile

class DummyNN(Model):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, pred, batch) -> ModelOutput:
        loss = F.l1_loss(pred, batch)
        return ModelOutput(loss=loss, log_output={'l2': F.mse_loss(pred, batch).item()})

    def config_optimizer(self) -> tuple[Optimizer, LRScheduler]:
        optimizer =  torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        
        return optimizer, lr_scheduler

class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(128, 10)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
def collater(batch):
    return torch.stack(batch)


def test_trainer():
    with tempfile.TemporaryDirectory() as save_dir:
        config = TrainerConfig(
            epochs=1,
            save_dir=save_dir
        )
        print(config)
        
        model = DummyNN()
        train_data = DummyDataset()
        valid_data = DummyDataset()
        test_data = DummyDataset()
        
        trainer = Trainer(config, model=model, collater=collater, train_data=train_data, valid_data=valid_data, test_data=test_data)
        trainer.train()