from sfm.pipeline.trainer import Trainer, TrainerConfig, Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyNN(Model):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, pred, batch):
        return F.l1_loss(pred, batch)

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(128, 10)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collater(self, batch):
        return torch.stack(batch)


def test_trainer():
    config = TrainerConfig()
    print(config)
    
    model = DummyNN()
    train_data = DummyDataset()
    valid_data = DummyDataset()
    test_data = DummyDataset()
    
    trainer = Trainer(config, model, train_data, valid_data, test_data)
    trainer.train()
