from sfm.pipeline.trainer import Trainer, TrainerConfig, Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
def collater(batch):
    return torch.stack(batch)

def test_trainer():
    config = TrainerConfig(
        lr_lambda=lambda epoch: 1.0/4000 if epoch < 4000 else 1.0/math.sqrt(epoch),
        optimizer_args={
            'betas': (0.9, 0.98),
        },
        epochs=1,
    )
    print(config)
    
    model = DummyNN()
    train_data = DummyDataset()
    valid_data = DummyDataset()
    test_data = DummyDataset()
    
    trainer = Trainer(config, model=model, collater=collater, train_data=train_data, valid_data=valid_data, test_data=test_data)
    trainer.train()
