import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        # super().__init__()
        self.config = config                
        self.conv1 = nn.Conv2d(config.spatial_dim, 32, 5, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        self.fc1 = nn.Linear(config.scalar_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(128, 128)
        self.fcv = nn.Linear(128, 1)
        self.fca = nn.Linear(128, config.action_dim)
    
    def forward(self, spatial, scalar):
        
        hs = F.relu(self.conv1(spatial))
        hs = self.conv2(hs)
        hs = torch.amax(hs, (2,3))
        hc = F.relu(self.fc1(scalar))
        hc = F.relu(self.fc2(hc))
        h = torch.concat((hs, hc), -1)        
        h = F.relu(self.fc3(h))        
        v = self.fcv(h)
        a = self.fca(h)
        return a, v

class TestConfig:
    def __init__(self) -> None:
        self.spatial_dim = 8
        self.scalar_dim = 32
        self.action_dim = 4
        
if __name__ == '__main__':
    config = TestConfig()
    model = Model(config)
    for i in range(100):
        n_batch = 32
        
        scalar = torch.rand((n_batch,config.scalar_dim))
        spatial = torch.rand((n_batch, config.spatial_dim, 32, 32))
        
        a, v = model.forward(spatial, scalar)
        print(a)
        print(v)

