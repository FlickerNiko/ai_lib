import torch
import torch.nn.functional as F
import numpy as np

class PreProcessor:
    def __init__(self, config) -> None:
        self.config = config
        self.spatial_dim = config.spatial_dim
        # self.n_batch = config.n_batch
        # self.n_player = config.n_player

    

    def pre_process(self, data):
        spatial = data['spatial']
        heads_pos = data['heads_pos']        
        data['spatial'] = self.extend_spatial(spatial, heads_pos)
        return data
    

    def extend_spatial(self, spatial, heads_pos):
        # extend up, down, left, right
        height = spatial.shape[1]
        width = spatial.shape[2]        
        n_batch = spatial.shape[0]
        n_player = heads_pos.shape[1]
        spatial = torch.concat([spatial]*3, 1)
        spatial = torch.concat([spatial]*3, 2)
        spatial = F.one_hot(spatial, self.spatial_dim).movedim(3, 1)        

        # todo: is there any vectorized operations to avoid loop
        spatial_fb = []    #full batch
        for b in range(n_batch):
            spatial_fp = []    #full player
            for p in range(n_player):
                y, x = heads_pos[b, p]
                item = spatial[b,:, y+1:y+2*height, x+1:x+2*width]
                spatial_fp.append(item)
            spatial_fp = torch.stack(spatial_fp, 0)
            spatial_fb.append(spatial_fp)
        spatial_fb = torch.stack(spatial_fb, 0)

        return spatial_fb

class TestConfig:
    def __init__(self) -> None:
        self.spatial_dim = 8

if __name__ == '__main__':
    weight, height = 28, 28
    n_batch = 32
    spatial_dim = 8
    n_player = 6
    config = TestConfig()

    processor = PreProcessor(config)
    
    for i in range(100):
        heads_pos = torch.randint(height, (n_batch, n_player, 2))
        spatial = torch.randint(spatial_dim, (n_batch, weight, height))
        data = {
            'heads_pos': heads_pos,
            'spatial': spatial
        }
        data = processor.preprocess(data)
    