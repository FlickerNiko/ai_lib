import torch
import torch.nn.functional as F
import numpy as np

class PreProcessor:
    def __init__(self, config) -> None:
        self.config = config
        self.spatial_dim = config.spatial_dim
        # self.n_batch = config.n_batch
        self.n_player = config.n_player
        self.scalar_dim = config.scalar_dim



    def pre_process(self, data):
        spatial = data['spatial']
        heads_pos = data['heads_pos']
        data['spatial'] = self.extend_spatial(spatial, heads_pos)
        del data['heads_pos']
        # stub
        data['scalar'] = np.zeros((self.n_player, self.scalar_dim), dtype=np.float32)
        return data

    def extend_spatial(self, spatial, heads_pos):
        # extend up, down, left, right
        height = spatial.shape[0]
        width = spatial.shape[1]
        n_player = heads_pos.shape[0]
        spatial = np.concatenate([spatial]*3, 0)
        spatial = np.concatenate([spatial]*3, 1)
        spatial = np.moveaxis(np.eye(self.spatial_dim, dtype = np.float32)[spatial], 2, 0)

        # todo: is there any vectorized operations to avoid loop

        spatial_fp = []    #full player
        for p in range(n_player):
            y, x = heads_pos[p]
            item = spatial[:, y+1:y+2*height, x+1:x+2*width]
            spatial_fp.append(item)
        spatial_fp = np.stack(spatial_fp, 0)
        return spatial_fp

class TestConfig:
    def __init__(self) -> None:
        self.spatial_dim = 8
        self.n_player = 6
        self.scalar_dim = 32

if __name__ == '__main__':
    weight, height = 28, 28
    spatial_dim = 8
    n_player = 6
    config = TestConfig()
    processor = PreProcessor(config)
    for i in range(100):
        heads_pos = np.random.randint(0, height, (n_player, 2))
        spatial = np.random.randint(0, spatial_dim, (weight, height))
        data = {
            'heads_pos': heads_pos,
            'spatial': spatial
        }
        data = processor.pre_process(data)
