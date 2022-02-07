import torch
import torch.nn.functional as F
import numpy as np


class PreProcessor:
    def __init__(self, config) -> None:
        self.config = config
        self.spatial_dim = config.spatial_dim
    
    def preprocess(self, data):

        spatial = data['spatial']
        heads_pos = data['heads_pos']

        # extend up, down, left, right
        width = spatial.shape[2]
        height = spatial.shape[3]

        spatial = torch.concat([spatial]*3, 2)
        spatial = torch.concat([spatial]*3, 3)
        spatial = F.one_hot(spatial, self.spatial_dim)
        
        
        idv_spatials = []
        for i in range(len(heads_pos)):
            head_pos = heads_pos[i]
            x, y = head_pos
            idv_spatial = spatial[:, :, y+1:y+2*height, x+1:x+2*width]
            idv_spatials.append(idv_spatial)            
        
        # n_batch, n_player, n_channel, height, width
        idv_spatials = torch.stack(idv_spatials, 1)
        # data['spatial'] = 
        return idv_spatial
        