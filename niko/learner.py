from .replay_buffer import ReplayBuffer
import torch

class Learner:
    def __init__(self, replay_buffer: ReplayBuffer, agent, algorithm, config) -> None:
        self.replay_buffer = replay_buffer
        self.agent = agent
        self.algorithm = algorithm        
        self.optim = None
        self.config = config
        


    def _to_torch(self, data):
        for key in data:
            data[key] = torch.from_numpy(data[key])
        return data


    def train(self, n_batch, n_step):
        for i in range(n_step):
            data = self.replay_buffer.sample(n_batch)
            data = self._to_torch(data)
            loss = self.algorithm.train(data)
            print(f"step {i}, loss {loss}")
                        