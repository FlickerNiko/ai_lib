import numpy as np


# GAE replay buffer

class ReplayBuffer:
    def __init__(self, size, proto) -> None:
        # size: buffer_size, gae_length
        # proto: dict {key: (shape, dtype)}
        self.proto = proto
        self.size = size
        self.buffer_size = size[0]
        self.gae_length = size[1]
        self.st_idx = 0
        self.n_sample = 0
        self.init_buffer(size, proto)

    def init_buffer(self, size, proto):
        self._buffer = {}
        for item_name in proto:
            shape, dtype = proto[item_name]
            full_shape = size + shape
            self._buffer[item_name] = np.zeros(full_shape, dtype)

    def sample(self, n_batch):

        indices = np.random.randint((self.n_sample, self.gae_length), size = (n_batch,2))
        # indices = np.random.randint(self.n_sample, (n_batch,))
        indices[:,0] += self.st_idx
        indices[:,0] %= self.buffer_size

        batch = {}
        for item_name in self._buffer:
            batch[item_name] = self._buffer[item_name][indices[:, 0], indices[:, 1]]

        return batch

    def push(self, data):

        insert_idx =  (self.st_idx + self.n_sample) % self.buffer_size
        for item_name in data:
            self._buffer[item_name][insert_idx] = data[item_name]

        if self.n_sample < self.buffer_size:
            self.n_sample += 1
        else:
            self.st_idx += 1
            self.st_idx %= self.buffer_size




