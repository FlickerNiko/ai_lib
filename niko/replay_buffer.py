import numpy as np


# GAE replay buffer

class ReplayBuffer:
    def __init__(self, proto, config) -> None:
        # size: buffer_size, gae_length
        # proto: dict {key: (shape, dtype)}
        self.proto = proto
        self.buffer_size = config.buffer_size
        self.gae_length = config.gae_length
        self.st_idx = 0
        self.n_sample = 0
        self.init_buffer((self.buffer_size, self.gae_length), proto)

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

    def push_n(self, data, n):
        for i in range(n):
            sub_data = {item_name: data[item_name][i] for item_name in data}
            self.push(sub_data)

    def push_n_2(self, data, n):
        for i in range(n):
            insert_idx =  (self.st_idx + self.n_sample) % self.buffer_size
            for item_name in data:
                self._buffer[item_name][insert_idx] = data[item_name][i]

            if self.n_sample < self.buffer_size:
                self.n_sample += 1
            else:
                self.st_idx += 1
                self.st_idx %= self.buffer_size
    