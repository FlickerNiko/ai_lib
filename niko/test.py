import numpy as np
import torch
from .replay_buffer import ReplayBuffer
from .rollout import Rollout
from .ppo import PPO


class TestEnv:

    def __init__(self) -> None:
        self.state_proto = {
            'main': ((12,4), np.float32),
            'scalar': ((32,), np.float32)
        }
        self.state_shapes = self.state_proto

    def reset(self):
        state = self.generate_state()
        return state

    def generate_state(self):
        state = {}
        for key in self.state_proto:
            shape, dtype = self.state_proto[key]
            state[key] = np.asarray(np.random.random(size = shape), dtype = dtype)
        return state

    def step(self, action):
        state = self.generate_state()
        # state = {key : np.random.random(size = self.state_proto[key]) for key in self.state_proto}
        reward = np.random.random()
        done = False
        info = None

        return state, reward, done, info


class TestAgent:

    def __init__(self) -> None:
        self.action_shapes = None
        


    def predict(self, state):
        n_batch = list(state.values())[0].shape[0]        
        logits = torch.rand((n_batch, 4))
        value = torch.rand((n_batch,))
        return logits, value
    

class TestConfig:
    def __init__(self) -> None:
        self.gamma = 0.99
        self.Lambda = 0.95
        self.coef_value = 1.0
        self.coef_entropy = 0.01
        self.epsilon = 0.2



if __name__ == '__main__':
    from .test import *
    
    buffer_size = 2048
    gae_length = 32
    n_episode  = 256
    n_step = 256
    n_batch = 32

    agent = TestAgent()
    env = TestEnv()

    buffer_proto = {
        'action': ((), np.int64),
        'logit': ((), np.float32),
        'reward': ((), np.float32),
        'value': ((), np.float32),
        'done' : ((), np.bool8),
        'gae': ((), np.float32),
    }

    state_proto = env.state_shapes

    for state_key in state_proto:
        buffer_proto[state_key] = state_proto[state_key]

    replay_buffer = ReplayBuffer((buffer_size,gae_length), buffer_proto)
    config  = TestConfig()
    rollout = Rollout(env, agent,replay_buffer, config)
    rollout.rollout(n_episode, n_step)
    

    ppo = PPO(agent, None, state_proto.keys(), config)

    def to_torch(data):
        for key in data:
            data[key] = torch.from_numpy(data[key])
        return data

    losses = []

    for i in range(128):
        samples = replay_buffer.sample(n_batch)
        samples = to_torch(samples)
        loss = ppo.train(samples)
        losses.append(loss)
    
    pass