from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .replay_buffer import ReplayBuffer
from .rollout import Rollout
from .ppo import PPO
from .learner import Learner

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

class TestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 32
        self.fc1 = nn.Linear(80, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.fc_a = nn.Linear(hidden_dim, 4)

    def forward(self, main, scalar):
        main = torch.reshape(main, (-1, 48))
        input  = torch.concat((main, scalar), -1)
        h = self.fc1(input)
        v = self.fc_v(h)
        a = F.softmax(self.fc_a(h), -1) 

        return a, v

class TestAgent:

    def __init__(self) -> None:
        self.action_shapes = None
        self.model = TestModel()

        self.params = self.model.parameters()
        

    def predict(self, state):
        return self.model.forward(state['main'], state['scalar'])

        # n_batch = list(state.values())[0].shape[0]
        # logits = torch.rand((n_batch, 4))
        # value = torch.rand((n_batch,))
        # return logits, value
    

class TestConfig:
    def __init__(self) -> None:
        self.gamma = 0.99
        self.Lambda = 0.95
        self.coef_value = 1.0
        self.coef_entropy = 0.01
        self.epsilon = 0.2
        self.lr = 1e-4
        self.eps = 1e-30



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
    

    ppo = PPO(agent, state_proto.keys(), config)

    def to_torch(data):
        for key in data:
            data[key] = torch.from_numpy(data[key])
        return data


    learner = Learner(replay_buffer, agent, ppo, config)
    learner.train(n_batch, 128)

    # losses = []

    # for i in range(128):
    #     samples = replay_buffer.sample(n_batch)
    #     samples = to_torch(samples)
    #     loss = ppo.train(samples)
    #     losses.append(loss)
    
    # pass