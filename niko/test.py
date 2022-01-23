

class TestEnv:

    def __init__(self) -> None:
        self.state_proto = {
            'main': (12,4),
            'scalar': (32,)
        }
        self.state_shapes = self.state_proto

    def reset(self):
        pass

    def step(self, action):

        state = {key : np.random.random(size = self.state_proto[key]) for key in self.state_proto}
        reward = np.random.random()
        done = False
        info = None

        return state, reward, done, info


class TestAgent:

    def __init__(self) -> None:
        self.action_shapes = None

    def predict(self, state):
        logits = np.random.random((4,))
        value = np.random.random()
        return logits, value

class TestConfig:
    def __init__(self) -> None:
        self.gamma = 0.99
        self.Lambda = 0.95
