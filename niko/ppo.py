

from copy import deepcopy


class PPO:

    def __init__(self, agent, state_keys, config) -> None:
        self.agent = agent
        self.agent_target = deepcopy(agent)
        self._config = config
        self.state_keys = state_keys
        pass


    def get_state(self, gae_buffer):
        state = {}
        for state_key in  self.state_keys:
            state[state_key] = gae_buffer[state_key]
        return state

    def train(self, data):
        # process loss
        coef_value = self._config.coef_value
        coef_entropy = self._config.coef_entropy
        epsilon = self._config.epsilon

        state = self.get_state(data)
        logits, value = self.get_state()

        gae = data['gae']
        def mse():
            pass
        loss_value = mse(gae, value)

        action = data['action']


        logit = logits[action]
        logit_old = data['logit']
        ratio = logit/logit_old        
        ratio_clip = clip(ratio, 1 - epsilon, 1 + epsilon)        
        loss_action = min(ratio * gae, ratio_clip * gae)
        
        loss_entropy = None
        loss = loss_action + coef_value * loss_action + coef_entropy * loss_entropy        
        # apply gradient        
        pass




if __name__ == '__main__':
    from .test import *
    pass