

from copy import deepcopy
import torch
from torch.nn import functional

class PPO:

    def __init__(self, agent, optim, state_keys, config) -> None:
        self.agent = agent
        self.agent_target = deepcopy(agent)
        self._config = config
        self.state_keys = state_keys
        self.optim = optim
        pass


    def get_state(self, data):
        state = {}
        for state_key in  self.state_keys:
            state[state_key] = data[state_key]
        return state

    def train(self, data):

        # data: dict of torch tensor
        # process loss
        coef_value = self._config.coef_value
        coef_entropy = self._config.coef_entropy
        epsilon = self._config.epsilon

        state = self.get_state(data)
        logits, value = self.agent.predict(state)

        gae = data['gae']


        loss_value = functional.mse_loss(value, gae)        

        action = data['action']        
        
        logit = logits.gather(-1, action.unsqueeze(-1))[:,0]
        
        logit_old = data['logit']
        ratio = logit/logit_old
        ratio_clip = torch.clip(ratio, 1 - epsilon, 1 + epsilon)
        loss_action = torch.min(ratio * gae, ratio_clip * gae)


        loss_entropy = - torch.mean(torch.sum(logits * torch.log(logits), dim = -1))

        loss = loss_action + coef_value * loss_value + coef_entropy * loss_entropy

        return loss
        
        # apply gradient        
        # self.optim.zero_grad()
        # loss.backward()
        # self.optim.step()



if __name__ == '__main__':
    from .test import *
    pass