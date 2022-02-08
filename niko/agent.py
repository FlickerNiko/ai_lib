from .model import Model
from .pre_processor import PreProcessor

class Agent:
    def __init__(self, config) -> None:
        self.config = config
        self.model = Model(config)
        self.processor = PreProcessor(config)
    
    def inference(self, data):
        data = self.processor.pre_process(data)
        action_prob, value = self.model.forward(data)
        return action_prob, value