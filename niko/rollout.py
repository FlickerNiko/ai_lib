
import numpy as np

class Rollout:
    def __init__(self, env, agent, replay_buffer, config) -> None:
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer        
        self.gae_length = self.replay_buffer.gae_length
        self.config = config

    def compute_gae(self, gae_buffer, value_T, n_step):

        rewards = gae_buffer['reward']
        values = gae_buffer['value']
        gae = gae_buffer['gae']
                
        gamma = self.config.gamma
        Lambda = self.config.Lambda
        # todo: value_last for gae
        i = n_step - 1
        gae[i] = rewards[i] + gamma * value_T - values[i]
        i -= 1
        while i >= 0:
            advantage = rewards[i] + gamma * values[i+1] - values[i]
            gae[i] = advantage + (gamma * Lambda) * gae[i+1]
            i -= 1
    
    
    def create_gae_buffer(self):
        buffer = {}
        gae_shape = (self.gae_length)
        buffer['gae'] = np.zeros(shape = gae_shape)
        buffer['logit'] = np.zeros(shape = gae_shape)
        buffer['done'] = np.zeros(shape = gae_shape)
        buffer['reward'] = np.zeros(shape = gae_shape)
        buffer['value'] = np.zeros(shape = gae_shape)
        buffer['action'] = np.zeros(shape = gae_shape)  
        
        for state_name in self.env.state_proto:
            state_shape = self.env.state_proto[state_name]
            buffer[state_name] = np.zeros(shape = gae_shape + state_shape)            

    def reset_gae_buffer(self, buffer):

        for item in buffer.values():
            item.fill(0)      
            # item[:] = 0
            pass
            

    def rollout(self, n_episode, n_step = None):
                
        
        gae_buffer = self.create_gae_buffer()


        for i in range(n_episode):

            gae_step = 0
            state = self.env.reset()
            done = False
            
            

            while not done:
                
                logits, value = self.agent.predict(state)
                action = logits.select()  # action: int index
                state, done, reward, info = self.env.step(action)
                for state_name in state:
                    gae_buffer[state_name][gae_step] = state[state_name]
                
                gae_buffer['action'][gae_step] = action
                gae_buffer['logits'][gae_step] = logits[action]
                gae_buffer['reward'][gae_step] = reward
                gae_buffer['value'][gae_step] = value
                gae_buffer['done'][gae_step] = done
                gae_buffer['gae'][gae_step] =  value
                            
                gae_step += 1

                if gae_step >= self.gae_length or done:
                    # gae submit
                    _, value_T = self.agent.predict(state)
                    self.compute_gae(gae_buffer, value_T, gae_step)
                    self.replay_buffer.push(gae_buffer)
                    gae_step = 0
                    self.reset_gae_buffer(gae_buffer)
                    
            