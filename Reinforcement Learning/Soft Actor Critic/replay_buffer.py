
import numpy as np

class ReplayBuffer:
    def __init__(self,  input_shape, num_actions, max_size = 1000000):
        self.buffer_state_shape = input_shape
        self.buffer_action_shape = num_actions
        self.buffer_size = max_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_size, self.buffer_state_shape))
        self.action_buffer = np.zeros((self.buffer_size, self.buffer_action_shape))
        self.reward_buffer = np.zeros(self.buffer_size)
        self.new_state_buffer = np.zeros((self.buffer_size, self.buffer_state_shape))
        self.done_buffer = np.zeros(self.buffer_size, dtype=bool)

    def validate_min_buffer_size(self, req_sample_size):
        if self.buffer_counter < req_sample_size:
            return False
        return True

    def store(self, state, action, reward, new_state, done):
        index_to_save = self.buffer_counter % self.buffer_size
        self.state_buffer[index_to_save] = state
        self.action_buffer[index_to_save] = action
        self.reward_buffer[index_to_save] = reward
        self.new_state_buffer[index_to_save] = new_state
        self.done_buffer[index_to_save] = done

        self.buffer_counter += 1

    def sample(self, req_sample_size):
        curr_size = min(self.buffer_size, self.buffer_counter)
        batch = np.random.choice(curr_size, req_sample_size)

        states = self.state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        new_states = self.new_state_buffer[batch]
        dones = self.done_buffer[batch]

        return states, actions, rewards, new_states, dones

