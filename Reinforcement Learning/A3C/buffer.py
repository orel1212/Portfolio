class Buffer:
    def __init__(self):
        self.clear()

    def insert(self, state, action, reward, new_state, v, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.v_data.append(v)
        self.log_probs.append(log_prob)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.v_data = []
        self.log_probs = []

    def get_buffer(self):
        return self.states, self.actions, self.rewards, self.new_states,self.v_data, self.log_probs