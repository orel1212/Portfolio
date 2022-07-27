import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Agent(nn.Module):
    def __init__(self, input_dims, n_actions, n_hidden, gamma=0.99, tau=1.0, cnn_dim=32, num_kernels=3):
        super(Agent, self).__init__()

        self.conv_layers = nn.ModuleList([nn.Conv2d(input_dims[0], cnn_dim, num_kernels, stride=2, padding=1), \
                                          nn.Conv2d(cnn_dim, cnn_dim, num_kernels, stride=2, padding=1), \
                                          nn.Conv2d(cnn_dim, cnn_dim, num_kernels, stride=2, padding=1), \
                                          nn.Conv2d(cnn_dim, cnn_dim, num_kernels, stride=2, padding=1)])
        conv_output_shape = self.get_conv_output_shape(input_dims)

        self.gru = nn.GRUCell(conv_output_shape, n_hidden)
        self.pi = nn.Linear(n_hidden, n_actions)
        self.v = nn.Linear(n_hidden, 1)

        self.gamma = gamma
        self.tau = tau

    def get_conv_output_shape(self, input_dims):
        data = torch.zeros(1, *input_dims)
        for l in self.conv_layers:
            data = l(data)
        return int(np.prod(data.size()))

    def forward(self, state, hx):
        data = state
        for l in self.conv_layers:
            data = l(data)

        conv_output = data.view((data.size()[0], -1))

        hx = self.gru(conv_output, (hx))

        pi = self.pi(hx)
        v = self.v(hx)

        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        action_sampling = dist.sample()
        log_prob = dist.log_prob(action_sampling)
        action = action_sampling.numpy()[0]

        return action, v, log_prob, hx

    def _R_eval(self, rewards, done, v_data):
        concat_v_data = torch.cat(v_data).squeeze()

        if len(concat_v_data.size()) == 1:  # batch of states
            v = concat_v_data[-1]
        elif len(concat_v_data.size()) == 0:  # single state
            v = concat_v_data
        R = v * (1 - int(done))

        R_returns = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            R_returns.append(R)
        R_returns.reverse()
        R_returns = torch.tensor(R_returns, dtype=torch.float).reshape(concat_v_data.size())
        return R_returns

    def loss_calc(self, new_state, hx, done,
                  rewards, v_data, log_probs, intrinsic_reward=None, entropy_temperature=0.01):

        if intrinsic_reward is not None:
            rewards += intrinsic_reward.detach().numpy()

        returns = self._R_eval(rewards, done, v_data)

        new_v = self.forward(torch.tensor(np.array([new_state]), dtype=torch.float), hx)[1] if not done \
            else torch.zeros(1, 1)
        v_data.append(new_v.detach())
        values = torch.cat(v_data).squeeze()
        concat_log_probs = torch.cat(log_probs)
        rewards = torch.tensor(rewards)

        deltas = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(deltas)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps - t):
                curr_gae_val = (self.gamma * self.tau) ** k * deltas[t + k]
                gae[t] += curr_gae_val
        gae = torch.tensor(gae, dtype=torch.float)

        actor_loss = -(concat_log_probs * gae).sum()
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = (-concat_log_probs * torch.exp(concat_log_probs)).sum()

        total_loss = actor_loss + critic_loss - entropy_temperature * entropy_loss
        return total_loss
