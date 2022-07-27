import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Value(nn.Module):
    def __init__(self, device, alpha, input_dims, num_hidden_layers, hidden_size, mode='value',
                 save_dir='./saved_model'):
        super(Value, self).__init__()
        self.device = device
        self.input_dims = input_dims
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.model_file = os.path.join(save_dir, mode + '.pt')
        if self.num_hidden_layers == 0:
            self.num_hidden_layers += 1  # append at least one hidden layer
        self.layers = nn.ModuleList([nn.Linear(self.input_dims, self.hidden_size)])
        for l in range(1, self.num_hidden_layers):  # step 0 cuz we already added one layer
            self.layers.append([nn.Linear(self.hidden_size, self.hidden_size)])
        self.layers.append(nn.Linear(self.hidden_size, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(self.device)

    def forward(self, state):
        data = state
        for i, l in enumerate(self.layers):
            data = l(data)
            if i != len(self.layers) - 1:
                data = F.relu(data)
        return data

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))


class Actor(nn.Module):
    def __init__(self, device, alpha, input_dims, num_actions, num_hidden_layers, hidden_size, action_tanh_mult,
                 save_dir='./saved_model'):
        super(Actor, self).__init__()
        self.device = device
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.action_tanh_mult = action_tanh_mult
        self.model_file = os.path.join(save_dir, "actor.pt")
        if self.num_hidden_layers == 0:
            self.num_hidden_layers += 1  # append at least one hidden layer
        self.h_layers = nn.ModuleList([nn.Linear(self.input_dims, self.hidden_size)])
        for l in range(1, self.num_hidden_layers):  # step 0 cuz we already added one layer
            self.h_layers.append([nn.Linear(self.hidden_size, self.hidden_size)])
        self.mu_layer = nn.Linear(self.hidden_size, self.num_actions)
        self.sigma_layer = nn.Linear(self.hidden_size, self.num_actions)
        self.noise = 2e-6
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(self.device)

    def forward(self, state):
        probs = state
        for h_l in self.h_layers:
            probs = h_l(probs)
            probs = F.relu(probs)
        mu = self.mu_layer(probs)
        sigma = self.sigma_layer(probs)
        sigma = torch.clamp(sigma, min=self.noise, max=1)
        return mu, sigma

    def normal_sampling(self, state, to_reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = torch.distributions.Normal(mu, sigma)

        if to_reparameterize:  # to reparameterizes the policy, and get a more exploration based actions
            actions = probabilities.rsample()
        else:  # just sample from the dist
            actions = probabilities.sample()

        action = torch.tanh(actions) * torch.tensor(self.action_tanh_mult).to(self.device)
        log_probs = probabilities.log_prob(actions) - torch.log(1 - action.pow(2) + self.noise)
        log_probs = log_probs.sum(1, keepdim=True)  # make it into scalar
        return action, log_probs

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))


class Critic(nn.Module):
    def __init__(self, device, alpha, input_dims, num_actions, num_hidden_layers, hidden_size, index,
                 save_dir='./saved_model'):
        super(Critic, self).__init__()
        self.device = device
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.index = index
        self.model_file = os.path.join(save_dir, 'critic_' + str(self.index) + '.pt')
        if self.num_hidden_layers == 0:
            self.num_hidden_layers += 1  # append at least one hidden layer
        self.layers = nn.ModuleList([nn.Linear(self.input_dims + self.num_actions, self.hidden_size)])
        for l in range(1, self.num_hidden_layers):  # step 0 cuz we already added one layer
            self.layers.append([nn.Linear(self.hidden_size, self.hidden_size)])
        self.layers.append(nn.Linear(self.hidden_size, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(self.device)

    def forward(self, state, action):
        data = torch.cat([state, action], dim=1)
        for i, l in enumerate(self.layers):
            data = l(data)
            if i != len(self.layers) - 1:
                data = F.relu(data)
        return data

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
