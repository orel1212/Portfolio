
import torch
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from sac_networks import Actor, Critic, Value


class SACAgent:
    def __init__(self, device,alpha_actor, alpha_others, input_dims,num_actions, action_tanh_mult, tau= 0.005,
                  gamma=0.99, num_hidden_layers=1, hidden_size=256, buffer_sample_size=256, scaling=2, num_critics = 2):

        self.device = device
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.action_tanh_mult = action_tanh_mult
        self.tau = tau
        self.gamma = gamma
        self.buffer_sample_size = buffer_sample_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_critics = num_critics
        self.scaling = scaling


        self.buffer = ReplayBuffer(self.input_dims, self.num_actions)
        self.actor = Actor(self.device, alpha_actor, self.input_dims, self.num_actions, self.num_hidden_layers, self.hidden_size, self.action_tanh_mult)
        self.critics = []
        for i in range(self.num_critics):
            critic_ = Critic(self.device, alpha_others, self.input_dims, self.num_actions, self.num_hidden_layers, self.hidden_size, i+1)
            self.critics.append(critic_)

        self.value = Value(self.device, alpha_others, self.input_dims, self.num_hidden_layers, self.hidden_size,'value')
        self.target_value = Value(self.device, alpha_others, self.input_dims, self.num_hidden_layers, self.hidden_size,'target')

        self.detune_update_value_network_parameters(tau=1)

    def store_in_buffer(self, state, action, reward, new_state, done):
        self.buffer.store(state, action, reward, new_state, done)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        actions, _ = self.actor.normal_sampling(state, False)
        np_actions = actions.detach().cpu().numpy()
        return np_actions[0]

    def train(self):

        def get_critic_output(actor,critics,num_critics,state,reparameterize=False):
            actions, log_probs = actor.normal_sampling(state, reparameterize)
            log_probs = log_probs.view(-1)
            critic_output = critics[0].forward(state, actions)
            for i in range(1, num_critics):
                critic_output_ = self.critics[i].forward(state, actions)
                critic_output = torch.min(critic_output.clone(), critic_output_)
            critic_output = critic_output.view(-1)
            return critic_output, log_probs


        if not self.buffer.validate_min_buffer_size(self.buffer_sample_size):
            return

        state, action, reward, new_state, done = self.buffer.sample(self.buffer_sample_size)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        state_ = torch.tensor(new_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done).to(self.device)

        #updating the value networks, need without reparameterized
        critic_output, log_probs = get_critic_output(self.actor, self.critics, self.num_critics, state, reparameterize=False)

        target_value_pred = self.target_value(state_).view(-1)
        target_value_pred[done] = 0.0 # updated terminal states

        value_pred = self.value(state).view(-1)
        self.value.optimizer.zero_grad()
        value_target = critic_output - log_probs
        value_loss = 0.5 * (F.mse_loss(value_pred, value_target))
        value_loss.backward(retain_graph=True) #should retrain_Graph to not lose the coupling between losses of actor and value functions
        self.value.optimizer.step()

        #updating the actor network, need with reparameterized to add exploration.
        critic_output, log_probs = get_critic_output(self.actor, self.critics, self.num_critics, state,
                                                     reparameterize=True)

        self.actor.optimizer.zero_grad()
        actor_loss = log_probs - critic_output
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward(retain_graph=True) #should retrain_Graph to not lose the coupling between losses of actor and value functions
        self.actor.optimizer.step()

        #update critics, by using the buffered actions!
        q_val_hat = self.gamma * target_value_pred + self.scaling * reward

        critic_total_loss = 0
        for i in range(self.num_critics):
            self.critics[i].optimizer.zero_grad()
            critic_output_ = self.critics[i].forward(state, action).view(-1)
            critic_loss_ = 0.5 * F.mse_loss(critic_output_, q_val_hat)
            critic_total_loss += critic_loss_

        critic_total_loss.backward()
        for i in range(self.num_critics):
            self.critics[i].optimizer.step()

        self.detune_update_value_network_parameters()

    def detune_update_value_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        value_params = dict(self.value.named_parameters())
        target_value_params = dict(self.target_value.named_parameters())
        for name in value_params:
            value_params[name] = tau * value_params[name].clone() + (1 - tau) * target_value_params[name].clone()
        self.target_value.load_state_dict(value_params)

    def save(self):
        self.actor.save()
        self.value.save()
        self.target_value.save()
        for i in range(self.num_critics):
            self.critics[i].save()
        print("Saved models...")

    def load(self):
        self.actor.load()
        for i in range(self.num_critics):
            self.critics[i].load()
        self.value.load()
        self.target_value.load()
        print("Loaded models...")
