import torch
import torch.multiprocessing as mp
from ac_agent import Agent
from worker import create_worker

LEARNING_RATE = 1e-4
n_hidden = 256

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=LEARNING_RATE, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ParallelEnvironment:
    def __init__(self, env_id, input_shape, n_actions, num_threads):

        names = [str(i) for i in range(num_threads)]
        global_actor_critic = Agent(input_shape, n_actions,n_hidden)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters(), lr=LEARNING_RATE)

        self.ps = [mp.Process(target=create_worker,
                              args=(name, input_shape, n_actions,
                                    global_actor_critic, global_optim, env_id,
                                    num_threads))
                   for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]