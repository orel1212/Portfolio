import os
import torch.multiprocessing as mp
from utils import ParallelEnvironment

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'PongNoFrameskip-v4'
    n_threads = 10
    n_actions = 6
    input_shape = [4, 42, 42]
    env = ParallelEnvironment(env_id, input_shape, n_actions, n_threads)