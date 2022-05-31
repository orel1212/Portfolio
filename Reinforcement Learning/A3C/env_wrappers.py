import collections
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Grayscale
import numpy as np
import gym


IMAGE_SHAPE = (84, 84, 1)
REPEAT_TIMES = 4

class ActionRepeat(gym.Wrapper):
    def __init__(self, env=None, repeat=4, fire_first=False):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env_step(1)
        return obs


class FramePreprocessing(gym.ObservationWrapper):
    def __init__(self, dims, env=None):
        super(FramePreprocessing, self).__init__(env)
        self.shape = (dims[2], dims[0], dims[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape,
                                                dtype=np.float32)

    def observation(self, obs):
        composed = transforms.Compose([ToTensor(),Grayscale(),Resize(size=self.shape[1:])])
        resized_screen = composed(obs)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs


class FrameStacking(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(FrameStacking, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                env.observation_space.low.repeat(repeat, axis=0),
                env.observation_space.high.repeat(repeat, axis=0),
                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=IMAGE_SHAPE, repeat=REPEAT_TIMES):
    gym_env = gym.make(env_name)
    #print(gym_env.unwrapped.get_action_meanings())
    env = ActionRepeat(gym_env, repeat)
    env = FramePreprocessing(shape, env)
    env = FrameStacking(env, repeat)
    return env