import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class SEDummyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.position = np.array([0, 0])
        self.observation_space = spaces.Box(low=np.array([-500., -500.]), high=np.array([500., 500.]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)

    def step(self, action):
        self.position = self.position + action
        self.position = np.clip(self.position, -500, 500)

        # return observation, reward, isDone, info
        return self.position, 0, False, None

    def reset(self):
        self.position = (0, 0)

        return self.position

    def render(self, mode='human'):
        ...

    def close(self):
        ...
