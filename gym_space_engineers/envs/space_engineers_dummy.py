import gym
import numpy as np
from gym import spaces


class SpaceEngineersDummyEnv(gym.Env):

    def __init__(self):
        self.bounds = 1000.
        self.position = np.array([0, 0])
        self.observation_space = spaces.Box(low=np.array([-self.bounds, -self.bounds]),
                                            high=np.array([self.bounds, self.bounds]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)

    def step(self, action):
        self.position = self.position + action
        self.position = np.clip(self.position, -self.bounds, self.bounds)

        # return observation, reward, isDone, info
        return self.position, 0, False, None

    def reset(self):
        self.position = (0, 0)

        return self.position

    def render(self, mode='human'):
        ...

    def close(self):
        ...
