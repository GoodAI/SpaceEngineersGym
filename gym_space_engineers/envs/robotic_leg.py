import gym
import numpy as np
from gym import spaces
from gym_space_engineers.agent import AgentController, MoveArgs
import random
import zmq
import time
import json
from scoop import futures

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5556")


class RoboticLegEnv(gym.Env):
    def __init__(self):
        self.agent = AgentController()

        self.rotorsCount = 3
        self.observation_bounds = 10
        self.observation_space = spaces.Box(
            low=np.array([-self.observation_bounds, -self.observation_bounds, -self.observation_bounds]),
            high=np.array([self.observation_bounds, self.observation_bounds, self.observation_bounds]),
            dtype=np.float32)

        self.action_bounds = 1
        self.action_space = spaces.Box(low=np.repeat(-self.action_bounds, self.rotorsCount),
                                       high=np.repeat(self.action_bounds, self.rotorsCount), dtype=np.float32)

    def step(self, action):
        configurations = [(action[i].item()) for i in range(self.rotorsCount)]
        # print(configurations)
        request = json.dumps(configurations)
        socket.send(request.encode("UTF-8"))
        response = json.loads(socket.recv())
        position = response["endEffectorPositionLocal"]

        info = {
            "was_successful": response["status"] == 3,
            "configuration": action,
        }

        return np.array([position["x"], position["y"], position["z"]]), 0, True, info

    def reset(self):
        return np.zeros(3)

    def render(self, mode='human'):
        ...


# Test the environment by doing 1000 random steps in the game
if __name__ == "__main__":
    import gym
    import time

    env = gym.make('SpaceEngineers-RoboticLeg-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        observation, _, _, _ = env.step(env.action_space.sample())
        print(observation)
        time.sleep(5)

    env.close()
