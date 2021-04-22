import os

import gym
import numpy as np
from gym import spaces
from gym_space_engineers.agent import AgentController, MoveArgs
import random
import zmq
import time
import json
from scoop import futures
import signal

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5560")


class WalkingRobotIKEnv(gym.Env):
    def __init__(self, detach=True):
        self.detach = detach
        self.id = None
        self.observation_space = None
        self.number_of_legs = 6
        self.action_space = spaces.Box(
            low=np.stack([[-np.inf, -np.inf, -np.inf, 0] for _ in range(self.number_of_legs)]),
            high=np.stack([[np.inf, np.inf, np.inf, 1] for _ in range(self.number_of_legs)]),
            dtype=np.float32,
            shape=(self.number_of_legs, 4)
        )

    def step(self, action):
        if self.id is None:
            raise Exception("Please call reset() before step()")

        commands = {}
        leg_ids = ["l1", "l2", "l3", "r1", "r2", "r3"]

        for i, leg_id in zip(range(self.number_of_legs), leg_ids):
            values = [(action[i][j].item()) for j in range(4)]
            commands[leg_id] = {
                "position": {
                    "x": values[0],
                    "y": values[1],
                    "z": values[2],
                },
                "speed": values[3] * 100,
            }

        request = {
            "id": self.id,
            "type": "Command",
            "commands": commands,
        }

        response = self._send_request(request)
        position = self._get_np_array_from_vector(response["position"])
        up = self._get_np_array_from_vector(response["up"])
        forward = self._get_np_array_from_vector(response["forward"])

        info = {
            "up": up,
            "forward": forward,
        }

        return position, 0, False, info

    def reset(self):
        if self.id is None:
            self._send_initial_request()
        else:
            raise NotImplementedError()

    @staticmethod
    def _get_np_array_from_vector(vector):
        return np.array([vector["x"], vector["y"], vector["z"]])

    @staticmethod
    def _send_request(request):
        request_message = json.dumps(request)
        socket.send(request_message.encode("UTF-8"))
        response = json.loads(socket.recv())

        return response

    def _send_initial_request(self):
        request = {
            "type": "Initial",
            "detach": self.detach,
        }
        response = self._send_request(request)
        self.id = response["id"]

    def render(self, mode='human'):
        ...

    def close(self):
        if self.id is not None:
            request = {
                "type": "Stop",
                "id": self.id,
            }
            self._send_request(request)


# Test the environment by doing 1000 random steps in the game
if __name__ == "__main__":
    import gym
    import time

    for _ in range(1):
        env = gym.make('SpaceEngineers-WalkingRobot-IK-v0', detach=False)
        env.reset()

        # All legs from low to high
        for y in np.linspace(-10, 10, 30):
            env.render()

            left_leg_position = [-5, y, -0.5, 0.1]
            action = np.stack(left_leg_position for i in range(6))
            action[3:, 0] *= -1

            observation, _, _, _ = env.step(action)
            print(observation)

            time.sleep(0.5)

        time.sleep(0.3)

        # All legs from back to front
        for z in np.linspace(5, -10, 30):
            env.render()

            left_leg_position = [-5, -2, z, 0.1]
            action = np.stack(left_leg_position for i in range(6))
            action[3:, 0] *= -1

            observation, _, _, _ = env.step(action)
            print(observation)

            time.sleep(0.5)

        time.sleep(0.3)

        env.close()
