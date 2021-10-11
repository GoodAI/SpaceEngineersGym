import json
from typing import Any, Dict, Tuple

import gym
import numpy as np
import zmq
from gym import spaces


class RoboticLegEnvV1(gym.Env):
    def __init__(self, robotic_leg_name: str):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5570")

        response = self._send_request(
            {
                "type": "Initial",
                "roboticLegName": robotic_leg_name,
            }
        )

        self.id = response["id"]
        self.rotors_count = response["rotorsCount"]
        self.observation_bounds = 10
        self.observation_space = spaces.Box(
            low=np.array([-self.observation_bounds, -self.observation_bounds, -self.observation_bounds]),
            high=np.array([self.observation_bounds, self.observation_bounds, self.observation_bounds]),
            dtype=np.float32,
        )

        self.action_bounds = 1
        self.action_space = spaces.Box(
            low=np.repeat(-self.action_bounds, self.rotors_count),
            high=np.repeat(self.action_bounds, self.rotors_count),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        configurations = [(action[i].item()) for i in range(self.rotors_count)]

        response = self._send_request(
            {
                "id": self.id,
                "type": "Command",
                "configurations": configurations,
            }
        )

        position = response["endEffectorPosition"]
        position_array = np.array([position["x"], position["y"], position["z"]])

        orientation = response["endEffectorOrientation"]
        orientation_array = np.array([orientation["x"], orientation["y"], orientation["z"]])

        info = {
            "was_successful": response["wasSuccessful"],
            "orientation": orientation_array,
        }

        return position_array, 0, True, info

    def reset(self):
        return None

    def render(self, mode="human"):
        ...

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_message = json.dumps(request)
        self.socket.send(request_message.encode("UTF-8"))
        response = json.loads(self.socket.recv())
        return response

    def close(self):
        request = {
            "type": "Stop",
            "id": self.id,
        }
        try:
            self._send_request(request)
            self.socket.close()
        except zmq.error.ZMQError:
            pass


# Test the environment by doing 20 random steps in the game
if __name__ == "__main__":
    import gym

    import gym_space_engineers  # noqa: F401

    env = gym.make("SpaceEngineers-RoboticLeg-v1", robotic_leg_name="v5")
    env.reset()
    for _ in range(20):
        env.render()
        observation, _, _, _ = env.step(env.action_space.sample())
        print(observation)

    env.close()
