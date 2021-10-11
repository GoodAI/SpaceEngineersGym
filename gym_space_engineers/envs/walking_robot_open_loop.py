import json
from typing import Any, Dict

import gym
import zmq


class WalkingRobotOpenLoopEnv(gym.Env):
    def __init__(self, mech_name: str, objective: str):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5572")
        self.objective = objective

        response = self._send_request(
            {
                "type": "Initial",
                "mechName": mech_name,
            }
        )

        self.id = response["id"]

    def step(self, action: dict):

        response = self._send_request(
            {
                "id": self.id,
                "type": "Command",
                "objective": self.objective,
                "hyperParams": action,
            }
        )

        return response, 0, True, None

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

    while True:
        env = WalkingRobotOpenLoopEnv(mech_name="v2", objective="")
        env.reset()

        observation, _, _, _ = env.step(
            {
                "duration": 50,
                "x": 5.250909035640978,
                "yRange": 6.6994406917744693,
                "yMin": -8.947822208617133,
                "zMin": -1.2499644968318842,
                "zRange": 2.005817017551913,
                "numberOfCycles": 20,
            }
        )

        print(observation)

        env.close()
