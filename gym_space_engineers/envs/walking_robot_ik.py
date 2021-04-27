import gym
import numpy as np
from gym import spaces
import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5560")


class WalkingRobotIKEnv(gym.Env):
    def __init__(self, detach=True):
        self.detach = detach
        self.id = None
        self.observation_space = None

        # For now, this is hardcoded for the 6-legged robot
        self.number_of_legs = 6

        # For now, we expect that the legs can move 10 meters in each direction
        # We use this value to map the [-1,1] interval to the actual reachable space
        self.action_space_size = 10

        # [X, Y, Z, Speed] for each of the 6 legs
        # (X, Y, Z) is a position relative to the shoulder joint of each leg
        # This position will be given to the inverse kinematics model
        self.action_space = spaces.Box(
            low=np.stack([[-1, -1, -1, -1] for _ in range(self.number_of_legs)]).flatten(),
            high=np.stack([[1, 1, 1, 1] for _ in range(self.number_of_legs)]).flatten(),
            dtype=np.float32,
        )

    def step(self, action):
        if self.id is None:
            raise Exception("Please call reset() before step()")

        commands = {}
        leg_ids = ["l1", "l2", "l3", "r1", "r2", "r3"]

        for i, leg_id in zip(range(self.number_of_legs), leg_ids):
            values = [(action[4 * i + j].item()) for j in range(4)]
            commands[leg_id] = {
                "position": {
                    "x": values[0] * self.action_space_size,
                    "y": values[1] * self.action_space_size,
                    "z": values[2] * self.action_space_size,
                },
                "speed": (values[3] + 1) * 50,
            }

        request = {
            "id": self.id,
            "type": "Command",
            "commands": commands,
        }

        response = self._send_request(request)

        return self._get_observation(response)

    def reset(self):
        if self.id is None:
            response = self._send_initial_request()
        else:
            request = {
                "id": self.id,
                "type": "Reset",
            }
            response = self._send_request(request)

        return self._get_observation(response)

    def _get_observation(self, response):
        position = self._get_array_from_vector(response["position"])
        up = self._get_array_from_vector(response["up"])
        forward = self._get_array_from_vector(response["forward"])
        right = self._get_array_from_vector(response["right"])
        end_effector_positions = np.stack([self._get_array_from_vector(pos) for pos in response["endEffectorPositions"]])

        info = {
            "up": up,
            "forward": forward,
            "right": right,
            "end_effector_positions": end_effector_positions,
        }

        return position, 0, False, info

    @staticmethod
    def _get_array_from_vector(vector):
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

        return response

    def render(self, mode='human'):
        ...

    def close(self):
        if self.id is not None:
            request = {
                "type": "Stop",
                "id": self.id,
            }
            self._send_request(request)


if __name__ == "__main__":
    import gym
    import time
    # noinspection PyUnresolvedReferences
    import gym_space_engineers

    def postprocess_action(action):
        # Multiply x by -1 for the legs on the right side
        action[3:, 0] *= -1

        # Divide x,y,z by 10 to fit them into [-1,1]
        action[:, 0:3] /= 10

    for _ in range(1):
        env = gym.make('SpaceEngineers-WalkingRobot-IK-v0', detach=False)

        observation, _, _, _ = env.reset()
        print(observation)

        # All legs from low to high
        for y in np.linspace(-10, 10, 30):
            env.render()

            left_leg_position = [-5, y, -0.5, 0.1]
            action = np.stack([left_leg_position for i in range(6)])
            postprocess_action(action)

            observation, _, _, _ = env.step(action.flatten())
            print(observation)

            time.sleep(0.5)

        time.sleep(0.3)

        observation, _, _, _ = env.reset()
        print(observation)

        # All legs from back to front
        for z in np.linspace(5, -10, 30):
            env.render()

            left_leg_position = [-5, -2, z, 0.1]
            action = np.stack([left_leg_position for i in range(6)])
            postprocess_action(action)

            observation, _, _, _ = env.step(action.flatten())
            print(observation)

            time.sleep(0.5)

        time.sleep(0.3)

        env.close()
