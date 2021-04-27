import json
from threading import Thread

import gym
import zmq
from stable_baselines3.common.env_checker import check_env

import gym_space_engineers  # noqa: F401


class FakeServer(object):
    def __init__(self):
        super(FakeServer, self).__init__()
        self.stop_thread = False
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()
        self.n_legs = 6

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5560")

        while not self.stop_thread:
            #  Wait for next request from client
            request = json.loads(socket.recv())
            # print(f"Received request: {request}")

            # Default response
            response = dict(
                position=dict(x=0, y=0, z=0),
                right=dict(x=1, y=0, z=0),
                forward=dict(x=0, y=1, z=0),
                up=dict(x=0, y=0, z=1),
                endEffectorPositions=[dict(x=0, y=0, z=0) for _ in range(self.n_legs)],
            )

            if request["type"] == "Initial":
                response["id"] = 1
            elif request["type"] == "Stop":
                self.stop_thread = True
            elif request["type"] in ["Reset", "Command"]:
                pass
            else:
                raise NotImplementedError()

            #  Send reply back to client
            response_message = json.dumps(response)
            # print(f"Sent {response_message}")
            socket.send(response_message.encode("UTF-8"))


def test_gym_env():
    server = FakeServer()
    env = gym.make("SpaceEngineers-WalkingRobot-IK-v0", detach=True)

    # import ipdb; ipdb.set_trace()
    check_env(env, warn=True)

    # obs = env.reset()
    #
    # for _ in range(5):
    #     obs, reward, done, info = env.step(env.action_space.sample())

    env.close()
