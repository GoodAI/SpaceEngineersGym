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
        self.n_legs = 6
        self.step = 0

    def start(self):
        self.thread.start()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5560")

        while not self.stop_thread:
            #  Wait for next request from client
            request = json.loads(socket.recv())
            # print(f"Received request: {request}")
            self.step += 1

            # Default response
            # Rotated 180deg
            response = dict(
                position=dict(x=1, y=1, z=-self.step),
                right=dict(x=1, y=0, z=0),
                forward=dict(x=0, y=0, z=-1),
                up=dict(x=0, y=1, z=0),
                endEffectorPositions=[dict(x=0, y=0, z=0) for _ in range(self.n_legs)],
            )
            # Aligned with z axis
            # response = dict(
            #     position=dict(x=1, y=1, z=self.step),
            #     right=dict(x=-1, y=0, z=0),
            #     forward=dict(x=0, y=0, z=1),
            #     up=dict(x=0, y=1, z=0),
            #     endEffectorPositions=[dict(x=0, y=0, z=0) for _ in range(self.n_legs)],
            # )

            if request["type"] == "Initial":
                response["id"] = 1
            elif request["type"] == "Stop":
                self.stop_thread = True
            elif request["type"] == "Command":
                pass
            elif request["type"] == "Reset":
                self.step = 0
            else:
                raise NotImplementedError()

            #  Send reply back to client
            response_message = json.dumps(response)
            # print(f"Sent {response_message}")
            socket.send(response_message.encode("UTF-8"))


def test_gym_env():
    server = FakeServer()
    server.start()
    env = gym.make("SpaceEngineers-WalkingRobot-IK-v0", detach=True, verbose=2)

    check_env(env, warn=True)

    env.reset()
    for _ in range(10):
        _, _, done, _ = env.step(env.action_space.sample())

        if done:
            env.reset()

    env.close()