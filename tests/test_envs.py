import json
import os
from threading import Thread

import gym
import pytest
import zmq
from stable_baselines3.common.env_checker import check_env

import gym_space_engineers  # noqa: F401

# Set test addr
os.environ["SE_SERVER_ADDR"] = "localhost:5566"


class FakeServer(object):
    def __init__(self, n_legs: int = 6):
        super(FakeServer, self).__init__()
        self.stop_thread = False
        self.thread = Thread(target=self.run, daemon=True)
        self.n_legs = n_legs
        self.step = 0
        self.started = False

    def start(self):
        self.thread.start()
        self.started = True

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5566")

        while not self.stop_thread:
            #  Wait for next request from client
            request = json.loads(socket.recv())
            # print(f"Received request: {request}")
            self.step += 1

            if request["type"] == "Initial":
                pass
            elif request["type"] == "Stop":
                self.stop_thread = True
            elif request["type"] == "Command":
                pass
            elif request["type"] == "Reset":
                self.step = 0
            else:
                raise NotImplementedError()

            # Default response
            # Rotated 180deg
            # response = dict(
            #     position=dict(x=1, y=1, z=-self.step),
            #     right=dict(x=1, y=0, z=0),
            #     forward=dict(x=0, y=0, z=-1),
            #     up=dict(x=0, y=1, z=0),
            #     endEffectorPositions=[dict(x=0, y=0, z=0) for _ in range(self.n_legs)],
            #     id=1,
            # )
            # Aligned with z axis
            # response = dict(
            #     position=dict(x=1, y=1, z=self.step),
            #     right=dict(x=-1, y=0, z=0),
            #     forward=dict(x=0, y=0, z=1),
            #     up=dict(x=0, y=1, z=0),
            #     endEffectorPositions=[dict(x=0, y=0, z=0) for _ in range(self.n_legs)],
            #     id=2,
            # )

            legIdentifiers = [f"l{i + 1}" for i in range(self.n_legs // 2)] + [f"r{i + 1}" for i in range(self.n_legs // 2)]
            # Rotated 90deg
            response = dict(
                position=dict(x=self.step, y=1, z=1),
                right=dict(x=0, y=0, z=1),
                forward=dict(x=1, y=0, z=0),
                up=dict(x=0, y=1, z=0),
                endEffectorPositions=[dict(x=0, y=0, z=0) for _ in range(self.n_legs)],
                touchSensors=[False for _ in range(self.n_legs)],
                id=3,
                legIdentifiers=legIdentifiers,
            )

            #  Send reply back to client
            response_message = json.dumps(response)
            # print(f"Sent {response_message}")
            socket.send(response_message.encode("UTF-8"))

        socket.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"symmetric_control": True},
        {"task": "forward_left"},
        {"task": "turn_left"},
        {"task": "turn_left", "randomize_task": True},
        {"symmetric_control": True, "symmetry_type": "per_leg"},
        {"add_end_effector_velocity": True},
        {"randomize_task": True, "randomize_interval": 10},
    ],
)
@pytest.mark.parametrize("n_legs", [4, 6])
def test_gym_env(kwargs, n_legs):
    server = FakeServer(n_legs)
    server.start()
    env = gym.make("SpaceEngineers-WalkingRobot-IK-v2", detach=True, verbose=2, **kwargs)

    check_env(env, warn=True)

    env.reset()
    for _ in range(10):
        _, _, done, _ = env.step(env.action_space.sample())

        if done:
            env.reset()

    env.close()
